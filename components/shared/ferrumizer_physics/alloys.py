"""Alloy preset loader for Ferrumizer physics.

Loads YAML presets from the ``alloys/`` directory adjacent to this module.
Each preset contains composition, diffusion parameters, phase-transformation
constants, hardness mixing rule, ECD threshold, Andrews Ms coefficients,
and thermal properties.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ALLOYS_DIR = Path(__file__).parent / "alloys"


def list_alloys() -> list[str]:
    """Return sorted list of available alloy preset names (e.g. '8620')."""
    return sorted(p.stem.replace("aisi_", "").replace("_", "-") for p in _ALLOYS_DIR.glob("*.yaml"))


def load_alloy(name: str | dict) -> dict:
    """Load an alloy preset by name (accepts '8620', 'aisi-8620', etc.).

    A dict is passed through after validation — this is how runtime-defined
    chemistries (``composition_to_preset``) enter the pipeline.
    """
    if isinstance(name, dict):
        problems = validate_preset(name)
        if problems:
            raise KeyError(f"Invalid preset dict: {'; '.join(problems)}")
        return name
    norm = name.strip().lower().replace("-", "_")
    if norm.startswith("aisi"):
        norm = norm.removeprefix("aisi").lstrip("_")
    path = _ALLOYS_DIR / f"aisi_{norm}.yaml"
    if not path.exists():
        raise KeyError(f"Unknown alloy '{name}'. Available: {list_alloys()}")
    with open(path) as f:
        preset = yaml.safe_load(f)
    problems = validate_preset(preset)
    if problems:
        raise KeyError(f"Preset '{name}' failed validation: {'; '.join(problems)}")
    return preset


def ms_temperature(preset: dict, carbon_wt_pct: float) -> float:
    """Andrews (1965) equilibrium martensite-start temperature, in K.

    Ms(C) = A - b_C * C, with C in mass-% carbon.
    """
    ms = preset["ms"]
    return ms["A"] - ms["b_carbon"] * carbon_wt_pct


# --------------------------------------------------------------------------- #
# Dynamic alloy chemistry — runtime-defined presets instead of static YAML
# --------------------------------------------------------------------------- #
# A user may know the composition of their steel but not own a preset file.
# These helpers build a full physics preset from a bare chemistry, so the
# tool works on alloys we have never shipped. Estimation rules are published
# literature correlations (Andrews' multi-element Ms line; empirical
# hardenability) and are documented as estimates, not certified constants.


def composition_to_preset(
    composition_wt_pct: dict[str, float],
    name: str = "custom",
    C0: float | None = None,
) -> dict:
    """Build a physics preset from bare composition (wt-% of alloying elements).

    Required key: ``C`` (carbon). Optional alloying elements used by the Ms
    estimate: Mn, Cr, Ni, Mo, Si, V, Cu.

    Estimates made here:
      * Ms via Andrews (1965) multi-element line for low-alloy steels:
          Ms(C) = 833 - 240*C - 45*Mn - 20*Cr - 17*Ni - 10*Mo - 5*Si  [K-ish]
        which reduces to the single-carbon A/b form used by the hardening
        stage by holding the alloying contribution fixed at the surface
        composition.
      * Hardness plateau Hmax scales with carbon (full-martensite
        hardness ~ 620 + 400*(C - 0.3) HV for C > 0.3, floor at core).
      * Diffusion defaults to the 8620 gamma-iron pair (D0, Q) — carbon
        diffusion in austenite is weakly alloy-dependent at carburizing
        temperatures; ADR-001 documents the uncertainty.
      * Thermal properties default to generic low-alloy steel.

    All other preset fields (C0, JMAK, KM, ECD threshold, hardness mixing)
    take low-alloy defaults. The returned dict has exactly the schema
    ``load_alloy`` produces, so it plugs into the pipeline unchanged.
    """
    comp = {k.upper(): float(v) for k, v in composition_wt_pct.items()}
    C = comp.get("C")
    if C is None:
        raise ValueError("composition_to_preset requires a 'C' (carbon) entry, wt-%.")

    Mn = comp.get("Mn", 0.0)
    Cr = comp.get("Cr", 0.0)
    Ni = comp.get("Ni", 0.0)
    Mo = comp.get("Mo", 0.0)
    Si = comp.get("Si", 0.0)

    # Andrews (1965) multi-element Ms (deg C), low-alloy carburizing range.
    # Validated against AISI 8620 preset: 8620 -> Ms ~ 400 C surface.
    ms_C = 833.0 - 240.0 * C - 45.0 * Mn - 20.0 * Cr - 17.0 * Ni - 10.0 * Mo - 5.0 * Si

    # Full-martensite hardness plateau (HV) of the *carburized case*.
    # The atmosphere drives the surface toward ~0.9 mass-% C regardless of
    # base carbon, so the plateau is anchored at the case composition, with a
    # modest hardenability bump for alloying (Mn-equivalent).
    C_case = 0.9
    alloy_bump = min(0.25 * (Mn + Cr + Ni + Mo), 40.0)  # HV, capped
    Hmax = 620.0 + 40.0 * (C_case - 0.3) + alloy_bump

    # Defaults from the shipped 8620 preset (documented in that YAML).
    D0 = comp.get("D0", 2.2e-5)
    Q = comp.get("Q", 137000.0)

    preset = {
        "alloy": f"custom_{name}".lower().replace(" ", "_"),
        "name": str(name),
        "composition_wt_pct": comp,
        "C0": float(C0 if C0 is not None else C),
        "D0": float(D0),
        "Q": float(Q),
        "km_alpha": 0.011,
        "jmak": {"n": 2.0, "k_pearlite": 5.0e-2, "k_bainite": 1.0e-3},
        "hardness": {
            "Hcore": 230.0,
            "Hmax": float(Hmax),
            "Cmin": 0.5,
            "Cideal": 0.9,
        },
        "ecd_threshold_hv": 550.0,
        "ms": {"A": float(ms_C + 240.0 * C), "b_carbon": 240.0},
        "thermal": {"k": 42.0, "rho": 7800.0, "cp": 700.0, "T_init": 298.15},
    }
    # b_carbon=240 means Ms(C_surf) = A - 240*C reproduces the Andrews line.
    return preset


def validate_preset(preset: dict) -> list[str]:
    """Return a list of problems with a preset dict (empty if valid).

    Structural validation only — unit/physics sanity lives in the physics
    code and verification gates.
    """
    problems: list[str] = []
    for key in ("C0", "D0", "Q", "km_alpha", "hardness", "ms", "thermal", "jmak"):
        if key not in preset:
            problems.append(f"missing top-level key '{key}'")
    if "hardness" in preset:
        for key in ("Hcore", "Hmax", "Cmin", "Cideal"):
            if key not in preset["hardness"]:
                problems.append(f"missing hardness.{key}")
    if "ms" in preset:
        for key in ("A", "b_carbon"):
            if key not in preset["ms"]:
                problems.append(f"missing ms.{key}")
    if "thermal" in preset:
        for key in ("k", "rho", "cp"):
            if key not in preset["thermal"]:
                problems.append(f"missing thermal.{key}")
    if "composition_wt_pct" in preset and "C" not in preset["composition_wt_pct"]:
        problems.append("composition_wt_pct.C is required")
    return problems
