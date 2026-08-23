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


def load_alloy(name: str) -> dict:
    """Load an alloy preset by name (accepts '8620', 'aisi-8620', etc.)."""
    norm = name.strip().lower().replace("-", "_")
    if norm.startswith("aisi"):
        norm = norm.removeprefix("aisi").lstrip("_")
    path = _ALLOYS_DIR / f"aisi_{norm}.yaml"
    if not path.exists():
        raise KeyError(f"Unknown alloy '{name}'. Available: {list_alloys()}")
    with open(path) as f:
        return yaml.safe_load(f)


def ms_temperature(preset: dict, carbon_wt_pct: float) -> float:
    """Andrews (1965) equilibrium martensite-start temperature, in K.

    Ms(C) = A - b_C * C, with C in mass-% carbon.
    """
    ms = preset["ms"]
    return ms["A"] - ms["b_carbon"] * carbon_wt_pct
