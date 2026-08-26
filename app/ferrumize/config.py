"""Scenario / run configuration loading and validation.

A run config is a YAML file describing the part, furnace schedule, and
discretization. It maps directly onto :class:`ferrumize.pipeline.Scenario`
plus a :class:`ferrumize.pipeline.ProcessParams`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ferrumize.pipeline import ProcessParams, Scenario
from ferrumizer_physics.alloys import list_alloys, load_alloy
from ferrumizer_physics.carbon import D_of_T_np, stability_check_carbon
from ferrumizer_physics.thermal import grid, stability_dt


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def scenario_from_config(cfg: dict) -> Scenario:
    sched = cfg.get("schedule", {})
    return Scenario(
        alloy=str(cfg.get("alloy", "8620")),
        geometry=cfg.get("geometry", "slab"),
        size_mm=float(cfg.get("size_mm", 16.0)),
        schedule_times=tuple(sched.get("times", (0.0, 600.0, 14400.0))),
        schedule_temps_C=tuple(sched.get("temps_C", (950.0, 950.0, 950.0))),
        t_total=float(cfg.get("t_total", 14400.0)),
        T_init_K=float(cfg.get("T_init_K", 298.15)),
        T_quench=float(cfg.get("T_quench", 298.15)),
        h_conv=float(cfg.get("h_conv", 20.0)),
        thermal_n=int(cfg.get("thermal", {}).get("n", 41)),
        thermal_sample_every=int(cfg.get("thermal", {}).get("sample_every", 40)),
        carbon_n=int(cfg.get("carbon", {}).get("n", 81)),
        carbon_dt=float(cfg.get("carbon", {}).get("dt", 2.0)),
        carbon_sample_every=int(cfg.get("carbon", {}).get("sample_every", 300)),
        carbon_mode=cfg.get("carbon", {}).get("mode", "dirichlet"),
    )


def scenario2_from_config(cfg: dict) -> Scenario | None:
    """Second-schedule scenario for the two-schedule identifiability protocol.

    Returns None when the config has no ``schedule2`` block. The second
    schedule shares alloy/geometry/discretization but uses its own knots —
    that is what breaks the D0-Q collinearity (see ADR-002 / figure F8).
    """
    sched2 = cfg.get("schedule2")
    if not sched2:
        return None
    sc = scenario_from_config(cfg)
    sc = Scenario(
        alloy=sc.alloy,
        geometry=sc.geometry,
        size_mm=sc.size_mm,
        schedule_times=tuple(sched2.get("times", sc.schedule_times)),
        schedule_temps_C=tuple(sched2.get("temps_C", sc.schedule_temps_C)),
        t_total=float(sched2.get("t_total", sc.t_total)),
        T_init_K=sc.T_init_K,
        T_quench=sc.T_quench,
        h_conv=sc.h_conv,
        thermal_n=sc.thermal_n,
        thermal_sample_every=sc.thermal_sample_every,
        carbon_n=sc.carbon_n,
        carbon_dt=sc.carbon_dt,
        carbon_sample_every=sc.carbon_sample_every,
        carbon_mode=sc.carbon_mode,
    )
    return sc


def params_from_config(cfg: dict) -> ProcessParams:
    p = cfg.get("params", {})
    return ProcessParams(
        D0=float(p.get("D0", 2.2e-5)),
        Q_kJ=float(p.get("Q_kJ", 137.0)),
        C_pot=float(p.get("C_pot", 1.0)),
        h_m=float(p.get("h_m", 1e-4)),
        eps=float(p.get("eps", 0.8)),
    )


def validate_config(cfg: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []

    alloy = str(cfg.get("alloy", "8620"))
    if alloy not in list_alloys():
        errors.append(f"Unknown alloy '{alloy}'. Available: {list_alloys()}")
        return errors

    preset = load_alloy(alloy)
    geometry = cfg.get("geometry", "slab")
    if geometry not in ("slab", "cylinder"):
        errors.append(f"geometry must be slab|cylinder, got {geometry!r}")

    size_mm = float(cfg.get("size_mm", 16.0))
    if size_mm <= 0:
        errors.append("size_mm must be positive")

    sched = cfg.get("schedule", {})
    times = sched.get("times", [])
    temps = sched.get("temps_C", [])
    if len(times) != len(temps):
        errors.append("schedule.times and schedule.temps_C must have equal length")
    elif len(times) < 2:
        errors.append("schedule needs at least two knots")
    elif times[0] != 0:
        errors.append("schedule.times must start at 0")
    elif any(b < a for a, b in zip(times, times[1:])):
        errors.append("schedule.times must be strictly increasing")

    t_total = float(cfg.get("t_total", 14400.0))
    if times and times[-1] < t_total:
        errors.append("last schedule knot time must be >= t_total")

    th = preset["thermal"]
    alpha = th["k"] / (th["rho"] * th["cp"])
    thermal_n = int(cfg.get("thermal", {}).get("n", 41))
    if thermal_n < 3:
        errors.append("thermal.n must be >= 3")
    else:
        _, dx = grid(geometry, size_mm, thermal_n)
        dt_max = stability_dt(alpha, dx, 0.45)
        # thermal dt is auto-derived in the pipeline; report the limit for info
        cfg.setdefault("_info", {})["thermal_dt_max"] = dt_max

    carbon = cfg.get("carbon", {})
    carbon_n = int(carbon.get("n", 81))
    carbon_dt = float(carbon.get("dt", 2.0))
    mode = carbon.get("mode", "dirichlet")
    if carbon_n < 3:
        errors.append("carbon.n must be >= 3")
    else:
        x_half_mm = size_mm / 2.0
        cdx = (x_half_mm / 1000.0) / (carbon_n - 1.0)
        p = cfg.get("params", {})
        D0 = float(p.get("D0", preset["D0"]))
        Q_J = float(p.get("Q_kJ", preset["Q"] / 1000.0)) * 1000.0
        T_peak = max(temps) + 273.15 if temps else 1223.15
        D_peak = float(D_of_T_np(D0, Q_J, T_peak))
        try:
            stability_check_carbon(D_peak, cdx, carbon_dt, mode, float(p.get("h_m", 1e-4)), 0.45)
        except ValueError as e:
            errors.append(str(e))

    return errors
