"""End-to-end Ferrumizer pipeline.

Composes the three stages — thermal -> carburizing -> hardening — into one
differentiable function. Two execution paths are provided:

* :meth:`FerrumizerPipeline.forward` — pure-JAX path over the shared physics
  library. Fast, fully differentiable; used by calibration, design, figures.
* :meth:`FerrumizerPipeline.forward_containers` — routes the same computation
  through the three Tesseract components via ``tesseract_jax.apply_tesseract``
  so gradients provably cross TWO container boundaries (G1). Used by the
  cross-AD verification (V4) and the composition demo.

Both paths share the same discretization and constants, so their outputs agree
to floating-point tolerance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from ferrumizer_physics.alloys import load_alloy
from ferrumizer_physics.carbon import CarburizeConfig, run_carburize
from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    km_fraction,
    ms_andrews,
)
from ferrumizer_physics.thermal import ThermalConfig, grid, run_thermal, stability_dt

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = REPO_ROOT / "components"


@dataclass
class ProcessParams:
    """Free / calibrated process parameters (the gradient targets)."""

    D0: float = 2.2e-5  # m^2/s (ADR-001)
    Q_kJ: float = 137.0  # kJ/mol [S3]
    C_pot: float = 1.0  # mass-%
    h_m: float = 1.0e-4  # m/s
    eps: float = 0.8  # emissivity


@dataclass
class Scenario:
    """Fixed scenario describing the part, schedule and discretization."""

    alloy: str = "8620"
    geometry: str = "slab"
    size_mm: float = 16.0
    # furnace schedule knots: (time s, setpoint deg C)
    schedule_times: tuple = (0.0, 600.0, 14400.0)
    schedule_temps_C: tuple = (950.0, 950.0, 950.0)
    t_total: float = 14400.0
    T_init_K: float = 298.15
    T_quench: float = 298.15
    h_conv: float = 20.0
    convective_model: str = "constant_h"
    # discretization
    thermal_n: int = 161
    thermal_sample_every: int = 400
    carbon_n: int = 81
    carbon_dt: float = 2.0
    carbon_sample_every: int = 300
    carbon_mode: str = "dirichlet"

    @property
    def x_half_mm(self) -> float:
        return self.size_mm / 2.0

    @property
    def schedule_knots(self) -> jnp.ndarray:
        return jnp.stack(
            [
                jnp.asarray(self.schedule_times, jnp.float64),
                jnp.asarray(self.schedule_temps_C, jnp.float64),
            ]
        )


def thermal_dt(sc: Scenario, alpha: float) -> float:
    """Stability-limited thermal time step for the scenario grid."""
    _, dx = grid(sc.geometry, sc.size_mm, sc.thermal_n)
    return stability_dt(alpha, dx, 0.45)


class FerrumizerPipeline:
    """Composes the three stages. Holds alloy preset and scenario."""

    def __init__(self, scenario: Scenario | None = None, params: ProcessParams | None = None):
        self.scenario = scenario or Scenario()
        self.params = params or ProcessParams()
        self.preset = load_alloy(self.scenario.alloy)

    # ------------------------------------------------------------------ #
    # pure-JAX forward (fast path)
    # ------------------------------------------------------------------ #
    def forward(self, params: ProcessParams | None = None) -> dict:
        """Run thermal -> carburize -> hardening entirely in JAX."""
        p = params or self.params
        sc = self.scenario
        preset = self.preset
        th = preset["thermal"]
        alpha = th["k"] / (th["rho"] * th["cp"])
        dt = thermal_dt(sc, alpha)

        tcfg = ThermalConfig(
            geometry=sc.geometry,
            size_mm=sc.size_mm,
            n=sc.thermal_n,
            dt=dt,
            t_total=sc.t_total,
            alpha=alpha,
            h=sc.h_conv,
            eps=p.eps,
            k=th["k"],
            T_init_K=sc.T_init_K,
            sample_every=sc.thermal_sample_every,
        )
        tout = run_thermal(sc.schedule_knots, tcfg)

        ccfg = CarburizeConfig(
            n=sc.carbon_n,
            dt=sc.carbon_dt,
            t_total=sc.t_total,
            mode=sc.carbon_mode,
            sample_every=sc.carbon_sample_every,
        )
        cout = run_carburize(
            tout["Ts"],
            ccfg,
            C0=jnp.asarray(preset["C0"], jnp.float64),
            D0=jnp.asarray(p.D0, jnp.float64),
            Q_J=jnp.asarray(p.Q_kJ * 1000.0, jnp.float64),
            C_pot=jnp.asarray(p.C_pot, jnp.float64),
            hm=jnp.asarray(p.h_m, jnp.float64),
            x_half_mm=sc.x_half_mm,
        )

        n = cout["n"]
        x_mm = jnp.linspace(0.0, sc.x_half_mm, n)
        Ms = ms_andrews(cout["C_final"], preset["ms"]["A"], preset["ms"]["b_carbon"])
        f_mart = km_fraction(Ms, sc.T_quench, preset["km_alpha"])
        H = hardness_profile(cout["C_final"], preset, f_mart)
        ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

        return {
            "thermal": tout,
            "carbon": cout,
            "x_mm": x_mm,
            "Ms": Ms,
            "f_martensite": f_mart,
            "H": H,
            "ecd_mm": ecd,
        }

    def ecd(self, params: ProcessParams | None = None):
        """Scalar effective case depth (mm) — the key differentiable objective."""
        return self.forward(params)["ecd_mm"]

    # ------------------------------------------------------------------ #
    # container path (gradients cross two Tesseract boundaries)
    # ------------------------------------------------------------------ #
    def _clients(self):
        from tesseract_core import Tesseract

        def _open(stage: str):
            env = f"FERRUMIZER_{stage.upper().replace('-', '_')}_IMAGE"
            image = os.environ.get(env)
            if image:
                return Tesseract.from_image(image)
            return Tesseract.from_tesseract_api(str(COMPONENTS / stage / "tesseract_api.py"))

        return _open("thermal-stage"), _open("carburizing-stage"), _open("hardening-stage")

    def forward_containers(self, params: ProcessParams | None = None):
        """Compose the three Tesseracts with ``apply_tesseract``.

        Gradients enter at {D0, Q, C_pot, h_m, eps} and flow through TWO
        container boundaries (thermal->carburizing, carburizing->hardening).
        """
        from tesseract_jax import apply_tesseract

        p = params or self.params
        sc = self.scenario
        preset = self.preset
        th = preset["thermal"]
        thermal, carburizing, hardening = self._clients()

        with thermal, carburizing, hardening:
            t_out = apply_tesseract(
                thermal,
                {
                    "geometry": sc.geometry,
                    "size_mm": sc.size_mm,
                    "n": sc.thermal_n,
                    "dt": thermal_dt(sc, th["k"] / (th["rho"] * th["cp"])),
                    "t_total": sc.t_total,
                    "schedule_times": jnp.asarray(sc.schedule_times, jnp.float64),
                    "schedule_temps_C": jnp.asarray(sc.schedule_temps_C, jnp.float64),
                    "k": th["k"],
                    "rho": th["rho"],
                    "cp": th["cp"],
                    "eps": jnp.asarray(p.eps, jnp.float64),
                    "h": sc.h_conv,
                    "convective_model": sc.convective_model,
                    "T_init_K": sc.T_init_K,
                    "sample_every": sc.thermal_sample_every,
                },
            )
            c_out = apply_tesseract(
                carburizing,
                {
                    "T_surface_history": t_out["T_surface"],
                    "x_half_mm": sc.x_half_mm,
                    "n": sc.carbon_n,
                    "dt": sc.carbon_dt,
                    "t_total": sc.t_total,
                    "mode": sc.carbon_mode,
                    "C0": preset["C0"],
                    "D0": jnp.asarray(p.D0, jnp.float64),
                    "Q_kJ_per_mol": jnp.asarray(p.Q_kJ, jnp.float64),
                    "C_pot": jnp.asarray(p.C_pot, jnp.float64),
                    "h_m": jnp.asarray(p.h_m, jnp.float64),
                    "sample_every": sc.carbon_sample_every,
                },
            )
            h_out = apply_tesseract(
                hardening,
                {
                    "C_profile": c_out["C_final"],
                    "x_mm": c_out["x_mm"],
                    "T_quench": jnp.asarray(sc.T_quench, jnp.float64),
                    "ms_A": preset["ms"]["A"],
                    "ms_b_carbon": preset["ms"]["b_carbon"],
                    "km_alpha": preset["km_alpha"],
                    "Hcore": preset["hardness"]["Hcore"],
                    "Hmax": preset["hardness"]["Hmax"],
                    "Cmin": preset["hardness"]["Cmin"],
                    "Cideal": preset["hardness"]["Cideal"],
                    "ecd_threshold_hv": preset["ecd_threshold_hv"],
                },
            )
            return {"thermal": t_out, "carbon": c_out, "hardening": h_out}
