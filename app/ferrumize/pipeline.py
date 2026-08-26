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
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp

from ferrumizer_physics.alloys import load_alloy
from ferrumizer_physics.carbon import CarburizeConfig, run_carburize
from ferrumizer_physics.hardening import (
    QUENCH_MEDIA_H,
    ecd_from_hardness,
    hardness_profile,
    km_fraction,
    ms_andrews,
    quench_fractions,
    quench_fractions_depth,
)
from ferrumizer_physics.thermal import (
    ThermalConfig,
    grid,
    run_quench_thermal,
    run_thermal,
    stability_dt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = REPO_ROOT / "components"
if str(COMPONENTS) not in sys.path:
    sys.path.insert(0, str(COMPONENTS))


# --------------------------------------------------------------------------- #
# JIT memoization for the solver stages.
#
# The PDE stages (thermal, carbon, quench) are `jax.lax.scan`-based; calling
# them without jit re-traces and re-dispatches on every call (~45 s each).
# These wrappers close over the immutable config so JAX's trace cache is keyed
# by (config, input shapes) and repeat calls reuse the compiled XLA program.
# lru_cache on the *factory* means identical configs return the same compiled
# function object — the actual cache hit.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _jit_run_thermal(geometry: str, size_mm: float, n: int, dt: float,
                     t_total: float, alpha: float, h: float, eps: float,
                     k: float, T_init_K: float, sample_every: int):
    cfg = ThermalConfig(
        geometry=geometry, size_mm=size_mm, n=n, dt=dt, t_total=t_total,
        alpha=alpha, h=h, eps=eps, k=k, T_init_K=T_init_K,
        sample_every=sample_every,
    )
    return jax.jit(lambda knots: run_thermal(knots, cfg))


@lru_cache(maxsize=64)
def _jit_quench_thermal(geometry: str, size_mm: float, n: int, dt: float,
                        t_total: float, alpha: float, h: float, k: float,
                        T_init_K: float, sample_every: int):
    cfg = ThermalConfig(
        geometry=geometry, size_mm=size_mm, n=n, dt=dt, t_total=t_total,
        alpha=alpha, h=h, eps=0.0, k=k, T_init_K=T_init_K,
        sample_every=sample_every,
    )
    return jax.jit(lambda T0_field, T_bath_K, h_quench: run_quench_thermal(
        T0_field, cfg, T_bath_K, h_quench
    ))


# Cache of jitted quench-fractions programs, keyed by preset id + geometry.
# A fresh (id, geometry, sizes, T_quench) tuple compiles one program; the
# preset dict is captured in the closure (static Python data, never traced).
_QUENCH_FRACTIONS_CACHE: dict[tuple, Callable] = {}


def _jit_quench_fractions(preset: dict, geometry: str, n_thermal: int,
                          n_carbon: int, size_mm: float, T_quench: float,
                          dt: float):
    """Jitted slice/interp/phase block for the spatial quench.

    The quench thermal solve returns a (M, n_thermal) history; we flip the
    geometry so column 0 = surface, interpolate onto the carbon depth nodes,
    then run the per-depth Scheil-JMAK + KM + phase-specific hardness mixing
    in one compiled unit. Compiling the slice/interp/phase chain together is
    what keeps repeat calls fast (per-op dispatch on the 43 s quench solve
    was the bottleneck). ``dt`` is the sampling interval of the quench
    history (qcfg.dt * qcfg.sample_every) — it feeds the Scheil integral.
    """
    key = (id(preset), geometry, n_thermal, n_carbon, size_mm, T_quench, dt)
    fn = _QUENCH_FRACTIONS_CACHE.get(key)
    if fn is None:

        def run(T_history, qx, C_final, Ms):
            T = T_history  # (M, n_thermal)
            depths_m = jnp.linspace(0.0, size_mm / 2000.0, n_carbon)  # 0=surface
            if geometry == "slab":
                # thermal slab grid runs -L/2..+L/2; the right half runs
                # center->surface. FLIP so column 0 = surface.
                thalf = T[:, qx.shape[0] // 2 :][:, ::-1]
                half_depths = jnp.linspace(0.0, size_mm / 2000.0, thalf.shape[1])
            elif geometry == "rod":
                # Jominy axial grid runs 0 (quench end) -> L (far end);
                # column 0 is already the quenched face (surface).
                thalf = T
                half_depths = jnp.linspace(0.0, size_mm / 2000.0, thalf.shape[1])
            else:
                # cylinder grid runs 0 (center) -> R (surface); flip so
                # column 0 = surface, columns go surface -> center.
                thalf = T[:, ::-1]
                half_depths = jnp.linspace(0.0, size_mm / 2000.0, thalf.shape[1])
            # interpolate the (M, n_thermal_half) history onto the carbon
            # depth nodes (surface->core), vectorized across time rows
            idx = jnp.clip(
                jnp.searchsorted(half_depths, depths_m, side="right") - 1, 0,
                half_depths.shape[0] - 2,
            )
            frac = (depths_m - half_depths[idx]) / jnp.maximum(
                half_depths[idx + 1] - half_depths[idx], 1e-15
            )
            T_depths = thalf[:, idx] * (1.0 - frac) + thalf[:, idx + 1] * frac  # (M, n_carbon)
            return quench_fractions_depth(C_final, Ms, preset, T_depths, dt=dt, T_quench=T_quench)

        fn = jax.jit(run)
        _QUENCH_FRACTIONS_CACHE[key] = fn
    return fn


@lru_cache(maxsize=64)
def _jit_carburize(n: int, dt: float, t_total: float, mode: str,
                   sample_every: int, x_half_mm: float):
    ccfg = CarburizeConfig(
        n=n, dt=dt, t_total=t_total, mode=mode, sample_every=sample_every,
    )
    # x_half_mm is consumed as a CONCRETE Python value inside run_carburize
    # (dx = (x_half_mm/1000)/(n-1)), so it must be a static closure variable,
    # not a jit argument.
    return jax.jit(
        lambda Ts, C0, D0, Q_J, C_pot, hm: run_carburize(
            Ts, ccfg, C0=C0, D0=D0, Q_J=Q_J, C_pot=C_pot, hm=hm,
            x_half_mm=x_half_mm,
        )
    )
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
    # quench model (None = instantaneous quench, legacy path; any medium =
    # finite-rate Newton cooling + Scheil-JMAK diffusional phases)
    quench_medium: str | None = None
    quench_temp_K: float = 333.15  # ~60 C, typical quench oil
    quench_agitation: float = 0.5
    quench_time_s: float = 600.0
    quench_n_samples: int = 120
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

    def __init__(
        self,
        scenario: Scenario | None = None,
        params: ProcessParams | None = None,
        preset: dict | None = None,
    ):
        self.scenario = scenario or Scenario()
        self.params = params or ProcessParams()
        self.preset = preset if preset is not None else load_alloy(self.scenario.alloy)

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
        tout = _jit_run_thermal(
            tcfg.geometry, tcfg.size_mm, tcfg.n, tcfg.dt, tcfg.t_total,
            tcfg.alpha, tcfg.h, tcfg.eps, tcfg.k, tcfg.T_init_K,
            tcfg.sample_every,
        )(sc.schedule_knots)

        ccfg = CarburizeConfig(
            n=sc.carbon_n,
            dt=sc.carbon_dt,
            t_total=sc.t_total,
            mode=sc.carbon_mode,
            sample_every=sc.carbon_sample_every,
        )
        cout = _jit_carburize(
            ccfg.n, ccfg.dt, ccfg.t_total, ccfg.mode, ccfg.sample_every,
            sc.x_half_mm,
        )(
            tout["Ts"],
            jnp.asarray(preset["C0"], jnp.float64),
            jnp.asarray(p.D0, jnp.float64),
            jnp.asarray(p.Q_kJ * 1000.0, jnp.float64),
            jnp.asarray(p.C_pot, jnp.float64),
            jnp.asarray(p.h_m, jnp.float64),
        )

        # n is a Python int (the carbon grid node count), never a device
        # value: passing a traced/device int to linspace forces a 20 s+ host
        # sync per call. Use the scenario's own carbon_n.
        n = sc.carbon_n
        x_mm = jnp.linspace(0.0, sc.x_half_mm, n)
        Ms = ms_andrews(cout["C_final"], preset["ms"]["A"], preset["ms"]["b_carbon"])
        th = preset["thermal"]
        if sc.quench_medium is not None:
            # SPATIAL quench: solve the 1-D conduction PDE with a quench BC
            # starting from the end-of-soak temperature field. This gives
            # depth-resolved cooling rates -> per-depth phase fractions
            # (bainite/pearlite at slow-cooling depths, martensite at fast
            # ones) instead of a single lumped part-average curve.
            alpha = th["k"] / (th["rho"] * th["cp"])
            qcfg = ThermalConfig(
                geometry=sc.geometry,
                size_mm=sc.size_mm,
                n=sc.thermal_n,
                dt=thermal_dt(sc, alpha),  # same stability-bound step
                t_total=sc.quench_time_s,
                alpha=alpha,
                h=sc.h_conv,
                eps=0.0,  # radiation negligible during quench
                k=th["k"],
                T_init_K=sc.quench_temp_K,
                sample_every=max(1, sc.thermal_n // 8),
            )
            T0_field = tout["T_final"]
            qt = _jit_quench_thermal(
                qcfg.geometry, qcfg.size_mm, qcfg.n, qcfg.dt, qcfg.t_total,
                qcfg.alpha, qcfg.h, qcfg.k, qcfg.T_init_K, qcfg.sample_every,
            )(
                T0_field,
                jnp.asarray(sc.quench_temp_K, jnp.float64),
                jnp.asarray(QUENCH_MEDIA_H[sc.quench_medium] * (1.0 + sc.quench_agitation), jnp.float64),
            )
            # Per-depth phase fractions from the spatial cooling history.
            # The slicing/interp/phase block is jitted as one unit so the
            # 43 s quench solve is not re-dispatched per indexing op.
            dt_q = float(qcfg.dt * qcfg.sample_every)
            qf = _jit_quench_fractions(
                preset, sc.geometry, qcfg.n, n, sc.size_mm, sc.quench_temp_K,
                dt_q,
            )(
                qt["T"], jnp.asarray(qt["x"], jnp.float64),
                cout["C_final"], Ms,
            )
            f_mart = qf["f_martensite"]
            H = qf["H"]
            ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])
        else:
            f_mart = km_fraction(Ms, sc.T_quench, preset["km_alpha"], preset.get("mf_offset_K", 200.0))
            H = hardness_profile(cout["C_final"], preset, f_mart)
            ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

        result = {
            "thermal": tout,
            "carbon": cout,
            "x_mm": x_mm,
            "Ms": Ms,
            "f_martensite": f_mart,
            "H": H,
            "ecd_mm": ecd,
        }
        if sc.quench_medium is not None and "qf" in locals():
            # expose the spatial cooling history for the CCT tab
            qf["cooling_history"] = qt["T"]
            qf["cooling_x"] = jnp.asarray(qt["x"], jnp.float64)
            qf["cooling_times_s"] = qt["times_s"]
            result["quench"] = qf
        return result

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
