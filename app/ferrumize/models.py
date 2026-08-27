"""Fast differentiable forward model for calibration and design loops.

Uses the lumped-capacitance thermal surrogate for the SOAK (V1-validated,
Bi ~ 0.004 in the furnace) to keep NUTS sampling tractable, but the QUENCH
uses the exact same spatial 1-D conduction solver and per-depth Scheil-JMAK
as :func:`ferrumize.pipeline.FerrumizerPipeline.forward`.

This is the review-driven unification: the calibration engine must produce
posteriors that agree with what the Virtual Furnace predicts. If calibration
used an instantaneous/lumped quench while the app showed spatial quench
results, a plant engineer calibrating against an oil-quenched traverse would
get a posterior that does not reproduce their data when replayed in the app.
The soak surrogate is identical physics at Bi << 1; the quench is not
surrogated at all.
"""

from __future__ import annotations

import jax.numpy as jnp

from ferrumizer_physics.carbon import CarburizeConfig, run_carburize
from ferrumizer_physics.hardening import (
    QUENCH_MEDIA_H,
    ecd_from_hardness,
    hardness_profile,
    km_fraction,
    ms_andrews,
    quench_fractions_depth,
)
from ferrumizer_physics.thermal import (
    ThermalConfig,
    lumped_surface_T,
    run_quench_thermal,
    stability_dt,
)


def _quench_fractions_spatial(
    C_final,
    Ms,
    preset,
    T0_uniform_K,
    *,
    geometry: str,
    size_mm: float,
    thermal_n: int,
    alpha: float,
    k: float,
    T_quench: float,
    h_quench: float,
    t_quench_total: float,
    n_carbon: int,
    x_half_mm: float,
):
    """Spatial quench (identical physics to pipeline.forward).

    Initial field is uniform at the end-of-soak lumped temperature (Bi << 1
    makes the part essentially isothermal at that point). Returns the same
    dict shape as ``quench_fractions_depth`` plus the cooling history.
    """
    x, dx = _grid(geometry, size_mm, thermal_n)
    qcfg = ThermalConfig(
        geometry=geometry,
        size_mm=size_mm,
        n=thermal_n,
        dt=_stability_dt(alpha, dx),
        t_total=t_quench_total,
        alpha=alpha,
        h=h_quench,
        eps=0.0,
        k=k,
        T_init_K=T_quench,
        sample_every=max(1, thermal_n // 8),
    )
    # Initial field is uniform at the end-of-soak temperature. Keep it a
    # JAX array (never float() it): the end-of-soak temperature depends on
    # eps, so converting to a Python float would break gradient flow through
    # the quench stage.
    T0 = jnp.full(thermal_n, T0_uniform_K, dtype=jnp.float64)
    qt = run_quench_thermal(T0, qcfg, T_quench, h_quench)

    # sample the quench history at the carbon depth nodes (surface->core)
    qx = jnp.asarray(qt["x"], jnp.float64)
    depths_m = jnp.linspace(0.0, x_half_mm / 1000.0, n_carbon)  # 0=surface
    if geometry == "slab":
        # thermal slab grid runs -L/2..+L/2; the right half runs
        # center->surface. FLIP so column 0 = surface.
        thalf = qt["T"][:, qx.shape[0] // 2 :][:, ::-1]
        half_depths = jnp.linspace(0.0, x_half_mm / 1000.0, thalf.shape[1])
    else:
        # cylinder grid runs 0 (center) -> R (surface); flip to surface-first
        thalf = qt["T"][:, ::-1]
        half_depths = jnp.linspace(0.0, x_half_mm / 1000.0, thalf.shape[1])
    idx = jnp.clip(
        jnp.searchsorted(half_depths, depths_m, side="right") - 1,
        0,
        half_depths.shape[0] - 2,
    )
    frac = (depths_m - half_depths[idx]) / jnp.maximum(
        half_depths[idx + 1] - half_depths[idx], 1e-15
    )
    T_depths = thalf[:, idx] * (1.0 - frac) + thalf[:, idx + 1] * frac  # (M, n_carbon)
    dt_q = float(qcfg.dt * qcfg.sample_every)
    qf = quench_fractions_depth(C_final, Ms, preset, T_depths, dt=dt_q, T_quench=T_quench)
    qf["cooling_history"] = qt["T"]
    qf["cooling_x"] = qx
    return qf


def _grid(geometry: str, size_mm: float, n: int):
    from ferrumizer_physics.thermal import cyl_grid, slab_grid

    if geometry == "slab":
        return slab_grid(size_mm, n)
    return cyl_grid(size_mm, n)


def _stability_dt(alpha: float, dx: float) -> float:
    return stability_dt(float(alpha), float(dx), 0.45)


def fast_forward(
    log_D0: jnp.ndarray,
    Q_kJ: jnp.ndarray,
    C_pot: jnp.ndarray,
    h_m: jnp.ndarray,
    eps: jnp.ndarray,
    *,
    schedule_knots: jnp.ndarray,
    t_total: float,
    T_init_K: float,
    T_quench: float,
    h_conv: float,
    k: float,
    rho_cp: float,
    half_thickness_m: float,
    x_half_mm: float,
    carbon_n: int,
    carbon_dt: float,
    carbon_mode: str,
    preset: dict,
    n_T_samples: int = 200,
    # optional finite-rate quench (None = instantaneous, legacy path)
    quench_medium: str | None = None,
    quench_temp_K: float = 333.15,
    quench_agitation: float = 0.5,
    quench_time_s: float = 600.0,
    quench_n_samples: int = 120,
    geometry: str = "cylinder",
    size_mm: float | None = None,
    thermal_n: int = 41,
) -> dict:
    """Run lumped-soak -> carbon-diffusion -> spatial-quench -> hardening.

    All scalar params are JAX arrays so gradients flow end-to-end.
    Returns dict with H profile, ECD, C profile, x grid.

    Quench: when ``quench_medium`` is set, the quench stage uses the SAME
    spatial conduction solve as the app (not the lumped single-curve
    approximation). ``size_mm`` defaults to ``2*half_thickness_m*1000`` so
    the characteristic length matches the cylinder/slab convention.
    """
    D0 = jnp.exp(log_D0)
    Q_J = Q_kJ * 1000.0

    # Lumped thermal surrogate for the soak: sample surface T at carbon timesteps
    t_samples = jnp.linspace(0.0, t_total, n_T_samples)
    T_surf = lumped_surface_T(
        schedule_knots,
        t_samples,
        h_conv,
        eps,
        k,
        rho_cp,
        half_thickness_m,
        T_init_K,
    )

    # Carbon diffusion
    ccfg = CarburizeConfig(
        n=carbon_n,
        dt=carbon_dt,
        t_total=t_total,
        mode=carbon_mode,
        sample_every=max(1, int(t_total / carbon_dt / n_T_samples)),
    )
    cout = run_carburize(
        T_surf,
        ccfg,
        C0=jnp.asarray(preset["C0"], jnp.float64),
        D0=D0,
        Q_J=Q_J,
        C_pot=C_pot,
        hm=h_m,
        x_half_mm=x_half_mm,
    )

    # Hardening
    x_mm = jnp.linspace(0.0, x_half_mm, carbon_n)
    Ms = ms_andrews(cout["C_final"], preset["ms"]["A"], preset["ms"]["b_carbon"])
    if quench_medium is not None:
        if size_mm is None:
            size_mm = 2.0 * half_thickness_m * 1000.0
        qf = _quench_fractions_spatial(
            cout["C_final"],
            Ms,
            preset,
            T_surf[-1],
            geometry=geometry,
            size_mm=float(size_mm),
            thermal_n=int(thermal_n),
            alpha=float(k / rho_cp),
            k=float(k),
            T_quench=float(quench_temp_K),
            h_quench=float(QUENCH_MEDIA_H[quench_medium] * (1.0 + quench_agitation)),
            t_quench_total=float(quench_time_s),
            n_carbon=carbon_n,
            x_half_mm=float(x_half_mm),
        )
        f_mart = qf["f_martensite"]
        H = qf["H"]
    else:
        f_mart = km_fraction(Ms, T_quench, preset["km_alpha"], preset.get("mf_offset_K", 200.0))
        H = hardness_profile(cout["C_final"], preset, f_mart)
    ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

    out = {
        "C_final": cout["C_final"],
        "x_mm": x_mm,
        "Ms": Ms,
        "f_martensite": f_mart,
        "H": H,
        "ecd_mm": ecd,
    }
    if quench_medium is not None:
        out["quench"] = qf
    return out
