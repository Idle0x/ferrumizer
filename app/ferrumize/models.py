"""Fast differentiable forward model for calibration and design loops.

Uses the lumped-capacitance thermal surrogate (V1-validated) instead of the
full PDE to keep NUTS sampling tractable. The full PDE path is used for
simulation, figures, and the cross-AD verification.
"""

from __future__ import annotations

import jax.numpy as jnp

from ferrumizer_physics.carbon import CarburizeConfig, run_carburize
from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    km_fraction,
    ms_andrews,
)
from ferrumizer_physics.thermal import lumped_surface_T


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
) -> dict:
    """Run lumped-thermal -> carbon-diffusion -> hardening.

    All scalar params are JAX arrays so gradients flow end-to-end.
    Returns dict with H profile, ECD, C profile, x grid.
    """
    D0 = jnp.exp(log_D0)
    Q_J = Q_kJ * 1000.0

    # Lumped thermal surrogate: sample surface T at carbon timesteps
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
    f_mart = km_fraction(Ms, T_quench, preset["km_alpha"])
    H = hardness_profile(cout["C_final"], preset, f_mart)
    ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

    return {
        "C_final": cout["C_final"],
        "x_mm": x_mm,
        "Ms": Ms,
        "f_martensite": f_mart,
        "H": H,
        "ecd_mm": ecd,
    }
