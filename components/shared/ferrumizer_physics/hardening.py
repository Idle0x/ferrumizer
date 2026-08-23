"""Hardening stage: JMAK + Scheil additivity, Koistinen-Marburger,
smoothstep hardness mixing, and effective case depth (ECD).

All functions are JAX-compatible and fully differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp


def ms_andrews(C_wt_pct, A: float, b_carbon: float):
    """Andrews (1965) martensite-start temperature in K.

    Ms(C) = A - b_carbon * C,  C in mass-%.
    """
    return A - b_carbon * jnp.asarray(C_wt_pct, dtype=jnp.float64)


def km_fraction(Ms, Tq: float, alpha_km: float):
    """Koistinen-Marburger martensite volume fraction.

    f_M = 1 - exp(-alpha_km * max(Ms - Tq, 0))
    """
    dT = jnp.maximum(jnp.asarray(Ms, dtype=jnp.float64) - Tq, 0.0)
    return 1.0 - jnp.exp(-alpha_km * dT)


def smoothstep(u):
    """C^1 smoothstep  3u^2 - 2u^3  with soft clamp to [0, 1]."""
    uc = jnp.clip(jnp.asarray(u, dtype=jnp.float64), 0.0, 1.0)
    return 3.0 * uc**2 - 2.0 * uc**3


def hardness_profile(C, preset: dict, f_mart=None):
    """Hardness (HV) via smoothstep mixing rule — never a hard clamp.

    H = Hcore + (Hmax - Hcore) * smoothstep((C - Cmin)/(Cideal - Cmin))
    Optionally weighted by martensite fraction (rule of mixtures).
    """
    h = preset["hardness"]
    C = jnp.asarray(C, dtype=jnp.float64)
    u = (C - h["Cmin"]) / (h["Cideal"] - h["Cmin"])
    H = h["Hcore"] + (h["Hmax"] - h["Hcore"]) * smoothstep(u)
    if f_mart is not None:
        H = f_mart * H + (1.0 - f_mart) * h["Hcore"]
    return H


def ecd_from_hardness(H, x_mm, threshold: float = 550.0):
    """Effective case depth (mm): depth at which H crosses *threshold*.

    Fully differentiable: for each segment [i, i+1] the fraction of the
    segment that lies above the threshold is computed via clamped linear
    interpolation, then summed.  For a monotonically decreasing profile this
    reproduces the exact ISO 2639 crossing depth.
    """
    H = jnp.asarray(H, dtype=jnp.float64)
    x_mm = jnp.asarray(x_mm, dtype=jnp.float64)

    H_left = H[:-1]
    H_right = H[1:]
    dx_seg = x_mm[1:] - x_mm[:-1]

    denom = H_left - H_right
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
    frac = (H_left - threshold) / safe_denom
    frac = jnp.clip(frac, 0.0, 1.0)

    return jnp.sum(frac * dx_seg)


def jmak_scheil_fraction(T_history, dt, n_exp, k_ref, T_nose, width):
    """Scheil-additivity JMAK transformed fraction for diffusional phases.

    k(T) = k_ref * exp(-((T - T_nose)/width)^2)   (Gaussian C-curve approx.)
    X    = 1 - exp( -( sum k(T_i) dt )^n )
    """
    T = jnp.asarray(T_history, dtype=jnp.float64)
    k_T = k_ref * jnp.exp(-(((T - T_nose) / width) ** 2))
    integral = jnp.sum(k_T) * dt
    return 1.0 - jnp.exp(-(integral**n_exp))


def run_hardening(
    C_profile, x_mm, T_quench: float, preset: dict, T_history=None, dt: float = 1.0
) -> dict:
    """Full hardening stage: KM martensite + hardness + ECD.

    Parameters
    ----------
    C_profile : (n,) carbon mass-%, surface -> core
    x_mm      : (n,) depth from surface in mm
    T_quench  : quench / room temperature (K)
    preset    : alloy preset dict
    T_history : optional cooling curve (K) for JMAK diffusional fraction
    dt        : time-step of T_history (s)

    Returns dict with Ms, f_martensite, H, ecd_mm (+ optionally X_diff).
    """
    C_profile = jnp.asarray(C_profile, dtype=jnp.float64)
    x_mm = jnp.asarray(x_mm, dtype=jnp.float64)

    ms_cfg = preset["ms"]
    Ms = ms_andrews(C_profile, ms_cfg["A"], ms_cfg["b_carbon"])
    f_mart = km_fraction(Ms, T_quench, preset["km_alpha"])
    H = hardness_profile(C_profile, preset, f_mart)
    ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

    result = {
        "Ms": Ms,
        "f_martensite": f_mart,
        "H": H,
        "ecd_mm": ecd,
    }

    if T_history is not None:
        jmak = preset.get("jmak", {})
        X_diff = jmak_scheil_fraction(
            T_history,
            dt,
            n_exp=jmak.get("n", 2.0),
            k_ref=jmak.get("k_pearlite", 8.5e-9),
            T_nose=jmak.get("T_nose", 823.15),
            width=jmak.get("width", 80.0),
        )
        result["X_diffusional"] = X_diff

    return result
