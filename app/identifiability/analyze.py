"""Identifiability analysis: Fisher information and parameter correlation.

Demonstrates why a single-schedule calibration cannot uniquely identify
D0 and Q (they are collinear), and how a two-schedule protocol resolves
the ambiguity (BUILD_PLAN §6, figure F8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy

jax.config.update("jax_enable_x64", True)

PARAM_NAMES = ["log_D0", "Q_kJ", "C_pot", "h_m", "eps"]


def _scenario_kwargs(scenario) -> dict:
    th = load_alloy(scenario.alloy)["thermal"]
    return dict(
        schedule_knots=scenario.schedule_knots,
        t_total=scenario.t_total,
        T_init_K=scenario.T_init_K,
        T_quench=scenario.T_quench,
        h_conv=scenario.h_conv,
        k=th["k"],
        rho_cp=th["rho"] * th["cp"],
        half_thickness_m=scenario.size_mm / 2000.0,
        x_half_mm=scenario.x_half_mm,
        carbon_n=scenario.carbon_n,
        carbon_dt=scenario.carbon_dt,
        carbon_mode=scenario.carbon_mode,
        preset=load_alloy(scenario.alloy),
    )


def fisher_information(
    param_vec: np.ndarray,
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma: float = 15.0,
) -> np.ndarray:
    """Approximate Fisher information matrix J^T J / sigma^2.

    J is the Jacobian of the hardness residuals w.r.t. the parameter vector.
    """
    kwargs = _scenario_kwargs(scenario)
    pv = jnp.asarray(param_vec, jnp.float64)
    od = jnp.asarray(obs_depths, jnp.float64)
    oH = jnp.asarray(obs_H, jnp.float64)

    def resid_fn(p):
        log_D0, Q_kJ, C_pot, h_m, eps = p
        out = fast_forward(
            log_D0,
            Q_kJ,
            C_pot,
            h_m,
            eps,
            schedule_knots=kwargs["schedule_knots"],
            t_total=kwargs["t_total"],
            T_init_K=kwargs["T_init_K"],
            T_quench=kwargs["T_quench"],
            h_conv=kwargs["h_conv"],
            k=kwargs["k"],
            rho_cp=kwargs["rho_cp"],
            half_thickness_m=kwargs["half_thickness_m"],
            x_half_mm=kwargs["x_half_mm"],
            carbon_n=kwargs["carbon_n"],
            carbon_dt=kwargs["carbon_dt"],
            carbon_mode=kwargs["carbon_mode"],
            preset=kwargs["preset"],
        )
        H_pred = jnp.interp(od, out["x_mm"], out["H"])
        return H_pred - oH

    J = jax.jacfwd(resid_fn)(pv)
    F = (J.T @ J) / (sigma**2)
    return np.asarray(F)


def correlation_matrix(F: np.ndarray) -> np.ndarray:
    """Convert Fisher information to a correlation matrix."""
    diag = np.sqrt(np.diag(F))
    diag[diag == 0] = 1.0
    corr = F / np.outer(diag, diag)
    return np.clip(corr, -1.0, 1.0)


def identifiability_report(
    param_vec: np.ndarray,
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma: float = 15.0,
) -> dict:
    """Full identifiability analysis for a single schedule.

    Returns Fisher matrix, correlation matrix, and condition number.
    """
    F = fisher_information(param_vec, obs_depths, obs_H, scenario, sigma)
    corr = correlation_matrix(F)
    eigvals = np.linalg.eigvalsh(F)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    return {
        "fisher": F,
        "correlation": corr,
        "condition_number": cond,
        "param_names": PARAM_NAMES,
    }


def two_schedule_comparison(
    param_vec: np.ndarray,
    scenario_a,
    scenario_b,
    obs_depths: np.ndarray,
    obs_H_a: np.ndarray,
    obs_H_b: np.ndarray,
    sigma: float = 15.0,
) -> dict:
    """Compare identifiability with one vs two schedules.

    Returns correlation matrices and condition numbers for both cases.
    """
    single = identifiability_report(param_vec, obs_depths, obs_H_a, scenario_a, sigma)

    # Combined Fisher from both schedules
    F_a = fisher_information(param_vec, obs_depths, obs_H_a, scenario_a, sigma)
    F_b = fisher_information(param_vec, obs_depths, obs_H_b, scenario_b, sigma)
    F_combined = F_a + F_b
    corr_combined = correlation_matrix(F_combined)
    eigvals = np.linalg.eigvalsh(F_combined)
    cond_combined = float(eigvals[-1] / max(eigvals[0], 1e-30))

    return {
        "single_schedule": single,
        "combined": {
            "fisher": F_combined,
            "correlation": corr_combined,
            "condition_number": cond_combined,
            "param_names": PARAM_NAMES,
        },
    }
