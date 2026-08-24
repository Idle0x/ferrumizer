"""V6 — Noiseless planted-parameter recovery.

Generates synthetic hardness + surface-temperature data from *planted*
parameters using the fast forward model (lumped thermal surrogate + carbon
diffusion + hardening), then recovers the parameters via gradient-based
optimization (L-BFGS-B with exact JAX gradients).

Two schedules (900 C and 1000 C soaks) are used jointly — this is the
identifiability protocol (see docs/adr/ADR-002 and figure F8): a single
schedule leaves D0 and Q collinear (Arrhenius compensation), two temperatures
break the degeneracy.

Gate: max relative error across {log D0, Q, C_pot, eps} < 1e-4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy
from ferrumizer_physics.thermal import lumped_surface_T

# Planted ground truth (within prior support)
PLANTED = {
    "log_D0": float(np.log(2.2e-5)),
    "Q_kJ": 137.0,
    "C_pot": 1.0,
    "eps": 0.8,
}

# Two-schedule protocol: different soak temperatures break D0-Q collinearity.
SCHEDULES = [
    {"temps_C": (900.0, 900.0), "label": "low-T soak 900C"},
    {"temps_C": (1000.0, 1000.0), "label": "high-T soak 1000C"},
]

OBS_DEPTHS_MM = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
T_OBS_N = 61
T_OBS_WINDOW_S = 1800.0
SIGMA_T = 2.0  # K
SIGMA_H = 15.0  # HV


def _kwargs_for(alloy: str = "8620", t_total: float = 7200.0) -> dict:
    preset = load_alloy(alloy)
    th = preset["thermal"]
    return dict(
        t_total=t_total,
        T_init_K=298.15,
        T_quench=298.15,
        h_conv=20.0,
        k=th["k"],
        rho_cp=th["rho"] * th["cp"],
        half_thickness_m=16.0 / 2000.0,
        x_half_mm=8.0,
        carbon_n=41,
        carbon_dt=5.0,
        carbon_mode="dirichlet",
        preset=preset,
        n_T_samples=120,
    )


def _schedule_knots(temps_C: tuple, t_total: float) -> jnp.ndarray:
    return jnp.array([[0.0, t_total], list(temps_C)], dtype=jnp.float64)


def _predict_T(param_vec, sched_temps, kwargs):
    """Predicted part surface temperature samples (first 30 min)."""
    _, _, _, eps = param_vec
    knots = _schedule_knots(sched_temps, T_OBS_WINDOW_S)
    t_obs = jnp.linspace(0.0, T_OBS_WINDOW_S, T_OBS_N)
    return lumped_surface_T(
        knots,
        t_obs,
        kwargs["h_conv"],
        eps,
        kwargs["k"],
        kwargs["rho_cp"],
        kwargs["half_thickness_m"],
        kwargs["T_init_K"],
    )


def _predict_H(param_vec, sched_temps, kwargs):
    """Predicted hardness traverse at the observed depths."""
    log_D0, Q_kJ, C_pot, eps = param_vec
    knots = _schedule_knots(sched_temps, kwargs["t_total"])
    out = fast_forward(
        log_D0,
        Q_kJ,
        C_pot,
        jnp.float64(1e-4),  # h_m unused in dirichlet mode
        eps,
        schedule_knots=knots,
        **kwargs,
    )
    return jnp.interp(jnp.asarray(OBS_DEPTHS_MM, jnp.float64), out["x_mm"], out["H"])


def generate_planted_data(noise_sigma: float = 0.0, seed: int = 0) -> dict:
    """Synthetic observations from planted parameters, optionally noisy."""
    kwargs = _kwargs_for()
    p = [
        jnp.float64(PLANTED["log_D0"]),
        jnp.float64(PLANTED["Q_kJ"]),
        jnp.float64(PLANTED["C_pot"]),
        jnp.float64(PLANTED["eps"]),
    ]
    rng = np.random.default_rng(seed)
    schedules = []
    for sched in SCHEDULES:
        H_clean = np.asarray(_predict_H(p, sched["temps_C"], kwargs))
        T_clean = np.asarray(_predict_T(p, sched["temps_C"], kwargs))
        H_obs = (
            H_clean + rng.normal(0.0, noise_sigma, size=H_clean.shape)
            if noise_sigma > 0
            else H_clean
        )
        T_obs = (
            T_clean + rng.normal(0.0, noise_sigma * 0.1, size=T_clean.shape)
            if noise_sigma > 0
            else T_clean
        )
        schedules.append(
            {
                "label": sched["label"],
                "temps_C": sched["temps_C"],
                "T_obs": T_obs,
                "H_obs": H_obs,
            }
        )
    return {
        "depths_mm": OBS_DEPTHS_MM.copy(),
        "t_obs_s": np.linspace(0.0, T_OBS_WINDOW_S, T_OBS_N),
        "schedules": schedules,
        "planted": dict(PLANTED),
    }


def _loss(param_vec, obs_list, kwargs) -> jnp.ndarray:
    total = jnp.float64(0.0)
    for obs in obs_list:
        T_pred = _predict_T(param_vec, obs["temps_C"], kwargs)
        H_pred = _predict_H(param_vec, obs["temps_C"], kwargs)
        total = total + jnp.sum((T_pred - obs["T_jax"]) ** 2) / (2.0 * SIGMA_T**2)
        total = total + jnp.sum((H_pred - obs["H_jax"]) ** 2) / (2.0 * SIGMA_H**2)
    return total


def run_v6(max_iter: int = 500, noise_sigma: float = 0.0) -> dict:
    """Recover planted parameters from two-schedule data.

    Args:
        max_iter: max L-BFGS iterations.
        noise_sigma: optional Gaussian noise (HV) added to synthetic hardness.
            0 (default) = noiseless recovery for the V6 gate.
    """
    from scipy.optimize import minimize

    kwargs = _kwargs_for()
    data = generate_planted_data(noise_sigma=noise_sigma, seed=0)
    obs_list = [
        {
            "temps_C": s["temps_C"],
            "T_jax": jnp.asarray(s["T_obs"], jnp.float64),
            "H_jax": jnp.asarray(s["H_obs"], jnp.float64),
        }
        for s in data["schedules"]
    ]

    # Start moderately offset from the planted point
    x0 = np.array(
        [
            PLANTED["log_D0"] + 0.2,
            PLANTED["Q_kJ"] + 8.0,
            PLANTED["C_pot"] - 0.1,
            PLANTED["eps"] - 0.2,
        ]
    )

    loss_np = lambda v: float(_loss(jnp.asarray(v), obs_list, kwargs))
    grad_np = lambda v: np.asarray(jax.grad(_loss)(jnp.asarray(v), obs_list, kwargs))

    bounds = [
        (np.log(1e-7), np.log(1e-3)),  # log D0
        (100.0, 200.0),  # Q kJ/mol
        (0.6, 1.4),  # C_pot
        (0.3, 1.0),  # eps
    ]

    res = minimize(
        loss_np,
        x0,
        jac=grad_np,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-16, "gtol": 1e-14},
    )

    names = ["log_D0", "Q_kJ", "C_pot", "eps"]
    recovered = dict(zip(names, [float(v) for v in res.x]))
    rel_err = {k: abs(recovered[k] - PLANTED[k]) / abs(PLANTED[k]) for k in names}
    worst = max(rel_err.values())
    return {
        "planted": dict(PLANTED),
        "recovered": recovered,
        "rel_err": rel_err,
        "max_rel_err": worst,
        "final_loss": float(res.fun),
        "converged": bool(res.success),
        "passed": worst < 1e-4 and bool(res.success),
        "threshold": 1e-4,
    }


if __name__ == "__main__":
    r = run_v6()
    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"V6 [{status}] max_rel_err={r['max_rel_err']:.3e} "
        f"(threshold {r['threshold']:.0e}, converged={r['converged']}, "
        f"loss={r['final_loss']:.3e})"
    )
    for k in r["rel_err"]:
        print(
            f"  {k:>7s} planted={r['planted'][k]:.6g} "
            f"recovered={r['recovered'][k]:.6g} rel_err={r['rel_err'][k]:.2e}"
        )
