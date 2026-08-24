"""Schedule design loop: gradient-based optimization toward a target ECD.

Optimizes furnace schedule knots (boost / diffuse temperatures and switch
times) so the predicted effective case depth hits a target value, with an
optional gas / energy penalty term to produce the Pareto front (figure F9).

The objective is fully differentiable through the lumped-thermal surrogate
and the JAX carbon + hardening chain, so ``jax.grad`` drives a projected
gradient descent / L-BFGS loop directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy

jax.config.update("jax_enable_x64", True)


def _scenario_kwargs(scenario) -> dict:
    th = load_alloy(scenario.alloy)["thermal"]
    return dict(
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


def ecd_from_schedule(schedule_temps_C, schedule_times, params, kwargs):
    """Predicted ECD (mm) for a given schedule knot set."""
    schedule_knots = jnp.stack(
        [jnp.asarray(schedule_times, jnp.float64), jnp.asarray(schedule_temps_C, jnp.float64)]
    )
    D0 = params["D0"] if "D0" in params else jnp.exp(params["log_D0"])
    out = fast_forward(
        jnp.log(jnp.asarray(D0, jnp.float64)),
        jnp.asarray(params["Q_kJ"], jnp.float64),
        jnp.asarray(params["C_pot"], jnp.float64),
        jnp.asarray(params["h_m"], jnp.float64),
        jnp.asarray(params["eps"], jnp.float64),
        schedule_knots=schedule_knots,
        **kwargs,
    )
    return out["ecd_mm"]


def energy_proxy(schedule_temps_C, schedule_times) -> jnp.ndarray:
    """Rough furnace energy proxy: time-integral of setpoint above ambient.

    Used only as a relative penalty axis for the Pareto front, not as an
    absolute physical energy figure (documented simplification).
    """
    t = jnp.asarray(schedule_times, jnp.float64)
    T = jnp.asarray(schedule_temps_C, jnp.float64)
    dT = jnp.maximum(T - 25.0, 0.0)
    dt_seg = jnp.diff(t, prepend=0.0)
    return jnp.sum(dT * dt_seg)


def design_schedule(
    target_ecd_mm: float,
    scenario,
    params: dict,
    penalty: str = "none",
    penalty_weight: float = 1e-6,
    n_steps: int = 120,
    lr: float = 3.0,
    T_bounds: tuple = (800.0, 1050.0),
    seed: int = 0,
):
    """Optimize boost/diffuse setpoints toward the target ECD.

    Optimizes the interior temperature knots (keeping the first and last
    knots anchored) with projected gradient descent on

        loss = (ECD - target)^2  [+ w * energy_proxy if penalty != 'none']

    Returns a dict with the optimized schedule, achieved ECD, loss trace and
    energy proxy.
    """
    kwargs = _scenario_kwargs(scenario)
    times = list(scenario.schedule_times)
    temps = jnp.asarray(scenario.schedule_temps_C, jnp.float64)

    # Only optimize interior knots; endpoints stay anchored for a valid schedule.
    n_knots = len(times)

    def loss_fn(temps_opt):
        ecd = ecd_from_schedule(temps_opt, times, params, kwargs)
        loss = (ecd - target_ecd_mm) ** 2
        if penalty != "none":
            loss = loss + penalty_weight * energy_proxy(temps_opt, times)
        return loss, ecd

    # --- Feasibility pre-check: what ECD is actually reachable inside bounds? ---
    # A target outside [ECD(T_min), ECD(T_max)] cannot be met by this schedule
    # (same duration, fixed geometry); report it honestly instead of grinding
    # 120 gradient steps toward an unreachable objective.
    t_lo = jnp.full_like(temps, T_bounds[0])
    t_hi = jnp.full_like(temps, T_bounds[1])
    ecd_lo = float(ecd_from_schedule(t_lo, times, params, kwargs))
    ecd_hi = float(ecd_from_schedule(t_hi, times, params, kwargs))
    ecd_min, ecd_max = min(ecd_lo, ecd_hi), max(ecd_lo, ecd_hi)
    feasible = (target_ecd_mm >= ecd_min) and (target_ecd_mm <= ecd_max)
    if not feasible:
        print(
            f"[design] WARNING: target ECD {target_ecd_mm:.3f} mm is OUTSIDE the reachable "
            f"range [{ecd_min:.3f}, {ecd_max:.3f}] mm for this schedule (T in "
            f"[{T_bounds[0]:.0f}, {T_bounds[1]:.0f}] C, {len(times)} knots, "
            f"t_total={scenario.t_total:.0f} s). "
            "Optimizer will head for the nearest bound and report the shortfall."
        )

    loss_trace = []
    ecd_trace = []
    cur = temps
    tol = 1e-6  # early stop: |ECD - target| within tolerance (mm)
    for step in range(n_steps):
        (loss, ecd), g = jax.value_and_grad(loss_fn, has_aux=True)(cur)
        loss_trace.append(float(loss))
        ecd_trace.append(float(ecd))
        if step % 20 == 0 or step == n_steps - 1:
            print(
                f"[design] step {step:3d}/{n_steps}: ECD={float(ecd):.4f} mm "
                f"(target {target_ecd_mm:.3f}) loss={float(loss):.3e}"
            )
        # early stop when converged (respecting penalty term)
        if abs(float(ecd) - target_ecd_mm) < tol and penalty == "none":
            print(f"[design] converged at step {step}.")
            break
        # Normalized gradient step: raw dECD/dT is physically tiny
        # (~1e-3 mm/K), so plain GD crawls. Normalizing turns `lr` into a
        # temperature step per iteration (units of K), which converges in
        # tens of steps while keeping the gradient direction doing the work.
        gnorm = jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)
        g_unit = g / gnorm
        # update only interior knots
        if n_knots > 2:
            interior = cur[1:-1] - lr * g_unit[1:-1]
            interior = jnp.clip(interior, T_bounds[0], T_bounds[1])
            cur = cur.at[1:-1].set(interior)
        else:
            cur = jnp.clip(cur - lr * g_unit, T_bounds[0], T_bounds[1])

    final_ecd = float(ecd_from_schedule(cur, times, params, kwargs))
    return {
        "schedule_times": times,
        "schedule_temps_C": [float(x) for x in cur],
        "target_ecd_mm": target_ecd_mm,
        "achieved_ecd_mm": final_ecd,
        "reachable_range_mm": [ecd_min, ecd_max],
        "feasible": feasible,
        "loss_trace": [float(x) for x in loss_trace],
        "ecd_trace": [float(x) for x in ecd_trace],
        "energy_proxy": float(energy_proxy(cur, times)),
        "penalty": penalty,
    }


def pareto_front(
    target_ecd_mm: float,
    scenario,
    params: dict,
    weights: np.ndarray | None = None,
    n_steps: int = 100,
    seed: int = 0,
):
    """Sweep penalty weights to trace the ECD-vs-energy Pareto front (F9)."""
    if weights is None:
        weights = np.array([0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
    points = []
    for w in weights:
        penalty = "none" if w == 0.0 else "energy"
        res = design_schedule(
            target_ecd_mm,
            scenario,
            params,
            penalty=penalty,
            penalty_weight=float(w),
            n_steps=n_steps,
            seed=seed,
        )
        points.append(
            {
                "weight": float(w),
                "ecd_mm": res["achieved_ecd_mm"],
                "energy_proxy": res["energy_proxy"],
                "temps_C": res["schedule_temps_C"],
            }
        )
    return points
