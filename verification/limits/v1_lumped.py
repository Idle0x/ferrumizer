"""V1 — Lumped-capacitance exponential-approach limit.

When the Biot number Bi = h*L_c/k is far below 0.1, internal conduction is
effectively instantaneous and the whole part follows the lumped ODE

    T(t) = T_inf - (T_inf - T0) * exp(-t / tau),   tau = rho*cp*L_c / h

with L_c = V/A = half-thickness for a slab exposed on both faces.

Test: a 16 mm steel slab (k = 42 W/m/K gives Bi = h*L_c/k = 0.0038 << 0.1)
heated purely by convection (eps = 0) must track the closed-form lumped
exponential. The core temperature from the full PDE solver is compared
against the analytic lumped solution.

Gate: max relative error < 0.5%.
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.thermal import ThermalConfig, grid, run_thermal, stability_dt


def run_v1(
    size_mm: float = 16.0,
    n: int = 41,
    t_total: float = 600.0,
    h: float = 20.0,
    k: float = 42.0,  # realistic steel -> Bi = 0.0038
    rho: float = 7800.0,
    cp: float = 700.0,
    T_inf_C: float = 950.0,
    T_init_K: float = 298.15,
) -> dict:
    alpha = k / (rho * cp)
    _, dx = grid("slab", size_mm, n)
    dt = stability_dt(alpha, dx, 0.45)

    cfg = ThermalConfig(
        geometry="slab",
        size_mm=size_mm,
        n=n,
        dt=dt,
        t_total=t_total,
        alpha=alpha,
        h=h,
        eps=0.0,
        k=k,
        T_init_K=T_init_K,
        sample_every=max(1, int(round(t_total / dt / 200))),
    )
    knots = np.array([[0.0, t_total], [T_inf_C, T_inf_C]])
    out = run_thermal(knots, cfg)

    times = np.asarray(out["times_s"])
    Tcore = np.asarray(out["Tcore"])
    T_inf = T_inf_C + 273.15
    L_c = (size_mm / 1000.0) / 2.0
    tau = rho * cp * L_c / h
    T_lumped = T_inf - (T_inf - T_init_K) * np.exp(-times / tau)

    rel = np.abs(Tcore - T_lumped) / np.abs(T_lumped)
    max_rel = float(rel[1:].max()) if rel.size > 1 else float(rel.max())
    return {
        "max_rel_err": max_rel,
        "passed": max_rel < 0.005,
        "threshold": 0.005,
        "Bi": h * L_c / k,
        "tau_s": tau,
        "times": times,
        "Tcore": Tcore,
        "T_lumped": T_lumped,
    }


if __name__ == "__main__":
    r = run_v1()
    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"V1 [{status}] max_rel_err={r['max_rel_err']:.4%} "
        f"(threshold {r['threshold']:.1%}, Bi={r['Bi']:.2e})"
    )
