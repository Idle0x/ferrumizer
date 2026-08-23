"""V4 — Cross-AD agreement between the FD (NumPy) box and the JAX twin.

For the carburizing stage the *apply* path uses a plain NumPy explicit-FD
stepper while the *gradient* path uses a numerically identical JAX twin.
This test verifies that gradients of a scalar objective (sum of final carbon
profile) with respect to {D0, Q, C_pot, h_m} computed via:

  (a) central finite differences on the NumPy apply, and
  (b) JAX reverse-mode AD through the twin,

agree to within a relative tolerance of 1e-3 in the infinity norm.

Gate: max_i |g_FD[i] - g_JAX[i]| / max_i |g_JAX[i]|  <  1e-3
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.carbon import (
    CarburizeConfig,
    run_carburize,
    run_carburize_numpy,
)


def _objective_numpy(T_hist, cfg, C0, D0, Q_J, C_pot, hm, x_half_mm):
    out = run_carburize_numpy(T_hist, cfg, C0, D0, Q_J, C_pot, hm, x_half_mm)
    return float(np.sum(out["C_final"]))


def _objective_jax(T_hist, cfg, C0, D0, Q_J, C_pot, hm, x_half_mm):
    out = run_carburize(T_hist, cfg, C0, D0, Q_J, C_pot, hm, x_half_mm)
    return jnp.sum(out["C_final"])


def run_v4(
    T_C: float = 950.0,
    t_total: float = 3600.0,
    x_half_mm: float = 8.0,
    n: int = 41,
    dt: float = 0.5,
    C0: float = 0.2,
    D0: float = 2.2e-5,
    Q_J: float = 137000.0,
    C_pot: float = 1.0,
    hm: float = 1e-4,
    fd_rel_step: float = 1e-5,
) -> dict:
    T_K = T_C + 273.15
    n_samples = 64
    T_hist_np = np.full(n_samples, T_K)
    T_hist_jx = jnp.asarray(T_hist_np)

    cfg = CarburizeConfig(n=n, dt=dt, t_total=t_total, mode="mass_transfer", sample_every=10**6)

    params = {"D0": D0, "Q_J": Q_J, "C_pot": C_pot, "hm": hm}
    g_fd = {}
    g_jax = {}

    # --- finite-difference gradients (NumPy apply) ---
    for name, val in params.items():
        step = abs(val) * fd_rel_step if val != 0 else fd_rel_step
        kw = dict(params)
        kw_plus = dict(kw)
        kw_plus[name] = val + step
        kw_minus = dict(kw)
        kw_minus[name] = val - step
        f_plus = _objective_numpy(T_hist_np, cfg, C0, **kw_plus, x_half_mm=x_half_mm)
        f_minus = _objective_numpy(T_hist_np, cfg, C0, **kw_minus, x_half_mm=x_half_mm)
        g_fd[name] = (f_plus - f_minus) / (2.0 * step)

    # --- JAX reverse-mode gradients (JAX twin) ---
    def obj(D0_, Q_J_, C_pot_, hm_):
        return _objective_jax(
            T_hist_jx,
            cfg,
            jnp.float64(C0),
            D0_,
            Q_J_,
            C_pot_,
            hm_,
            x_half_mm,
        )

    grads = jax.grad(obj, argnums=(0, 1, 2, 3))(
        jnp.float64(D0), jnp.float64(Q_J), jnp.float64(C_pot), jnp.float64(hm)
    )
    g_jax = {
        "D0": float(grads[0]),
        "Q_J": float(grads[1]),
        "C_pot": float(grads[2]),
        "hm": float(grads[3]),
    }

    # --- compare ---
    g_fd_arr = np.array([g_fd[k] for k in params])
    g_jax_arr = np.array([g_jax[k] for k in params])
    denom = max(float(np.max(np.abs(g_jax_arr))), 1e-30)
    rel_inf = float(np.max(np.abs(g_fd_arr - g_jax_arr)) / denom)

    return {
        "g_fd": g_fd,
        "g_jax": g_jax,
        "rel_inf_norm": rel_inf,
        "passed": rel_inf < 1e-3,
        "threshold": 1e-3,
    }


if __name__ == "__main__":
    r = run_v4()
    status = "PASS" if r["passed"] else "FAIL"
    print(f"V4 [{status}]  rel_inf={r['rel_inf_norm']:.3e}  (threshold {r['threshold']:.0e})")
    for k in r["g_fd"]:
        print(f"  {k:>6s}  FD={r['g_fd'][k]:+.6e}  JAX={r['g_jax'][k]:+.6e}")


def run_v4_containers() -> dict:
    """V4-container: gradients flow THROUGH the composed Tesseract containers.

    This closes the gap between the twin-level cross-AD check above and the
    submission's headline claim (G1): end-to-end gradients must actually
    traverse the two container boundaries via ``apply_tesseract`` and agree
    with a central-difference reference.

    Gate: d(ECD)/d(eps) via reverse-mode through the containers is finite,
    non-zero, and within 20% of the finite-difference slope.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from ferrumize.config import load_config, scenario_from_config
    from ferrumize.pipeline import FerrumizerPipeline, ProcessParams

    cfg = load_config(str(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "calibration_data.yaml"))
    sc = scenario_from_config(cfg)
    pipe = FerrumizerPipeline(scenario=sc)
    p = pipe.params

    def ecd_from_eps(eps):
        p2 = ProcessParams(D0=p.D0, Q_kJ=p.Q_kJ, C_pot=p.C_pot, h_m=p.h_m, eps=eps)
        return pipe.forward_containers(p2)["hardening"]["ecd_mm"]

    eps_val = jnp.asarray(p.eps, jnp.float64)
    g_ad = float(jax.grad(ecd_from_eps)(eps_val))
    e0 = float(ecd_from_eps(eps_val))
    e1 = float(ecd_from_eps(eps_val + 0.05))
    g_fd = (e1 - e0) / 0.05

    denom = max(abs(g_ad), abs(g_fd), 1e-30)
    rel = abs(g_ad - g_fd) / denom
    passed = bool(
        (g_ad == g_ad)  # not NaN
        and (abs(g_ad) > 1e-12)  # genuinely non-zero: gradients do real work
        and rel < 0.2  # agrees with FD reference within 20%
    )
    return {
        "passed": passed,
        "g_ad": g_ad,
        "g_fd": g_fd,
        "rel_err": rel,
        "ecd_at_eps": e0,
        "threshold": 0.2,
    }


if __name__ == "__main__":
    import sys

    if "--containers" in sys.argv:
        r = run_v4_containers()
        status = "PASS" if r["passed"] else "FAIL"
        print(f"V4-containers [{status}]  g_ad={r['g_ad']:.4e} g_fd={r['g_fd']:.4e} rel={r['rel_err']:.3f}")
    else:
        r = run_v4()
        status = "PASS" if r["passed"] else "FAIL"
        print(f"V4 [{status}]  rel_inf={r['rel_inf_norm']:.3e}  (threshold {r['threshold']:.0e})")
        for k in r["g_fd"]:
            print(f"  {k:>6s}  FD={r['g_fd'][k]:+.6e}  JAX={r['g_jax'][k]:+.6e}")
