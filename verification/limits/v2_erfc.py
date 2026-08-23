"""V2 — Semi-infinite erfc diffusion overlay.

For an isothermal soak with constant surface concentration (Dirichlet), the
analytic Crank solution is

    C(x, t) = C0 + (Cs - C0) * erfc( x / (2 sqrt(D t)) )

where x is measured inward from the surface. The FD solver must reproduce it.
The normalized L2 difference over the profile must be < 1e-3.

Reference: Crank, "The Mathematics of Diffusion" (constant-surface-concentration
semi-infinite solution); see BUILD_PLAN Appendix A [S8].
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.carbon import (
    CarburizeConfig,
    D_of_T_np,
    erfc_reference,
    run_carburize_numpy,
)


def run_v2(
    T_C: float = 950.0,
    t_total: float = 14400.0,  # 4 h soak
    x_half_mm: float = 8.0,
    n: int = 161,
    dt: float = 20.0,
    D0: float = 2.2e-5,
    Q_J: float = 137000.0,
    C0: float = 0.2,
    Cs: float = 1.0,
) -> dict:
    T_K = T_C + 273.15
    D = float(D_of_T_np(D0, Q_J, T_K))

    # constant temperature history (isothermal)
    n_samples = 64
    T_hist = np.full(n_samples, T_K)

    cfg = CarburizeConfig(n=n, dt=dt, t_total=t_total, mode="dirichlet", sample_every=10**6)
    out = run_carburize_numpy(
        T_hist, cfg, C0=C0, D0=D0, Q_J=Q_J, C_pot=Cs, hm=1e-4, x_half_mm=x_half_mm
    )
    C_num = out["C_final"]  # surface -> core

    # analytic reference at the same node depths from surface
    x_mm = np.linspace(0.0, x_half_mm, n)
    C_ref = erfc_reference(x_mm, t_total, D, Cs, C0)

    # restrict comparison to the semi-infinite-valid region (x <= 4*sqrt(D t))
    valid = x_mm <= 4.0 * np.sqrt(D * t_total) * 1000.0
    diff = np.linalg.norm(C_num[valid] - C_ref[valid])
    norm = np.linalg.norm(C_ref[valid])
    nl2 = float(diff / norm)
    return {
        "norm_l2": nl2,
        "passed": nl2 < 1e-3,
        "threshold": 1e-3,
        "D_m2_s": D,
        "x_mm": x_mm,
        "C_num": C_num,
        "C_ref": C_ref,
    }


if __name__ == "__main__":
    r = run_v2()
    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"V2 [{status}] norm_l2={r['norm_l2']:.3e} (threshold {r['threshold']:.0e}, "
        f"D={r['D_m2_s']:.3e} m^2/s)"
    )
