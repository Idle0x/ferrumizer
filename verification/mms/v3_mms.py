"""V3 — Method of Manufactured Solutions: observed convergence order.

Verifies that the discrete spatial operators used by BOTH stages converge at
the theoretical order (2nd order in space for the central-difference
Laplacian) including the variable-coefficient case D(T(x)).

The thermal/carbon solvers use the interior stencil

    L_h[u]_i = D_i * (u[i+1] - 2 u[i] + u[i-1]) / dx^2

(non-conservative form, valid when D varies smoothly — it varies here through
temperature, which is near-uniform during the soak). V3 verifies this stencil
converges to D(x) u''(x) at 2nd order for a manufactured u and a spatially
varying D manufactured through T(x). The cylinder operator adds a (1/r) du/dr
term, verified separately.

Gate: observed order >= theoretical (2.0) - 0.15 on a log-log refinement.
"""

from __future__ import annotations

import numpy as np

from ferrumizer_physics.carbon import D_of_T_np


def _slab_operator(u: np.ndarray, D: np.ndarray, dx: float) -> np.ndarray:
    """Interior stencil D_i * (u[i+1]-2u[i]+u[i-1])/dx^2 (mirrors the solver)."""
    op = np.zeros_like(u)
    op[1:-1] = D[1:-1] * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    return op[1:-1]


def _cyl_operator(u: np.ndarray, r: np.ndarray, dx: float) -> np.ndarray:
    """Cylindrical radial operator u'' + (1/r) u' on interior nodes."""
    op = np.zeros_like(u)
    d2 = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    d1 = (u[2:] - u[:-2]) / (2.0 * dx)
    op[1:-1] = d2 + d1 / r[1:-1]
    return op[1:-1]


def _observed_order(errs: list[float], refinements: list[int]) -> list[float]:
    """Orders between consecutive refinements (assumes uniform refinement ratio)."""
    orders = []
    for i in range(1, len(errs)):
        ratio = refinements[i] / refinements[i - 1]
        orders.append(float(np.log(errs[i - 1] / errs[i]) / np.log(ratio)))
    return orders


def run_v3_thermal_slab(
    n_levels: tuple[int, ...] = (101, 201, 401),
    L: float = 1.0,
) -> dict:
    """Manufactured u(x)=sin(pi x/L), constant alpha=1 on a slab.

    Exact operator value: u'' = -(pi/L)^2 sin(pi x/L).
    """
    errs = []
    for n in n_levels:
        x = np.linspace(0.0, L, n)
        dx = x[1] - x[0]
        u = np.sin(np.pi * x / L)
        D = np.ones(n)  # constant coefficient
        num = _slab_operator(u, D, dx)
        exact = -((np.pi / L) ** 2) * np.sin(np.pi * x[1:-1] / L)
        errs.append(float(np.max(np.abs(num - exact))))
    orders = _observed_order(errs, list(n_levels))
    obs = min(orders) if orders else 0.0
    return {
        "name": "V3a thermal slab operator",
        "errors": errs,
        "orders": orders,
        "observed_order": obs,
        "theoretical": 2.0,
        "passed": obs >= 2.0 - 0.15,
    }


def run_v3_carbon_variable_D(
    n_levels: tuple[int, ...] = (101, 201, 401),
    L: float = 1.0,
    D0: float = 2.2e-5,
    Q_J: float = 137000.0,
) -> dict:
    """Manufactured u(x)=x^2(L-x)^2 with D varying via a manufactured T(x).

    T(x) is chosen so D(T(x)) varies smoothly across the domain; the discrete
    operator D_i * u''_h must converge to D(x) u''(x) at 2nd order.
    u''(x) = 2L^2 - 12Lx + 12x^2 for u = x^2 (L-x)^2.
    """
    errs = []
    for n in n_levels:
        x = np.linspace(0.0, L, n)
        dx = x[1] - x[0]
        u = x**2 * (L - x) ** 2
        # Manufactured temperature field: 900..1100 C across the domain so
        # D(T) genuinely varies (Arrhenius), in Kelvin.
        T_K = (900.0 + 200.0 * x / L) + 273.15
        D = D_of_T_np(D0, Q_J, T_K)
        num = _slab_operator(u, D, dx)
        u_second = 2.0 * L**2 - 12.0 * L * x[1:-1] + 12.0 * x[1:-1] ** 2
        exact = D[1:-1] * u_second
        errs.append(float(np.max(np.abs(num - exact))))
    orders = _observed_order(errs, list(n_levels))
    obs = min(orders) if orders else 0.0
    return {
        "name": "V3b carbon variable-D(T) operator",
        "errors": errs,
        "orders": orders,
        "observed_order": obs,
        "theoretical": 2.0,
        "passed": obs >= 2.0 - 0.15,
    }


def run_v3_thermal_cylinder(
    n_levels: tuple[int, ...] = (101, 201, 401),
    R: float = 1.0,
) -> dict:
    """Manufactured u(r)=r^4 on the cylinder radial operator.

    u'' + (1/r)u' = 12r^2 + 4r^2 = 16r^2 exactly. Central differences give a
    clean O(dx^2) truncation error here (unlike r^2, which is reproduced
    exactly and yields only round-off). Grid mirrors the solver: node 0 at the
    axis, interior nodes at r = i*dx.
    """
    errs = []
    for n in n_levels:
        dx = R / (n - 1)
        r = np.arange(n) * dx
        u = r**4
        num = _cyl_operator(u, r, dx)
        exact = 16.0 * r[1:-1] ** 2
        errs.append(float(np.max(np.abs(num - exact))))
    orders = _observed_order(errs, list(n_levels))
    obs = min(orders) if orders else 0.0
    return {
        "name": "V3c thermal cylinder operator",
        "errors": errs,
        "orders": orders,
        "observed_order": obs,
        "theoretical": 2.0,
        "passed": obs >= 2.0 - 0.15,
    }


def run_v3() -> dict:
    """Run all three MMS sub-checks; overall pass = all pass."""
    a = run_v3_thermal_slab()
    b = run_v3_carbon_variable_D()
    c = run_v3_thermal_cylinder()
    return {
        "sub": [a, b, c],
        "passed": a["passed"] and b["passed"] and c["passed"],
    }


if __name__ == "__main__":
    r = run_v3()
    for s in r["sub"]:
        status = "PASS" if s["passed"] else "FAIL"
        orders = ", ".join(f"{o:.2f}" for o in s["orders"])
        print(
            f"{s['name']} [{status}] observed={s['observed_order']:.2f} "
            f"(need >= {s['theoretical'] - 0.15:.2f}) orders=[{orders}]"
        )
    print("V3 overall:", "PASS" if r["passed"] else "FAIL")
