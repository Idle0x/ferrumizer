"""Thermal stage: 1-D heat conduction with convective/radiative Robin BC.

This is the JAX-native differentiable reference implementation used by the
``thermal-stage`` Tesseract and by verification harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ferrumizer_physics.constants import SIGMA


@dataclass(frozen=True)
class ThermalConfig:
    """Discretization and material configuration for the thermal stage."""

    geometry: str
    size_mm: float
    n: int
    dt: float
    t_total: float
    alpha: float
    h: float
    eps: float
    k: float
    T_init_K: float
    sample_every: int = 1
    safety: float = 0.45

    def replace(self, **kwargs) -> ThermalConfig:
        data = self.__dict__.copy()
        data.update(kwargs)
        return ThermalConfig(**data)


def slab_grid(size_mm: float, n: int) -> tuple[jnp.ndarray, float]:
    """Node-centered slab grid on [-L/2, L/2], with L in metres."""
    if n < 3:
        raise ValueError("slab grid requires n >= 3")
    L = size_mm / 1000.0
    x = jnp.linspace(-L / 2.0, L / 2.0, n)
    return x, float(L / (n - 1.0))


def cyl_grid(d_mm: float, n: int) -> tuple[jnp.ndarray, float]:
    """Node-centered radial grid on [0, R], with R in metres."""
    if n < 3:
        raise ValueError("cylinder grid requires n >= 3")
    R = d_mm / 2000.0
    x = jnp.linspace(0.0, R, n)
    return x, float(R / (n - 1.0))


def grid(geometry: str, size_mm: float, n: int) -> tuple[jnp.ndarray, float]:
    if geometry == "slab":
        return slab_grid(size_mm, n)
    if geometry == "cylinder":
        return cyl_grid(size_mm, n)
    raise ValueError(f"geometry must be 'slab' or 'cylinder', got {geometry!r}")


def stability_dt(diffusivity: float, dx: float, safety: float = 0.45) -> float:
    """Explicit FTCS stability limit with safety factor."""
    if not np.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError(f"diffusivity must be positive finite, got {diffusivity!r}")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"dx must be positive finite, got {dx!r}")
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must be in (0, 1], got {safety!r}")
    return safety * dx * dx / diffusivity


def furnace_T(schedule_knots: jnp.ndarray, t: jnp.ndarray | float) -> jnp.ndarray:
    """Piecewise-linear furnace setpoint in K.

    Args:
        schedule_knots: array of shape (2, K); row 0 is time(s), row 1 is setpoint(C).
        t: scalar or array time(s) in seconds.
    """
    schedule_knots = jnp.asarray(schedule_knots, dtype=jnp.float64)
    ts = schedule_knots[0, :]
    TK = schedule_knots[1, :] + 273.15
    idx = jnp.clip(jnp.searchsorted(ts, t, side="right") - 1, 0, ts.shape[0] - 2)
    t0 = ts[idx]
    t1 = ts[idx + 1]
    T0 = TK[idx]
    T1 = TK[idx + 1]
    w = jnp.clip((t - t0) / jnp.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    return T0 + w * (T1 - T0)


def face_temperature(T_inner, T_furn, h, eps, k, dx):
    """Solve for surface-node temperature under Robin BC.

    Balance: k (Ts - T_inner)/dx = h (T_furn - Ts) + eps sigma (T_furn^4 - Ts^4).
    A linearized radiation term gives a closed-form initial guess, followed by
    three Newton polish steps. This is differentiable and stable.
    """
    T_inner = jnp.asarray(T_inner, dtype=jnp.float64)
    T_furn = jnp.asarray(T_furn, dtype=jnp.float64)
    Rlin = (T_furn + T_inner) * (T_furn**2 + T_inner**2)
    H = h + eps * SIGMA * Rlin
    t = (H * T_furn + (k / dx) * T_inner) / (H + k / dx)

    def f(s):
        return k * (s - T_inner) / dx - h * (T_furn - s) - eps * SIGMA * (T_furn**4 - s**4)

    def fp(s):
        return k / dx + h + 4.0 * eps * SIGMA * s**3

    for _ in range(3):
        t = t - f(t) / fp(t)
    return t


def _step_slab(T, dx, alpha, dt, T_left, T_right):
    ii = jnp.arange(1, T.shape[0] - 1)
    lap = (T[ii + 1] - 2.0 * T[ii] + T[ii - 1]) / (dx * dx)
    Tn = T.at[ii].set(T[ii] + alpha * dt * lap)
    return Tn.at[0].set(T_left).at[-1].set(T_right)


def _step_cyl(T, dx, alpha, dt, T_surface):
    n = T.shape[0]
    ii = jnp.arange(1, n - 1)
    r = ii * dx
    d2 = (T[ii + 1] - 2.0 * T[ii] + T[ii - 1]) / (dx * dx)
    d1 = (T[ii + 1] - T[ii - 1]) / (2.0 * dx)
    lap = d2 + d1 / r
    Tn = T.at[ii].set(T[ii] + alpha * dt * lap)
    lap0 = 4.0 * (T[1] - T[0]) / (dx * dx)
    Tn = Tn.at[0].set(T[0] + alpha * dt * lap0)
    return Tn.at[-1].set(T_surface)


def run_thermal(schedule_knots, cfg: ThermalConfig) -> dict:
    """Integrate the thermal history.

    Returns arrays:
        times_s: sampled times including t=0, shape (n_blocks+1,)
        Ts: surface temperature history, same shape
        Tcore: core/midplane temperature history, same shape
        T_final: final field, shape (n,)
        x: grid coordinates, shape (n,)
        dx: grid spacing
        n_steps: number of explicit steps actually taken
    """
    x, dx = grid(cfg.geometry, cfg.size_mm, cfg.n)
    if not isinstance(cfg.alpha, jax.core.Tracer):
        dt_max = stability_dt(float(cfg.alpha), dx, cfg.safety)
        if cfg.dt > dt_max:
            raise ValueError(
                f"thermal dt={cfg.dt:.4g} s exceeds explicit stability limit "
                f"{dt_max:.4g} s (alpha={cfg.alpha:.3g} m^2/s, dx={dx:.4g} m)."
            )

    N = int(np.ceil(cfg.t_total / cfg.dt))
    m = max(1, int(cfg.sample_every))
    n_blocks = (N + m - 1) // m
    dx_a = jnp.asarray(dx, dtype=jnp.float64)
    surf_idx = cfg.n - 1 if cfg.geometry == "slab" else cfg.n - 1
    core_idx = 0 if cfg.geometry == "cylinder" else cfg.n // 2

    def block_step(T, i_block):
        base = i_block * m

        def inner(Tk, k):
            ti = (base + k) * cfg.dt
            Tinf = furnace_T(schedule_knots, ti)
            if cfg.geometry == "slab":
                Tl = face_temperature(Tk[1], Tinf, cfg.h, cfg.eps, cfg.k, dx_a)
                Tr = face_temperature(Tk[-2], Tinf, cfg.h, cfg.eps, cfg.k, dx_a)
                return _step_slab(Tk, dx_a, cfg.alpha, cfg.dt, Tl, Tr), None
            Tr = face_temperature(Tk[-2], Tinf, cfg.h, cfg.eps, cfg.k, dx_a)
            return _step_cyl(Tk, dx_a, cfg.alpha, cfg.dt, Tr), None

        T, _ = jax.lax.scan(inner, T, jnp.arange(m))
        return T, ((base + m) * cfg.dt, T[surf_idx], T[core_idx])

    T0 = jnp.full(cfg.n, float(cfg.T_init_K), dtype=jnp.float64)
    T_final, (t_blocks, Ts_blocks, Tc_blocks) = jax.lax.scan(block_step, T0, jnp.arange(n_blocks))

    times_s = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), t_blocks])
    Ts = jnp.concatenate([jnp.full(1, float(cfg.T_init_K), dtype=jnp.float64), Ts_blocks])
    Tcore = jnp.concatenate([jnp.full(1, float(cfg.T_init_K), dtype=jnp.float64), Tc_blocks])

    return {
        "times_s": times_s,
        "Ts": Ts,
        "Tcore": Tcore,
        "T_final": T_final,
        "x": x,
        "dx": dx_a,
        "n_steps": n_blocks * m,
    }


def lumped_surface_T(
    schedule_knots,
    t,
    h,
    eps,
    k,
    rho_cp,
    half_thickness_m,
    T_init_K,
):
    """Lumped-capacitance surrogate used for fast calibration (ADR-002).

    Returns the volume-averaged/surface temperature approximation:
        T(t) = T_f - (T_f - T0) exp(-h_eff t / (rho_cp * half_thickness))
    with h_eff = h + 4 eps sigma T_f^3. This is the Bi << 1 limit verified by V1.
    """
    Tf = furnace_T(schedule_knots, t)
    h_eff = h + 4.0 * eps * SIGMA * Tf**3
    tau = rho_cp * half_thickness_m / h_eff
    return Tf - (Tf - T_init_K) * jnp.exp(-t / tau)
