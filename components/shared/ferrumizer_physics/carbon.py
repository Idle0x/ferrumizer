"""Carburizing stage: 1-D carbon diffusion with Arrhenius D(T).

Two implementations, one source of truth for constants:

- :func:`run_carburize` — the JAX twin (differentiable, used for cross-AD
  verification V4 and by the hardening/app layer).
- :func:`run_carburize_numpy` — the legacy-style explicit NumPy FD stepper
  (what the ``carburizing-stage`` Tesseract's apply endpoint runs).

Both use identical discretization (FTCS, ghost-point BCs) so their results
agree to float tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ferrumizer_physics.constants import RGAS


@dataclass(frozen=True)
class CarburizeConfig:
    """Discretization for the carbon diffusion stage (half-domain)."""

    n: int  # nodes on [core, surface]
    dt: float  # s
    t_total: float  # s
    mode: str = "dirichlet"  # "dirichlet" | "mass_transfer"
    sample_every: int = 1
    safety: float = 0.45

    def replace(self, **kwargs) -> CarburizeConfig:
        data = self.__dict__.copy()
        data.update(kwargs)
        return CarburizeConfig(**data)


def D_of_T(D0, Q_J, T_K):
    """Arrhenius diffusivity D(T) = D0 exp(-Q / (R T)) in m^2/s (JAX)."""
    return D0 * jnp.exp(-Q_J / (RGAS * jnp.maximum(T_K, 1.0)))


def D_of_T_np(D0: float, Q_J: float, T_K) -> np.ndarray:
    """Arrhenius diffusivity (NumPy variant)."""
    return D0 * np.exp(-Q_J / (RGAS * np.maximum(T_K, 1.0)))


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer | jax.Array)


def stability_check_carbon(
    D_peak: float, dx: float, dt: float, mode: str, hm: float, safety: float = 0.45
) -> None:
    """Raise ValueError if dt violates explicit stability bounds (N5)."""
    dt_max = safety * dx * dx / max(D_peak, 1e-30)
    if dt > dt_max:
        raise ValueError(
            f"carburizing dt={dt:.4g} s exceeds stability limit {dt_max:.4g} s "
            f"(D_peak={D_peak:.3e} m^2/s, dx={dx:.4g} m)."
        )
    if mode == "mass_transfer" and hm > 0:
        dt_bc = dx / (2.0 * hm)
        if dt > dt_bc:
            raise ValueError(
                f"carburizing dt={dt:.4g} s exceeds mass-transfer BC stability "
                f"limit {dt_bc:.4g} s (hm={hm:.3g} m/s)."
            )


def run_carburize(
    T_surf_history,  # (M,) K, uniformly sampled over [0, t_total]
    cfg: CarburizeConfig,
    C0,
    D0,
    Q_J,
    C_pot,
    hm,
    x_half_mm: float,
) -> dict:
    """JAX twin: dC/dt = d/dx(D(T) dC/dx) on [core, surface].

    Node 0 = core (zero-flux ghost), node n-1 = surface.
    Returns C_final ordered surface -> core (index 0 = surface).
    """
    n = cfg.n
    dx = float((x_half_mm / 1000.0) / (n - 1.0))
    N0 = int(np.ceil(cfg.t_total / cfg.dt))
    m = max(1, min(int(cfg.sample_every), max(1, N0)))
    # Round N up to a multiple of m so every sample slot is filled and the
    # JAX/NumPy paths step identically.
    N = ((N0 + m - 1) // m) * m
    Ts_flat = jnp.ravel(jnp.asarray(T_surf_history, dtype=jnp.float64))
    M = Ts_flat.shape[0]

    if not any(_is_tracer(v) for v in (T_surf_history, D0, Q_J, C_pot, hm)):
        D_peak = float(D_of_T(float(D0), float(Q_J), float(jnp.max(Ts_flat))))
        stability_check_carbon(D_peak, dx, cfg.dt, cfg.mode, float(hm), cfg.safety)

    dx_a = jnp.asarray(dx)
    n_out = N // m + 1

    # Adaptive sub-stepping (tracer-safe, reverse-mode-safe): the explicit
    # scheme must stay stable for ANY draw inside the calibration/design
    # prior, not just nominal parameters. For each macro step we branch:
    #   * stable draw  -> the exact single explicit step (bit-identical to
    #     the legacy NumPy stepper; parity tests hold),
    #   * unstable draw -> subdivide the step so r <= safety.
    # ``jax.lax.cond`` keeps both branches differentiable (JAX rejects
    # reverse-mode through while_loop with dynamic bounds).
    # n_sub_max is STATIC (derived from the config + a conservative bound on
    # D over the documented priors), so the inner scan has static length.
    ii = jnp.arange(1, n - 1)
    # Conservative worst-case diffusivity bound: D0 up to 1e-3 m^2/s with Q=0
    # would require 10^6 substeps; in practice the documented priors cap D0
    # near 1e-4 and Q >= 100 kJ/mol (Appendix A [S3]). Use a generous static
    # bound that still keeps the scan small: D_peak <= 1e-7 m^2/s at
    # carburizing temperatures (2e-4 * exp(-100e3/(R*1223)) ~ 1e-8; 10x margin).
    _D_BOUND = 1.0e-7
    n_sub_max = int(np.ceil(cfg.dt / (cfg.safety * dx * dx / _D_BOUND)))
    n_sub_max = max(1, min(n_sub_max, 4096))

    def micro(C, k, i_base):
        step_idx = i_base + k
        j = jnp.clip((step_idx * M) // N, 0, M - 1)
        T_i = Ts_flat[j]
        D_i = D_of_T(D0, Q_J, T_i)
        r_full = D_i * cfg.dt / dx_a**2
        stable = cfg.dt <= cfg.safety * dx_a**2 / jnp.maximum(D_i, 1e-30)
        # mass-transfer BC has its own stability limit; compute it JAX-natively
        # (hm may be a tracer inside the scan).
        stable_mt = jnp.where(
            (cfg.mode == "mass_transfer") & (hm > 0),
            cfg.dt <= dx_a / (2.0 * hm),
            True,
        )
        stable = stable & stable_mt

        def single_step(Cs):
            lap = (Cs[ii + 1] - 2.0 * Cs[ii] + Cs[ii - 1]) / dx_a**2
            Cn = Cs.at[ii].set(Cs[ii] + D_i * cfg.dt * lap)
            # core: zero-flux ghost point C[-1] = C[1]
            Cn = Cn.at[0].set(Cs[0] + 2.0 * r_full * (Cs[1] - Cs[0]))
            if cfg.mode == "dirichlet":
                Cn = Cn.at[n - 1].set(C_pot)
            else:
                # surface: ghost node from Robin condition D dC/dx = hm (Cgas - Cs)
                Cn = Cn.at[n - 1].set(
                    Cs[n - 1]
                    + 2.0 * r_full * (Cs[n - 2] - Cs[n - 1])
                    + 2.0 * hm * cfg.dt / dx_a * (C_pot - Cs[n - 1])
                )
            return Cn

        def substepped(Cs):
            # tracer-safe dynamic substep count, clamped to the static bound
            dt_max_diff = cfg.safety * dx_a**2 / jnp.maximum(D_i, 1e-30)
            dt_max = dt_max_diff
            dt_max = jnp.where(
                (cfg.mode == "mass_transfer") & (hm > 0),
                jnp.minimum(dt_max, dx_a / (2.0 * hm)),
                dt_max,
            )
            n_sub = jnp.minimum(
                jnp.maximum(1, jnp.ceil(cfg.dt / dt_max).astype(jnp.int32)),
                jnp.int32(n_sub_max),
            )
            dt_eff = cfg.dt / n_sub
            r_eff = D_i * dt_eff / dx_a**2

            def substep_body(c, i):
                lap = (c[ii + 1] - 2.0 * c[ii] + c[ii - 1]) / dx_a**2
                c_new = c.at[ii].set(c[ii] + D_i * dt_eff * lap)
                c_new = c_new.at[0].set(c[0] + 2.0 * r_eff * (c[1] - c[0]))
                if cfg.mode == "dirichlet":
                    c_new = c_new.at[n - 1].set(C_pot)
                else:
                    c_new = c_new.at[n - 1].set(
                        c[n - 1]
                        + 2.0 * r_eff * (c[n - 2] - c[n - 1])
                        + 2.0 * hm * dt_eff / dx_a * (C_pot - c[n - 1])
                    )
                return jax.lax.cond(i < n_sub, lambda: c_new, lambda: c), None

            out, _ = jax.lax.scan(substep_body, Cs, jnp.arange(n_sub_max))
            return out

        return jax.lax.cond(stable, single_step, substepped, C), None

    def outer(C, i_out):
        C, _ = jax.lax.scan(lambda c, k: micro(c, k, i_out * m), C, jnp.arange(m))
        return C, jnp.flip(C)

    C0v = jnp.full(n, jnp.asarray(C0, jnp.float64))
    n_blocks = N // m
    C_last, C_hist_blocks = jax.lax.scan(outer, C0v, jnp.arange(n_blocks))
    # Prepend the initial state (t=0) so sampling matches the NumPy path.
    C_hist = jnp.concatenate([jnp.flip(C0v)[None, :], C_hist_blocks], axis=0)
    return {
        "C_final": jnp.flip(C_last),  # (n,) surface -> core
        "C_hist": C_hist,  # (n_blocks+1, n) surface -> core
        "times_s": jnp.arange(n_blocks + 1) * (m * cfg.dt),
        "dx": dx_a,
        "n": n,
    }


def run_carburize_numpy(
    T_surf_history: np.ndarray,
    cfg: CarburizeConfig,
    C0: float,
    D0: float,
    Q_J: float,
    C_pot: float,
    hm: float,
    x_half_mm: float,
) -> dict:
    """Legacy-style explicit NumPy FD stepper (same discretization as the
    JAX twin). This is what the carburizing-stage container's apply runs.
    """
    n = cfg.n
    dx = float((x_half_mm / 1000.0) / (n - 1.0))
    N0 = int(np.ceil(cfg.t_total / cfg.dt))
    m = max(1, min(int(cfg.sample_every), max(1, N0)))
    # Round N up to a multiple of m so every sample slot is filled and the
    # JAX/NumPy paths step identically.
    N = ((N0 + m - 1) // m) * m
    Ts_flat = np.ravel(np.asarray(T_surf_history, dtype=np.float64))
    M = Ts_flat.shape[0]

    D_peak = float(D_of_T_np(D0, Q_J, float(Ts_flat.max())))
    stability_check_carbon(D_peak, dx, cfg.dt, cfg.mode, float(hm), cfg.safety)

    n_out = N // m + 1
    C = np.full(n, float(C0), dtype=np.float64)
    C_hist = np.empty((n_out, n), dtype=np.float64)
    C_hist[0] = C[::-1].copy()
    out_i = 1

    for step in range(N):
        j = min(int(step * M / N), M - 1)
        T_i = Ts_flat[j]
        D_i = float(D_of_T_np(D0, Q_J, T_i))
        r = D_i * cfg.dt / dx**2
        Cn = C.copy()
        Cn[1:-1] = C[1:-1] + D_i * cfg.dt * (C[2:] - 2.0 * C[1:-1] + C[:-2]) / dx**2
        Cn[0] = C[0] + 2.0 * r * (C[1] - C[0])
        if cfg.mode == "dirichlet":
            Cn[-1] = C_pot
        else:
            Cn[-1] = C[-1] + 2.0 * r * (C[-2] - C[-1]) + 2.0 * hm * cfg.dt / dx * (C_pot - C[-1])
        C = Cn
        if (step + 1) % m == 0 and out_i < n_out:
            C_hist[out_i] = C[::-1].copy()
            out_i += 1

    return {
        "C_final": C[::-1].copy(),  # surface -> core
        "C_hist": C_hist,
        "times_s": np.arange(n_out) * (m * cfg.dt),
        "dx": dx,
        "n": n,
    }


def erfc_reference(x_mm, t_s: float, D: float, Cs: float, C0: float) -> np.ndarray:
    """Crank-class semi-infinite solution (constant surface concentration):
    C(x,t) = C0 + (Cs - C0) erfc(x / 2 sqrt(D t)). Used by V2."""
    from scipy.special import erfc

    x = np.asarray(x_mm, dtype=np.float64) * 1e-3
    return C0 + (Cs - C0) * erfc(x / (2.0 * np.sqrt(D * t_s)))
