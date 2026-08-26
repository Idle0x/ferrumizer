"""Identifiability analysis: Fisher information and parameter correlation.

Demonstrates why a single-schedule calibration cannot uniquely identify
D0 and Q (they are collinear), and how a two-schedule protocol resolves
the ambiguity (see docs/adr/ADR-002 and figure F8).
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


def _d0q_block_condition(F: np.ndarray) -> dict:
    """Conditioning of the (log D0, Q) pair after profiling nuisance params.

    The full Fisher condition number is dominated by h_m, which is weakly
    identified from end-state hardness (see V6) — a statement about h_m, not
    about the D0-Q degeneracy the two-schedule protocol fixes. This helper
    Schur-complements out the nuisance block {C_pot, h_m, eps} and reports
    the 2x2 curvature of (log D0, Q) alone: the flat direction of the
    D0-Q ridge is visible as a tiny eigenvalue in the single-schedule case,
    and two schedules must grow it by orders of magnitude.
    """
    A = F[:2, :2]
    B = F[:2, 2:]
    D = F[2:, 2:]
    S = A - B @ np.linalg.inv(D + 1e-12 * np.eye(3)) @ B.T
    eigvals = np.linalg.eigvalsh(S)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    corr = S[0, 1] / np.sqrt(max(S[0, 0] * S[1, 1], 1e-30))
    return {
        "d0q_condition_number": cond,
        "d0q_min_eigenvalue": float(eigvals[0]),
        "d0q_correlation": float(corr),
    }


def identifiability_report(
    param_vec: np.ndarray,
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma: float = 15.0,
) -> dict:
    """Full identifiability analysis for a single schedule.

    Returns Fisher matrix, correlation matrix, and condition number.

    The single-schedule Fisher is expected to be (nearly) singular: that IS
    the D0-Q collinearity ridge the two-schedule protocol resolves. A tiny
    diagonal jitter (1e-8 relative) keeps the eigenvalue computation stable
    so the report can say "condition number ~ 1e12" instead of crashing.
    """
    F = fisher_information(param_vec, obs_depths, obs_H, scenario, sigma)
    corr = correlation_matrix(F)
    F_jit = F + 1e-8 * np.trace(F) / F.shape[0] * np.eye(F.shape[0])
    eigvals = np.linalg.eigvalsh(F_jit)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    return {
        "fisher": F,
        "correlation": corr,
        "condition_number": cond,
        "param_names": PARAM_NAMES,
        **_d0q_block_condition(F),
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
            **_d0q_block_condition(F_combined),
        },
    }


# --------------------------------------------------------------------------- #
# Profile likelihoods — the non-Gaussian complement to Fisher information
# --------------------------------------------------------------------------- #
# Fisher information is local: it describes the curvature at a single point.
# For diffusion problems with boundary layers the posterior is frequently
# non-Gaussian, so the correlation matrix can mislead. Profile likelihoods
# evaluate the log-likelihood over a grid while optimizing out the nuisance
# parameters — a global, robust picture of the D0-Q degeneracy and its
# resolution by the two-schedule protocol.


def _neg_log_lik(pv, obs_depths, obs_H, scenario, sigma):
    """Negative log-likelihood of the hardness residuals at a parameter point."""
    kwargs = _scenario_kwargs(scenario)
    log_D0, Q_kJ, C_pot, h_m, eps = pv
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
    H_pred = jnp.interp(jnp.asarray(obs_depths, jnp.float64), out["x_mm"], out["H"])
    resid = jnp.asarray(H_pred) - jnp.asarray(obs_H, jnp.float64)
    return jnp.sum(resid**2) / (2.0 * sigma**2)


def profile_likelihood_grid(
    param_vec: np.ndarray,
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma: float = 15.0,
    log_D0_range: tuple[float, float, int] = (-11.5, -10.0, 25),
    Q_range: tuple[float, float, int] = (110.0, 165.0, 25),
    n_nuisance_iters: int = 15,
    obs2_depths: np.ndarray | None = None,
    obs2_H: np.ndarray | None = None,
    scenario2=None,
) -> dict:
    """2-D profile log-likelihood over (log D0, Q), nuisance params optimized.

    For each (log D0, Q) grid point, the remaining parameters {C_pot, h_m,
    eps} are re-fit by a few projected-gradient steps, then the marginal
    log-likelihood is recorded. This is the "does the data actually constrain
    the pair?" surface — flat ridges reveal degeneracy; a tight single well
    proves identifiability.

    For the two-schedule protocol, pass ``obs2_depths``/``obs2_H``/
    ``scenario2``: the total likelihood is the SUM over both schedules at
    each grid point (a single schedule's data cannot resolve the D0-Q ridge;
    two temperatures can).

    The whole grid is vmapped through ONE jitted function (nuisance steps via
    ``lax.scan``), so the cost is one compile + a few seconds of batched
    compute, not a Python loop over grid points.

    Returns grid arrays plus the point of maximum profile likelihood.
    """
    l0s = np.linspace(*log_D0_range)
    qs = np.linspace(*Q_range)
    nll = np.empty((len(qs), len(l0s)))

    od = jnp.asarray(obs_depths, jnp.float64)
    oH = jnp.asarray(obs_H, jnp.float64)
    base = jnp.asarray(param_vec, jnp.float64)

    od2 = jnp.asarray(obs2_depths, jnp.float64) if obs2_depths is not None else None
    oH2 = jnp.asarray(obs2_H, jnp.float64) if obs2_H is not None else None

    # nuisance bounds (C_pot, log h_m, eps)
    lo = jnp.array([0.6, np.log(1e-6), 0.3], jnp.float64)
    hi = jnp.array([1.4, np.log(1e-2), 1.0], jnp.float64)

    def obj(nu, l0, q):
        pv = jnp.array([l0, q, nu[0], jnp.exp(nu[1]), nu[2]], jnp.float64)
        total = _neg_log_lik(pv, od, oH, scenario, sigma)
        if od2 is not None and oH2 is not None and scenario2 is not None:
            total = total + _neg_log_lik(pv, od2, oH2, scenario2, sigma)
        return total

    grad_obj = jax.jit(jax.grad(obj))

    @jax.jit
    def point_nll(l0, q):
        """Profile the nuisance block for ONE (l0, q) via fixed scan steps."""
        nu0 = jnp.array([base[2], jnp.log(base[3]), base[4]], jnp.float64)
        # Normalized gradient step: raw NLL gradients are scale-mismatched
        # (NLL ~ 1-100, gradients ~ 1e2-1e3), so a fixed lr either crawls or
        # slams the bounds. Unit-direction steps make `step_size` a
        # parameter-space step per iteration, robust across the grid.
        step_size = 0.05

        def step(nu, _):
            g = grad_obj(nu, l0, q)
            gnorm = jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)
            g_unit = g / gnorm
            nu_new = jnp.clip(nu - step_size * g_unit, lo, hi)
            return nu_new, nu_new

        nu_final, _ = jax.lax.scan(step, nu0, jnp.arange(n_nuisance_iters))
        return obj(nu_final, l0, q)

    # vmap over the flattened grid: one compile, batched compute.
    L0, Q = np.meshgrid(l0s, qs)
    flat_l0 = L0.ravel()
    flat_q = Q.ravel()
    nll_flat = jax.jit(jax.vmap(point_nll))(jnp.asarray(flat_l0), jnp.asarray(flat_q))
    nll = np.asarray(nll_flat).reshape(len(qs), len(l0s))

    i_min = np.unravel_index(np.argmin(nll), nll.shape)
    return {
        "log_D0_grid": l0s,
        "Q_grid": qs,
        "neg_log_lik": nll,
        "best_log_D0": float(l0s[i_min[1]]),
        "best_Q": float(qs[i_min[0]]),
        "min_neg_log_lik": float(nll[i_min]),
    }
