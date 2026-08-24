"""Bayesian calibration of carburizing parameters via NumPyro NUTS.

Calibrates theta = {log D0, Q, C_pot, h_m, eps} against a measured hardness
traverse. Gates (see docs/adr/ADR-002 and docs/verification.md): R-hat < 1.01
and bulk ESS > 400, otherwise results are blocked from release.

The forward model uses the lumped-capacitance thermal surrogate (ADR-002) so
each likelihood evaluation stays cheap enough for thousands of NUTS steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy

jax.config.update("jax_enable_x64", True)

# Prior bounds (Appendix A [S3] for Q; ADR-001 for D0)
LOG_D0_LOC = float(np.log(2.2e-5))
LOG_D0_SCALE = 0.5
Q_MIN_KJ, Q_MAX_KJ = 100.0, 200.0
C_POT_MIN, C_POT_MAX = 0.6, 1.4
LOG_HM_LOC = float(np.log(1e-4))
LOG_HM_SCALE = 0.75
EPS_MIN, EPS_MAX = 0.3, 1.0


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


def _predict_hardness(log_D0, Q_kJ, C_pot, h_m, eps, obs_depths, kwargs):
    out = fast_forward(log_D0, Q_kJ, C_pot, h_m, eps, **kwargs)
    # interpolate predicted H at observed depths (surface->core grid)
    H_pred = jnp.interp(obs_depths, out["x_mm"], out["H"])
    return H_pred, out["ecd_mm"]


def model(obs_depths, obs_H, sigma, kwargs):
    log_D0 = numpyro.sample("log_D0", dist.Normal(LOG_D0_LOC, LOG_D0_SCALE))
    Q_kJ = numpyro.sample("Q_kJ", dist.Uniform(Q_MIN_KJ, Q_MAX_KJ))
    C_pot = numpyro.sample("C_pot", dist.Uniform(C_POT_MIN, C_POT_MAX))
    log_hm = numpyro.sample("log_hm", dist.Normal(LOG_HM_LOC, LOG_HM_SCALE))
    eps = numpyro.sample("eps", dist.Uniform(EPS_MIN, EPS_MAX))

    H_pred, _ = _predict_hardness(log_D0, Q_kJ, C_pot, jnp.exp(log_hm), eps, obs_depths, kwargs)
    numpyro.sample("obs", dist.Normal(H_pred, sigma), obs=obs_H)


def run_calibration(
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma_hv: float = 15.0,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.9,
):
    """Run NUTS calibration. Returns (mcmc, summary_dict)."""
    kwargs = _scenario_kwargs(scenario)
    obs_depths_j = jnp.asarray(obs_depths, jnp.float64)
    obs_H_j = jnp.asarray(obs_H, jnp.float64)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept,
        max_tree_depth=8,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    rng = jax.random.PRNGKey(seed)
    mcmc.run(rng, obs_depths_j, obs_H_j, sigma_hv, kwargs)

    summary = summarize(mcmc)
    return mcmc, summary


def summarize(mcmc) -> dict:
    """Compute release gates: R-hat < 1.01, bulk ESS > 400."""
    from numpyro.diagnostics import summary as np_summary

    samples = mcmc.get_samples(group_by_chain=True)
    stats = np_summary(
        {k: np.asarray(v) for k, v in samples.items()},
        prob=0.9,
    )
    gates_ok = True
    rows = {}
    for name, s in stats.items():
        rhat = float(s["r_hat"])
        ess = float(s["n_eff"])
        ok = (rhat < 1.01) and (ess > 400)
        gates_ok = gates_ok and ok
        rows[name] = {
            "mean": float(s["mean"]),
            "sd": float(s["std"]),
            "r_hat": rhat,
            "bulk_ess": ess,
            "hdi_5%": float(s["5.0%"]),
            "hdi_95%": float(s["95.0%"]),
            "gate_ok": ok,
        }
    return {"params": rows, "gates_ok": gates_ok}


def posterior_predictive_ecd(mcmc, scenario, n_draws: int = 200, seed: int = 1):
    """Draw posterior predictive ECD values for reporting."""
    kwargs = _scenario_kwargs(scenario)
    samples = mcmc.get_samples(group_by_chain=False)
    idx = np.random.default_rng(seed).choice(
        len(samples["log_D0"]), size=min(n_draws, len(samples["log_D0"])), replace=False
    )
    ecds = []
    for i in idx:
        _, ecd = _predict_hardness(
            samples["log_D0"][i],
            samples["Q_kJ"][i],
            samples["C_pot"][i],
            jnp.exp(samples["log_hm"][i]),
            samples["eps"][i],
            jnp.asarray([0.0], jnp.float64),  # depths unused for ECD
            kwargs,
        )
        ecds.append(float(ecd))
    return np.array(ecds)
