"""Bayesian calibration of carburizing parameters via NumPyro NUTS.

Calibrates theta = {log D0, Q, C_pot, h_m, eps} against a measured hardness
traverse. Gates (see docs/adr/ADR-002 and docs/verification.md): R-hat < 1.01
and bulk ESS > 400, otherwise results are blocked from release.

Boundary condition: calibration uses the **mass-transfer (Robin)** carbon
boundary condition by default so that the mass-transfer coefficient ``h_m``
is actually exercised by the likelihood. In Dirichlet mode the surface
concentration is pinned at ``C_pot`` and ``h_m`` has zero effect — sampling
it there would be calibrating a dead parameter (a flat posterior direction
that wastes ESS and can destabilize the other parameters). An assertion
enforces this.

Measurement noise: ``sigma_hv`` is either fixed by the caller or inferred
hierarchically (``infer_sigma=True``), because treating measurement scatter
as known and constant makes the posterior overconfident when the true
measurement noise is larger or heteroscedastic.

The forward model uses the lumped-capacitance thermal surrogate (ADR-002) so
each likelihood evaluation stays cheap enough for thousands of NUTS steps.
Non-finite forward outputs (e.g. an unstable explicit step at an extreme
prior draw) are handled by a **hard penalty** rather than silent clamping:
the likelihood is pushed to a huge residual so the sampler moves away, and
the result is never presented as a plausible flat line.
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
# Prior for inferred measurement noise (HV). HalfNormal(15) keeps the scale
# near the documented repeatability value while letting the data move it.
SIGMA_HV_PRIOR_SCALE = 15.0


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
        # The quench model must be IDENTICAL to the app path (review 2).
        # If the scenario carries a quench medium, fast_forward uses the
        # same spatial conduction solve + per-depth Scheil-JMAK as
        # pipeline.forward; if None, the legacy instantaneous-quench path
        # is used and the calibration assumes an instant quench (the UI
        # warns about this).
        quench_medium=scenario.quench_medium,
        quench_temp_K=scenario.quench_temp_K,
        quench_agitation=scenario.quench_agitation,
        quench_time_s=scenario.quench_time_s,
        geometry=scenario.geometry,
        size_mm=scenario.size_mm,
        thermal_n=scenario.thermal_n,
    )


def _predict_hardness(log_D0, Q_kJ, C_pot, h_m, eps, obs_depths, kwargs):
    out = fast_forward(log_D0, Q_kJ, C_pot, h_m, eps, **kwargs)
    # interpolate predicted H at observed depths (surface->core grid)
    H_pred = jnp.interp(obs_depths, out["x_mm"], out["H"])
    # Hard non-finite guard: never let a NaN/inf masquerade as a plausible
    # hardness. A huge finite residual is a strong likelihood penalty, so the
    # sampler walks away instead of settling on a flat 230 HV line.
    H_pred = jnp.where(jnp.isfinite(H_pred), H_pred, 1e6)
    return H_pred, out["ecd_mm"]


def model(obs_depths, obs_H, sigma, kwargs, infer_sigma: bool = True, obs2=None, kwargs2=None):
    log_D0 = numpyro.sample("log_D0", dist.Normal(LOG_D0_LOC, LOG_D0_SCALE))
    Q_kJ = numpyro.sample("Q_kJ", dist.Uniform(Q_MIN_KJ, Q_MAX_KJ))
    C_pot = numpyro.sample("C_pot", dist.Uniform(C_POT_MIN, C_POT_MAX))
    log_hm = numpyro.sample("log_hm", dist.Normal(LOG_HM_LOC, LOG_HM_SCALE))
    eps = numpyro.sample("eps", dist.Uniform(EPS_MIN, EPS_MAX))
    if infer_sigma:
        sigma = numpyro.sample("sigma_hv", dist.HalfNormal(SIGMA_HV_PRIOR_SCALE))

    H_pred, _ = _predict_hardness(log_D0, Q_kJ, C_pot, jnp.exp(log_hm), eps, obs_depths, kwargs)
    numpyro.sample("obs", dist.Normal(H_pred, sigma), obs=obs_H)

    if obs2 is not None and kwargs2 is not None:
        # two-schedule protocol: the same parameters must explain BOTH
        # traverses; this is what breaks the D0-Q collinearity.
        H_pred2, _ = _predict_hardness(log_D0, Q_kJ, C_pot, jnp.exp(log_hm), eps, obs2[0], kwargs2)
        numpyro.sample("obs2", dist.Normal(H_pred2, sigma), obs=obs2[1])


def run_calibration(
    obs_depths: np.ndarray,
    obs_H: np.ndarray,
    scenario,
    sigma_hv: float = 15.0,
    infer_sigma: bool = True,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.9,
    obs2_depths: np.ndarray | None = None,
    obs2_H: np.ndarray | None = None,
    scenario2=None,
):
    """Run NUTS calibration. Returns (mcmc, summary_dict).

    ``scenario.carbon_mode`` must be ``"mass_transfer"`` when ``h_m`` is
    sampled (which it always is) — see module docstring.

    Pass ``obs2_depths``/``obs2_H``/``scenario2`` to run the two-schedule
    identifiability protocol: the same parameters must explain both traverses.
    """
    if scenario.carbon_mode != "mass_transfer":
        raise ValueError(
            "calibration samples h_m (mass-transfer coefficient), which has no "
            "effect under carbon_mode='dirichlet'. Use carbon_mode='mass_transfer' "
            "(Robin boundary condition) so h_m is exercised by the likelihood."
        )
    kwargs = _scenario_kwargs(scenario)
    obs_depths_j = jnp.asarray(obs_depths, jnp.float64)
    obs_H_j = jnp.asarray(obs_H, jnp.float64)

    kwargs2 = None
    obs2 = None
    if obs2_depths is not None and obs2_H is not None:
        if scenario2 is None:
            raise ValueError("scenario2 is required when a second traverse is provided")
        if scenario2.carbon_mode != "mass_transfer":
            raise ValueError("scenario2.carbon_mode must be 'mass_transfer'")
        kwargs2 = _scenario_kwargs(scenario2)
        obs2 = (jnp.asarray(obs2_depths, jnp.float64), jnp.asarray(obs2_H, jnp.float64))

    from numpyro.infer.initialization import init_to_sample

    kernel = NUTS(
        model,
        target_accept_prob=target_accept,
        max_tree_depth=8,
        # Sample initialization from the prior (not numpyro's default
        # init_to_uniform). With the R2 physics (ASM martensite curve,
        # phase-specific hardness), extreme uniform-bound draws can produce
        # non-finite likelihoods; init_to_uniform then fails with "Cannot
        # find valid initial parameters" and the app's calibration tab
        # crashes. init_to_sample + the hard non-finite guard in
        # _predict_hardness keep the sampler robust (same strategy as V7).
        init_strategy=init_to_sample(),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    rng = jax.random.PRNGKey(seed)
    mcmc.run(rng, obs_depths_j, obs_H_j, sigma_hv, kwargs, infer_sigma, obs2, kwargs2)

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


def posterior_predictive_hardness(
    mcmc,
    obs_depths: np.ndarray,
    scenario,
    n_draws: int = 200,
    seed: int = 1,
) -> dict:
    """Posterior predictive check: predicted hardness at observed depths.

    Draws posterior samples, runs the forward model for each, and returns the
    mean, 5/95 credible band, and per-draw samples at ``obs_depths``. This is
    the "does the posterior actually reproduce the data?" check the
    convergence gates cannot answer.
    """
    kwargs = _scenario_kwargs(scenario)
    samples = mcmc.get_samples(group_by_chain=False)
    idx = np.random.default_rng(seed).choice(
        len(samples["log_D0"]), size=min(n_draws, len(samples["log_D0"])), replace=False
    )
    od = jnp.asarray(obs_depths, jnp.float64)
    H_draws = np.empty((len(idx), len(od)))
    ecds = []
    for i, n in enumerate(idx):
        H_pred, ecd = _predict_hardness(
            samples["log_D0"][n],
            samples["Q_kJ"][n],
            samples["C_pot"][n],
            jnp.exp(samples["log_hm"][n]),
            samples["eps"][n],
            od,
            kwargs,
        )
        H_draws[i] = np.asarray(H_pred)
        ecds.append(float(ecd))
    return {
        "obs_depths": np.asarray(obs_depths),
        "H_mean": H_draws.mean(axis=0),
        "H_lo": np.percentile(H_draws, 5, axis=0),
        "H_hi": np.percentile(H_draws, 95, axis=0),
        "H_draws": H_draws,
        "ecd_ppc": np.array(ecds),
    }


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
