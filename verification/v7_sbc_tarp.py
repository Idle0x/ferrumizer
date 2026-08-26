"""V7 — Posterior calibration diagnostics: SBC rank uniformity + TARP coverage.

Implements Simulation-Based Calibration (Talts et al. 2018, arXiv:1804.06788)
and expected-coverage checking (Lemos et al. ICML 2023, arXiv:2302.03026).

SBC protocol (statistically valid):

* Draw parameters from the prior (NOT initialized at the truth).
* Simulate synthetic observations from the model at those draws.
* Run NUTS from a **prior-based initialization** (init_to_sample), never from
  the planted values — initializing at the truth would make the diagnostic
  pass trivially and invalidate it.
* Rank each true draw against the posterior samples; ranks must be uniform.

N_SIM = 200 (2 parameters x 200 simulations = 400 ranks; chi-squared test has
real power at this count, unlike the former N_SIM=4 which was pure theater).
To keep the gate tractable on CPU, V7 calibrates a reduced two-parameter
subset {log D0, C_pot} on a single schedule with the remaining parameters held
at their true values. The full five-parameter two-schedule protocol is
exercised by the calibration app and V6.

Gate: SBC rank-histogram chi-squared p-value > 0.05 AND 90% TARP coverage
within the binomial tolerance band for N_SIM draws
(coverage_ok = |coverage - 0.90| <= 1.96 * sqrt(0.9*0.1/N_SIM)).

Non-finite guard: forward-model outputs that blow up (an unstable explicit
step at an extreme prior draw) are handled with a hard likelihood penalty
(H -> 1e6), NOT silent clamping to 230 HV. A NaN must never masquerade as a
plausible flat line.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_sample

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy

numpyro.set_host_device_count(1)

# --- reduced problem settings (kept small enough for N_SIM=200 on CPU) ---
ALLOY = "8620"
T_TOTAL = 7200.0  # 2 h soak -> case deep enough to be observable
CARBON_N = 31  # dx = 0.27 mm over an 8 mm half-thickness (coarser = faster)
CARBON_DT = 5.0  # s (within explicit stability limit for mass_transfer)
N_T_SAMPLES = 30
# Fine sampling near the surface where the case forms.
OBS_DEPTHS_MM = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0])
SIGMA_H = 10.0  # HV measurement noise

# true / planted values (used only to SIMULATE data, never to initialize)
TRUE_LOG_D0 = float(np.log(2.2e-5))
TRUE_Q_KJ = 137.0
TRUE_C_POT = 1.0
TRUE_H_M = 1e-4
TRUE_EPS = 0.8

# Prior widths (narrow enough to ensure numerical stability, wide enough to
# be a genuine test: NUTS must find the posterior from a prior draw).
#
# C_pot is BOUNDED. 0.8-1.2 is the validated operating window: outside it the
# 8-point hardness profile becomes weakly informative (low C_pot -> nearly
# flat profile; high C_pot -> retained-austenite roll-off makes the surface
# hardness non-monotone in C_pot). Verified 2026-08-26: widening to
# 0.6-1.4 made the SBC WORSE (p 0.074->0.000, cov 0.83->0.73, ranks skewed
# to the top bin) — the extra width put truth draws in unidentifiable
# regions. Keep the box at the validated window.
PRIOR_SIGMA_LOG_D0 = 0.15
PRIOR_C_POT_LO, PRIOR_C_POT_HI = 0.8, 1.2

# SBC settings — N_SIM is the statistical power of the test. 200 simulations
# x 2 params = 400 ranks; the chi-squared test can distinguish uniformity
# from a loaded die at this count. Do not reduce below 200 (the former
# N_SIM=4 was statistically void).
#
# Warmup/draws must be long enough that the posterior intervals themselves
# are trustworthy: with too few draws the 5/95 percentile estimates carry
# large Monte Carlo error, which biases measured coverage DOWN even for a
# calibrated sampler (observed: coverage 0.84 at N_DRAWS=60). 200 warmup
# steps give the mass-matrix adaptation room; 200 draws make the percentile
# estimates stable.
#
# max_tree_depth: 8. Depth 6 with accept 0.8 / warmup 200 produced
# systematically too-narrow posteriors on the R2 physics (full gate:
# p=0.005 / coverage 0.82 at depth 6, and p=0.007 / 0.83 after the
# C∞-smoothing fix). Mini-SBC (30 sims) at depth 8 + accept 0.9 +
# warmup 300: chi2=12.67, p=0.855, clean rank histogram — the deeper
# trees + longer adaptation let the sampler express the true posterior
# spread. Cost: ~3-6x slower per sim on hard draws (26-45 s) — a full
# run takes ~2-3 h on this box.
N_SIM = 200
N_WARMUP = 300
N_DRAWS = 200
N_BINS = 20  # ~20 expected per bin at 400 ranks
MAX_TREE_DEPTH = 8


def _kwargs():
    preset = load_alloy(ALLOY)
    th = preset["thermal"]
    return dict(
        t_total=T_TOTAL,
        T_init_K=298.15,
        T_quench=298.15,
        h_conv=20.0,
        k=th["k"],
        rho_cp=th["rho"] * th["cp"],
        half_thickness_m=16.0 / 2000.0,
        x_half_mm=8.0,
        carbon_n=CARBON_N,
        carbon_dt=CARBON_DT,
        # Dirichlet: h_m is held at its true value in the reduced subset, so
        # the SBC diagnostic tests sampler self-consistency without paying the
        # mass-transfer substep cost (8 ms vs 1.2 ms per eval). Statistical
        # validity comes from N_SIM + prior init + coverage band, not from the
        # boundary condition.
        carbon_mode="dirichlet",
        preset=preset,
        n_T_samples=N_T_SAMPLES,
    )


_KW = _kwargs()
_KNOTS = jnp.array([[0.0, T_TOTAL], [950.0, 950.0]], dtype=jnp.float64)
_OBS_DEPTHS = jnp.asarray(OBS_DEPTHS_MM, jnp.float64)


def _predict_H(log_D0, C_pot):
    out = fast_forward(
        log_D0,
        jnp.float64(TRUE_Q_KJ),
        C_pot,
        jnp.float64(TRUE_H_M),
        jnp.float64(TRUE_EPS),
        schedule_knots=_KNOTS,
        **_KW,
    )
    H = jnp.interp(_OBS_DEPTHS, out["x_mm"], out["H"])
    # HARD non-finite guard: an unstable draw must produce a huge residual
    # (massive likelihood penalty), never a plausible 230 HV flat line.
    return jnp.where(jnp.isfinite(H), H, 1e6)


# JIT-compile the forward prediction for speed during NUTS sampling.
_predict_H_jit = jax.jit(_predict_H)


def _model(obs_H):
    log_D0 = numpyro.sample("log_D0", dist.Normal(TRUE_LOG_D0, PRIOR_SIGMA_LOG_D0))
    C_pot = numpyro.sample("C_pot", dist.Uniform(PRIOR_C_POT_LO, PRIOR_C_POT_HI))
    H_pred = _predict_H_jit(log_D0, C_pot)
    numpyro.sample("obs", dist.Normal(H_pred, SIGMA_H), obs=obs_H)


def _run_inference(obs_H, seed):
    kernel = NUTS(
        _model,
        # 0.9 with depth 8 / warmup 300: accept 0.8 with shallow trees
        # adapted a step size that expressed too little posterior spread
        # (the under-coverage root cause on the R2 physics). 0.9 keeps
        # the random walk near unit-acceptance where the adapted step is
        # best calibrated, and the longer warmup gives mass-matrix
        # adaptation room. Cost: some prior draws explore large trees
        # (26-45 s/sim instead of ~5 s) — acceptable for the gate.
        target_accept_prob=0.9,
        max_tree_depth=MAX_TREE_DEPTH,
        # Sample initialization from the prior — the honest SBC start.
        init_strategy=init_to_sample(),
    )
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_DRAWS, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), obs_H)
    return mcmc.get_samples(group_by_chain=False)


def run_v7(n_sim: int = N_SIM) -> dict:
    """Run the SBC + TARP gate.

    Args:
        n_sim: number of prior->data->posterior simulations. Default N_SIM=200.
            Do not go below 200 for a meaningful rank test.
    """
    rng = np.random.default_rng(12345)
    ranks = {p: [] for p in ("log_D0", "C_pot")}
    covers_90 = dict.fromkeys(("log_D0", "C_pot"), 0)

    for s in range(n_sim):
        # draw parameters from the PRIOR (never the truth)
        t_log_D0 = float(rng.normal(TRUE_LOG_D0, PRIOR_SIGMA_LOG_D0))
        t_C_pot = float(rng.uniform(PRIOR_C_POT_LO, PRIOR_C_POT_HI))
        # simulate noisy data from the same model
        H_clean = np.asarray(_predict_H_jit(jnp.float64(t_log_D0), jnp.float64(t_C_pot)))
        obs_H = jnp.asarray(H_clean + rng.normal(0.0, SIGMA_H, size=H_clean.shape))

        t0_sim = time.time()
        samples = _run_inference(obs_H, seed=s)
        dt_sim = time.time() - t0_sim
        if (s + 1) % 10 == 0 or dt_sim > 20:
            print(f"  [sim {s + 1}/{n_sim}] {dt_sim:.1f}s", flush=True)

        for p, true_val in (("log_D0", t_log_D0), ("C_pot", t_C_pot)):
            post = np.asarray(samples[p])
            ranks[p].append(int(np.sum(post < true_val)))
            lo, hi = np.percentile(post, [5, 95])
            if lo <= true_val <= hi:
                covers_90[p] += 1

    # SBC chi-squared test on pooled ranks
    from scipy.stats import chisquare

    all_ranks = np.concatenate([np.array(ranks[p]) for p in ranks])
    hist, _ = np.histogram(all_ranks, bins=N_BINS, range=(0, N_DRAWS))
    expected = np.full(N_BINS, len(all_ranks) / N_BINS)
    chi2_stat, p_value = chisquare(hist, expected)

    # TARP-style expected coverage at the 90% level with an honest binomial
    # tolerance band: coverage must be within ~2 sigma of 0.90 for N_SIM
    # simulations. (The former >=0.60 threshold was a rubber stamp.)
    coverage = float(np.mean([covers_90[p] / n_sim for p in covers_90]))
    tol = 1.96 * np.sqrt(0.9 * 0.1 / n_sim)
    coverage_ok = abs(coverage - 0.90) <= tol

    passed = (p_value > 0.05) and coverage_ok
    return {
        "n_sim": n_sim,
        "ranks": {p: ranks[p] for p in ranks},
        "rank_hist": hist.tolist(),
        "chi2_stat": float(chi2_stat),
        "sbc_p_value": float(p_value),
        "coverage_90": coverage,
        "coverage_tolerance": float(tol),
        "coverage_ok": bool(coverage_ok),
        "passed": bool(passed),
    }


if __name__ == "__main__":
    r = run_v7()
    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"V7 [{status}]  SBC p={r['sbc_p_value']:.3f} (>0.05)  "
        f"coverage90={r['coverage_90']:.2f} (tol {r['coverage_tolerance']:.3f})  "
        f"chi2={r['chi2_stat']:.2f}  N_SIM={r['n_sim']}"
    )
    print(f"  rank hist: {r['rank_hist']}")
