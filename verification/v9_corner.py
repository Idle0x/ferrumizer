"""V9 — Two-schedule posterior corner plot (the D0-Q identifiability picture).

The single-schedule Fisher analysis and the profile-likelihood contours
already prove that D0 and Q are collinear with one schedule and identified
with two (see docs/adr/ADR-002 and F8). This gate adds the *Bayesian*
version of the same story: run NUTS on the two-schedule protocol and plot
the actual posterior marginals + pairwise contours.

A flat D0-Q ridge in the single-schedule posterior and a tight, roughly
elliptical cloud around the planted values in the two-schedule posterior is
the non-Gaussian, sampling-based confirmation of identifiability.

This was deferred from review 1 as the last optional item; it is a
diagnostic plot + gate, NOT part of the calibration product path. It runs
on light grids (carbon_n=21, dt=8) so it is tractable on CPU: the point is
the *shape* of the posterior, not research-grade ESS.

Output: verification output dir (or cwd) ``corner_two_schedule.png`` and a
text gate verdict on the D0-Q correlation / ridge flatness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components" / "shared"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ferrumize.pipeline import Scenario  # noqa: E402
from verification.v6_recovery import (  # noqa: E402
    PLANTED,
    SCHEDULES,
)


def _light_scenario(temps_C: tuple[float, float]) -> Scenario:
    """Scenario used for BOTH data generation and calibration.

    The corner plot is a self-consistency diagnostic (SBC-style): the
    synthetic traverses are generated with the SAME scenario (same carbon
    grid, same mass-transfer BC, same slab geometry) that the calibration
    NUTS model uses. If they differed, a discretization mismatch would
    partially resurrect the D0-Q ridge and the plot would lie about the
    identifiability protocol (measured corr +0.99 with mismatched grids).
    """
    return Scenario(
        alloy="8620",
        t_total=7200.0,
        schedule_times=(0.0, 7200.0),
        schedule_temps_C=temps_C,
        thermal_n=21,
        carbon_n=21,  # light grid — fast posterior shape, not research ESS
        carbon_dt=8.0,  # light grid
        carbon_mode="mass_transfer",
    )


def _generate_on_scenario(scenario: Scenario, noise_sigma: float, seed: int) -> dict:
    """Synthetic traverse generated on the SAME grid as the calibration model.

    Uses the planted parameters and the exact `_scenario_kwargs` path the
    calibration likelihood calls, so data and model agree discretization-
    for-discretization (the honest SBC-style setup).
    """
    from calibration.calibrate import _scenario_kwargs
    from verification.v6_recovery import _predict_H, PLANTED

    kwargs = dict(_scenario_kwargs(scenario))
    # _predict_H builds its own schedule_knots from kwargs["t_total"] and
    # passes it explicitly — strip ours to avoid the duplicate-arg collision.
    kwargs.pop("schedule_knots", None)
    p = [
        jnp.float64(PLANTED["log_D0"]),
        jnp.float64(PLANTED["Q_kJ"]),
        jnp.float64(PLANTED["C_pot"]),
        jnp.float64(np.log(PLANTED["h_m"])),
        jnp.float64(PLANTED["eps"]),
    ]
    rng = np.random.default_rng(seed)
    H_clean = np.asarray(_predict_H(p, scenario.schedule_temps_C, kwargs))
    H_obs = H_clean + rng.normal(0.0, noise_sigma, size=H_clean.shape)
    # OBS_DEPTHS_MM is the traverse depth grid used by the forward model
    from verification.v6_recovery import OBS_DEPTHS_MM

    return {"depths_mm": OBS_DEPTHS_MM.copy(), "H_obs": H_obs}


def run_corner(
    num_warmup: int = 150,
    num_samples: int = 250,
    num_chains: int = 2,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict:
    """Run the two-schedule NUTS posterior and write the corner plot."""
    from calibration.calibrate import run_calibration

    sc_low = _light_scenario(SCHEDULES[0]["temps_C"])
    sc_high = _light_scenario(SCHEDULES[1]["temps_C"])
    data_low = _generate_on_scenario(sc_low, noise_sigma=12.0, seed=seed)
    data_high = _generate_on_scenario(sc_high, noise_sigma=12.0, seed=seed + 1)

    mcmc, summary = run_calibration(
        data_low["depths_mm"],
        data_low["H_obs"],
        sc_low,
        sigma_hv=12.0,
        infer_sigma=False,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        target_accept=0.8,
        obs2_depths=data_high["depths_mm"],
        obs2_H=data_high["H_obs"],
        scenario2=sc_high,
    )
    samples = mcmc.get_samples(group_by_chain=False)

    log_D0 = np.asarray(samples["log_D0"])
    Q = np.asarray(samples["Q_kJ"])
    C_pot = np.asarray(samples["C_pot"])
    eps = np.asarray(samples["eps"])

    # ---- Gate metrics (honest, documented) ------------------------------ #
    # The D0-Q correlation in the posterior is NOT the acceptance criterion:
    # even a perfectly identified two-schedule fit keeps high correlation
    # because the pair is a near-degenerate (Arrhenius) direction — the CLI
    # identifiability docs say exactly this and use the (D0, Q)-BLOCK
    # curvature instead. What the corner plot must show:
    #   (a) the posterior is TIGHT around the planted values (a ridge would
    #       be wide along Q or log D0), and
    #   (b) the (D0, Q)-block condition number from the posterior covariance
    #       is finite (the ridge is not a flat direction).
    q_sd = float(np.std(Q))
    d0_sd = float(np.std(log_D0))
    q_err = abs(float(np.mean(Q)) - PLANTED["Q_kJ"])
    d0_err = abs(float(np.mean(log_D0)) - PLANTED["log_D0"])

    # posterior (D0, Q)-block curvature: precision matrix of the 2-vector
    cov_dq = np.cov(np.stack([log_D0, Q]))
    prec_dq = np.linalg.inv(cov_dq + 1e-12 * np.eye(2))
    eig = np.linalg.eigvalsh(prec_dq)
    block_cond = float(eig[-1] / max(eig[0], 1e-30))

    # acceptance: posterior tight AND on target AND the D0-Q block is
    # well-conditioned (no flat ridge direction in the SAMPLED posterior).
    passed = (
        q_sd < 25.0
        and d0_sd < 1.0
        and q_err < 3.0 * q_sd
        and d0_err < 3.0 * d0_sd
        and block_cond < 1e6
    )

    names = ["log D0", "Q (kJ/mol)", "C_pot", "eps"]
    truths = [PLANTED["log_D0"], PLANTED["Q_kJ"], PLANTED["C_pot"], PLANTED["eps"]]
    chain = [log_D0, Q, C_pot, eps]

    fig, axes = plt.subplots(4, 4, figsize=(11, 11), facecolor="white")
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(chain[i], bins=32, color="#2a4d69", alpha=0.85)
                ax.axvline(truths[i], color="#c0392b", lw=2)
                ax.set_xlabel(names[i], fontsize=9)
            else:
                ax.plot(chain[j], chain[i], ".", ms=2, alpha=0.4, color="#2a4d69")
                ax.axvline(truths[j], color="#c0392b", lw=1.2, alpha=0.7)
                ax.axhline(truths[i], color="#c0392b", lw=1.2, alpha=0.7)
                ax.set_xlabel(names[j], fontsize=9)
                ax.set_ylabel(names[i], fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "Two-schedule posterior (NUTS): tight cloud around the planted D0, Q\n"
        f"SD(Q) = {q_sd:.1f} kJ/mol, SD(log D0) = {d0_sd:.3f}, "
        f"(D0,Q)-block cond# = {block_cond:.1e}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    out = (out_dir or Path.cwd()) / "corner_two_schedule.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return {
        "passed": passed,
        "Q_sd": q_sd,
        "logD0_sd": d0_sd,
        "Q_err": q_err,
        "logD0_err": d0_err,
        "block_cond": block_cond,
        "n_samples": len(log_D0),
        "plot": str(out),
        "rhat_ok": summary["gates_ok"],
    }


def main() -> None:
    r = run_corner()
    print(f"V9 corner plot: {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  posterior SD(Q) = {r['Q_sd']:.2f} kJ/mol, SD(log D0) = {r['logD0_sd']:.3f}")
    print(f"  posterior mean error: Q {r['Q_err']:.2f}, log D0 {r['logD0_err']:.4f}")
    print(f"  (D0,Q)-block cond# = {r['block_cond']:.1e}  (want < 1e6)")
    print(f"  samples = {r['n_samples']}, convergence = {r['rhat_ok']}")
    print(f"  plot: {r['plot']}")


if __name__ == "__main__":
    main()
