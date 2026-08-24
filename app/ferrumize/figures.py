"""Figure generation for F1–F10 (see README "The figures" section).

All figures are regenerated deterministically from raw physics via
``make figures`` / ``ferrumize figures``. A fixed seed and fixed matplotlib
metadata keep the outputs byte-stable across runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation

jax.config.update("jax_enable_x64", True)

from ferrumize.pipeline import FerrumizerPipeline, ProcessParams, Scenario
from ferrumizer_physics.alloys import load_alloy

# Deterministic PNG metadata (no timestamps -> byte-stable output).
_PNG_META = {"Software": "ferrumizer", "Creation Time": "2026-08-22"}

# Brand palette (mirrors the README figure styling)
INK = "#ECE7DB"
GRAPHITE = "#23262A"
CHARCOAL = "#16181C"
GOLD = "#D6B57C"
EMBER = "#A05F30"
CREAM = "#EFE6CF"


def _style(ax, dark: bool = True):
    if dark:
        ax.set_facecolor(CHARCOAL)
        ax.figure.set_facecolor(CHARCOAL)
        for spine in ax.spines.values():
            spine.set_color(INK)
        ax.tick_params(colors=INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
    ax.grid(True, alpha=0.15, color=INK if dark else GRAPHITE)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=_PNG_META)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# F3 — erfc overlay
# --------------------------------------------------------------------------- #
def fig_f3_erfc(out: Path):
    from verification.limits.v2_erfc import run_v2

    r = run_v2()
    x, C_num, C_ref = r["x_mm"], r["C_num"], r["C_ref"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, C_ref, color=GOLD, lw=2.2, label="analytic erfc (Crank)")
    ax.plot(x, C_num, "--", color=INK, lw=1.4, label="FD solver")
    ax.set_xlabel("depth from surface, mm")
    ax.set_ylabel("carbon, mass-%")
    ax.set_title(f"F3 — semi-infinite erfc overlay  (norm L2 = {r['norm_l2']:.2e})")
    ax.legend(facecolor=CHARCOAL, edgecolor=INK, labelcolor=INK)
    _style(ax)
    _save(fig, out / "F3_erfc_overlay.png")


# --------------------------------------------------------------------------- #
# F4 — MMS convergence log-log
# --------------------------------------------------------------------------- #
def fig_f4_mms(out: Path):
    from verification.mms.v3_mms import (
        run_v3_carbon_variable_D,
        run_v3_thermal_cylinder,
        run_v3_thermal_slab,
    )

    checks = [
        run_v3_thermal_slab(),
        run_v3_carbon_variable_D(),
        run_v3_thermal_cylinder(),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [GOLD, EMBER, CREAM]
    for c, col in zip(checks, colors):
        n_levels = (101, 201, 401)
        dxs = [1.0 / (n - 1) for n in n_levels]
        ax.loglog(
            dxs,
            c["errors"],
            "o-",
            color=col,
            lw=1.8,
            label=f"{c['name']}  (order {c['observed_order']:.2f})",
        )
    # reference slope 2
    xs = np.array([dxs[0], dxs[-1]])
    ax.loglog(xs, xs**2 * 0.5, ":", color=INK, alpha=0.5, label="slope 2")
    ax.set_xlabel("grid spacing dx")
    ax.set_ylabel("max operator error")
    ax.set_title("F4 — MMS convergence (both stages, incl. variable D(T))")
    ax.legend(facecolor=CHARCOAL, edgecolor=INK, labelcolor=INK, fontsize=7)
    _style(ax)
    _save(fig, out / "F4_mms_convergence.png")


# --------------------------------------------------------------------------- #
# F5 — cross-AD agreement
# --------------------------------------------------------------------------- #
def fig_f5_cross_ad(out: Path):
    from verification.cross_ad.v4_cross_ad import run_v4

    r = run_v4()
    names = list(r["g_fd"].keys())
    fd = np.array([r["g_fd"][k] for k in names])
    jx = np.array([r["g_jax"][k] for k in names])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    ax.bar(np.arange(len(names)) - 0.2, fd, width=0.4, color=GOLD, label="FD box")
    ax.bar(np.arange(len(names)) + 0.2, jx, width=0.4, color=EMBER, label="JAX twin")
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("d(sum C_final)/d param")
    ax.set_title("parameter gradients")
    ax.legend(facecolor=CHARCOAL, edgecolor=INK, labelcolor=INK, fontsize=8)
    _style(ax)

    ax = axes[1]
    lim = max(np.max(np.abs(fd)), np.max(np.abs(jx))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ":", color=INK, alpha=0.5)
    ax.scatter(jx, fd, color=GOLD, s=60)
    for i, nm in enumerate(names):
        ax.annotate(
            nm, (jx[i], fd[i]), color=INK, fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    ax.set_xlabel("JAX twin gradient")
    ax.set_ylabel("FD box gradient")
    ax.set_title(f"agreement  (rel ∞-norm = {r['rel_inf_norm']:.2e})")
    _style(ax)
    fig.suptitle("F5 — cross-AD agreement", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out / "F5_cross_ad.png")


# --------------------------------------------------------------------------- #
# F8 — identifiability before/after two-schedule
# --------------------------------------------------------------------------- #
def fig_f8_identifiability(out: Path):
    from identifiability.analyze import two_schedule_comparison

    scenario_a = Scenario(
        alloy="8620",
        t_total=3600.0,
        schedule_times=(0.0, 3600.0),
        schedule_temps_C=(900.0, 900.0),
        thermal_n=21,
        carbon_n=21,
        carbon_dt=2.0,
    )
    scenario_b = Scenario(
        alloy="8620",
        t_total=3600.0,
        schedule_times=(0.0, 3600.0),
        schedule_temps_C=(1000.0, 1000.0),
        thermal_n=21,
        carbon_n=21,
        carbon_dt=2.0,
    )
    preset = load_alloy("8620")
    pipe_a = FerrumizerPipeline(scenario_a, ProcessParams())
    res_a = pipe_a.forward()
    pipe_b = FerrumizerPipeline(scenario_b, ProcessParams())
    res_b = pipe_b.forward()

    obs_depths = np.asarray(res_a["x_mm"])
    obs_H_a = np.asarray(res_a["H"])
    obs_H_b = np.asarray(res_b["H"])
    param_vec = np.array([np.log(2.2e-5), 137.0, 1.0, 1e-4, 0.8])

    rep = two_schedule_comparison(param_vec, scenario_a, scenario_b, obs_depths, obs_H_a, obs_H_b)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    names = rep["single_schedule"]["param_names"]
    for ax, key, title in (
        (axes[0], "single_schedule", "single schedule (D0–Q collinear)"),
        (axes[1], "combined", "two schedules (identifiability restored)"),
    ):
        corr = rep[key]["correlation"]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)), [n[:6] for n in names], rotation=45)
        ax.set_yticks(range(len(names)), [n[:6] for n in names])
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7
                )
        ax.set_title(f"{title}\ncond = {rep[key]['condition_number']:.2e}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("F8 — parameter correlation before/after two-schedule protocol", color=INK)
    fig.set_facecolor(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out / "F8_identifiability.png")


# --------------------------------------------------------------------------- #
# F9 — Pareto front
# --------------------------------------------------------------------------- #
def fig_f9_pareto(out: Path):
    from design.optimize import pareto_front

    scenario = Scenario(
        alloy="8620",
        t_total=3600.0,
        schedule_times=(0.0, 1200.0, 3600.0),
        schedule_temps_C=(950.0, 950.0, 950.0),
        thermal_n=21,
        carbon_n=21,
        carbon_dt=2.0,
    )
    params = ProcessParams().__dict__
    params = {**params, "log_D0": np.log(params.pop("D0"))}
    # Target must sit INSIDE the reachable ECD range of this 1 h schedule
    # (~[0.08, 0.12] mm) or the front degenerates to all points at the bound
    # and every weight grinds to max steps. 5 log-spaced weights trace the
    # ECD-vs-energy tradeoff; heavier sets are available via
    # ``pareto_front(..., weights=...)`` for a converged figure.
    points = pareto_front(
        0.10,
        scenario,
        params,
        weights=np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4]),
        n_steps=30,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [p["energy_proxy"] for p in points]
    ys = [p["ecd_mm"] for p in points]
    ax.plot(xs, ys, "o-", color=GOLD, lw=1.8, ms=6)
    for i, p in enumerate(points):
        ax.annotate(
            f"w={p['weight']:.0e}",
            (xs[i], ys[i]),
            color=INK,
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("energy proxy  (∫ setpoint above ambient dt)")
    ax.set_ylabel("achieved ECD, mm")
    ax.set_title("F9 — Pareto front: ECD vs energy penalty")
    _style(ax)
    _save(fig, out / "F9_pareto.png")


# --------------------------------------------------------------------------- #
# F10 — alloy strip
# --------------------------------------------------------------------------- #
def fig_f10_alloys(out: Path):
    alloys = ["8620", "9310", "5120"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, alloy in zip(axes, alloys):
        scenario = Scenario(
            alloy=alloy,
            t_total=7200.0,
            schedule_times=(0.0, 7200.0),
            schedule_temps_C=(950.0, 950.0),
            thermal_n=21,
            carbon_n=41,
            carbon_dt=2.0,
        )
        pipe = FerrumizerPipeline(scenario, ProcessParams())
        res = pipe.forward()
        x = np.asarray(res["x_mm"])
        ax.plot(x, np.asarray(res["carbon"]["C_final"]), color=GOLD, lw=1.8, label="C, mass-%")
        ax2 = ax.twinx()
        ax2.plot(x, np.asarray(res["H"]), color=EMBER, lw=1.8, label="H, HV")
        ax2.axhline(550, color=CREAM, ls=":", lw=1.0, alpha=0.6)
        ax.set_xlabel("depth, mm")
        ax.set_title(f"AISI {alloy}  (ECD {float(res['ecd_mm']):.2f} mm)", fontsize=10)
        ax.set_facecolor(CHARCOAL)
        ax.tick_params(colors=INK)
        ax2.tick_params(colors=INK)
        for spine in list(ax.spines.values()) + list(ax2.spines.values()):
            spine.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(GOLD)
        ax2.yaxis.label.set_color(EMBER)
        ax.set_ylabel("carbon", color=GOLD)
        ax2.set_ylabel("hardness HV", color=EMBER)
    axes[0].figure.suptitle("F10 — alloy strip: carbon & hardness profiles", color=INK)
    fig.set_facecolor(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out / "F10_alloy_strip.png")


# --------------------------------------------------------------------------- #
# F1 — hero chain-loop animation (GIF)
# --------------------------------------------------------------------------- #
def fig_f1_hero(out: Path):
    scenario = Scenario(
        alloy="8620",
        t_total=1800.0,
        schedule_times=(0.0, 1800.0),
        schedule_temps_C=(950.0, 950.0),
        thermal_n=21,
        thermal_sample_every=20,
        carbon_n=21,
        carbon_dt=2.0,
        carbon_sample_every=30,
    )
    pipe = FerrumizerPipeline(scenario, ProcessParams())
    res = pipe.forward()

    times = np.asarray(res["thermal"]["times_s"])
    Ts = np.asarray(res["thermal"]["Ts"])
    Tc = np.asarray(res["thermal"]["Tcore"])
    C_hist = np.asarray(res["carbon"]["C_hist"])
    x_mm = np.asarray(res["x_mm"])
    preset = pipe.preset

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    fig.set_facecolor(CHARCOAL)

    def frame(i):
        for ax in axes.flat:
            ax.clear()
            ax.set_facecolor(CHARCOAL)
            ax.tick_params(colors=INK)
            for s in ax.spines.values():
                s.set_color(INK)
        idx = min(i, len(times) - 1)
        cidx = min(i, C_hist.shape[0] - 1)

        ax = axes[0, 0]
        ax.plot(times[: idx + 1] / 60, Ts[: idx + 1] - 273.15, color=GOLD, lw=1.6)
        ax.plot(times[: idx + 1] / 60, Tc[: idx + 1] - 273.15, color=EMBER, lw=1.6)
        ax.set_title("schedule → temperature", color=INK, fontsize=10)
        ax.set_xlabel("time, min", color=INK)
        ax.set_ylabel("T, °C", color=INK)

        ax = axes[0, 1]
        ax.plot(x_mm, C_hist[cidx], color=GOLD, lw=1.8)
        ax.set_title("carbon diffusion C(x)", color=INK, fontsize=10)
        ax.set_xlabel("depth, mm", color=INK)
        ax.set_ylabel("C, mass-%", color=INK)

        from ferrumizer_physics.hardening import (
            ecd_from_hardness,
            hardness_profile,
            km_fraction,
            ms_andrews,
        )

        C = jnp.asarray(C_hist[cidx])
        Ms = ms_andrews(C, preset["ms"]["A"], preset["ms"]["b_carbon"])
        fM = km_fraction(Ms, scenario.T_quench, preset["km_alpha"])
        H = hardness_profile(C, preset, fM)
        ecd = ecd_from_hardness(H, jnp.asarray(x_mm), preset["ecd_threshold_hv"])

        ax = axes[1, 0]
        ax.plot(x_mm, np.asarray(H), color=EMBER, lw=1.8)
        ax.axhline(550, color=CREAM, ls=":", alpha=0.6)
        ax.set_title("hardness H(x)", color=INK, fontsize=10)
        ax.set_xlabel("depth, mm", color=INK)
        ax.set_ylabel("HV", color=INK)

        ax = axes[1, 1]
        ax.bar(["ECD"], [float(ecd)], color=GOLD, width=0.4)
        ax.set_title(f"effective case depth = {float(ecd):.3f} mm", color=INK, fontsize=10)
        ax.set_ylabel("mm", color=INK)

        fig.suptitle("Ferrumizer — gradients through the furnace", color=INK, fontsize=13)
        return axes

    n_frames = min(len(times), C_hist.shape[0])
    step = max(1, n_frames // 40)
    anim = animation.FuncAnimation(fig, frame, frames=range(0, n_frames, step))
    anim.save(out / "F1_hero_loop.gif", writer="pillow", fps=8)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# F2 — architecture diagram
# --------------------------------------------------------------------------- #
def fig_f2_architecture(out: Path):
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.set_facecolor(CHARCOAL)
    fig.set_facecolor(CHARCOAL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.5, 2.0, "thermal-stage\n(JAX · autodiff)", GOLD),
        (3.8, 2.0, "carburizing-stage\n(NumPy FD · legacy box)", EMBER),
        (7.1, 2.0, "hardening-stage\n(JAX · autodiff)", GOLD),
    ]
    for x, y, label, col in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y), 2.4, 1.4, fill=True, facecolor=col, edgecolor=INK, alpha=0.85, lw=1.5
            )
        )
        ax.text(
            x + 1.2,
            y + 0.7,
            label,
            ha="center",
            va="center",
            color=CHARCOAL,
            fontsize=10,
            fontweight="bold",
        )

    # arrows with ∂ crossing the two container boundaries
    for x0 in (2.9, 6.2):
        ax.annotate(
            "",
            xy=(x0 + 0.9, 2.7),
            xytext=(x0, 2.7),
            arrowprops=dict(arrowstyle="->", color=INK, lw=2),
        )
        ax.text(
            x0 + 0.45,
            3.0,
            "∂",
            ha="center",
            va="center",
            color=CREAM,
            fontsize=20,
            fontweight="bold",
        )

    ax.text(
        5.0,
        4.4,
        "gradients flow through TWO container boundaries",
        ha="center",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        5.0,
        0.8,
        "θ = { D₀, Q, C_pot, h_m, ε }   →   ECD @ 550 HV (ISO 2639)",
        ha="center",
        color=GOLD,
        fontsize=10,
    )
    _save(fig, out / "F2_architecture.png")


# --------------------------------------------------------------------------- #
# F6 — posterior pairplot
# --------------------------------------------------------------------------- #
def fig_f6_posterior(out: Path):
    from calibration.calibrate import run_calibration
    from ferrumize.models import fast_forward

    preset = load_alloy("8620")
    th = preset["thermal"]
    knots = jnp.array([[0.0, 1800.0], [950.0, 950.0]], dtype=jnp.float64)
    res = fast_forward(
        jnp.log(jnp.float64(2.2e-5)), jnp.float64(137.0), jnp.float64(1.0),
        jnp.float64(1e-4), jnp.float64(0.8),
        schedule_knots=knots, t_total=1800.0, T_init_K=298.15,
        T_quench=298.15, h_conv=20.0, k=th["k"],
        rho_cp=th["rho"] * th["cp"], half_thickness_m=0.008,
        x_half_mm=8.0, carbon_n=21, carbon_dt=2.0,
        carbon_mode="dirichlet", preset=preset, n_T_samples=60,
    )
    scenario = Scenario(
        alloy="8620", t_total=1800.0,
        schedule_times=(0.0, 1800.0), schedule_temps_C=(950.0, 950.0),
        thermal_n=21, carbon_n=21, carbon_dt=2.0,
    )
    rng = np.random.default_rng(0)
    obs_depths = np.asarray(res["x_mm"])[::4]
    obs_H = np.asarray(res["H"])[::4] + rng.normal(0, 10.0, size=len(np.asarray(res["x_mm"])[::4]))

    mcmc, summary = run_calibration(
        obs_depths,
        obs_H,
        scenario,
        num_warmup=200,
        num_samples=200,
        num_chains=1,
        seed=0,
    )
    samples = mcmc.get_samples(group_by_chain=False)
    names = ["log_D0", "Q_kJ", "C_pot", "log_hm", "eps"]
    fig, axes = plt.subplots(len(names), len(names), figsize=(9, 9), squeeze=False)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axes[i, j]
            if i == j:
                ax.hist(np.asarray(samples[name_i]), bins=24, color=GOLD, alpha=0.85)
            else:
                ax.scatter(np.asarray(samples[name_j]), np.asarray(samples[name_i]),
                           s=2, alpha=0.15, color=EMBER)
            ax.tick_params(labelsize=6)
            if i == len(names) - 1:
                ax.set_xlabel(name_j, fontsize=7)
            if j == 0:
                ax.set_ylabel(name_i, fontsize=7)
    fig.patch.set_facecolor(CHARCOAL)
    fig.suptitle("F6 — calibrated posterior (NUTS) with reference anchors", y=0.995)
    fig.savefig(out / "F6_posterior.png", dpi=120, bbox_inches="tight", metadata=_PNG_META)
    plt.close("all")


# --------------------------------------------------------------------------- #
# F7 — noise sweep
# --------------------------------------------------------------------------- #
def fig_f7_noise_sweep(out: Path):
    from verification.v6_recovery import run_v6

    sigmas = [0.0, 5.0, 10.0, 20.0]
    errs = []
    # The figure shows the recovery-error-vs-noise TREND, so heavy convergence
    # is not needed: 20 L-BFGS iterations per point keeps the figure tractable
    # on CPU (~2 min/point at the noisy settings; the sigma=0 point converges
    # early). V6's own gate runs the same recovery to 1e-4 rel. error.
    for s in sigmas:
        r = run_v6(noise_sigma=s, max_iter=20)
        errs.append(r["max_rel_err"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(sigmas, [max(e, 1e-16) for e in errs], "o-", color=GOLD, lw=1.8)
    ax.axhline(1e-4, color=CREAM, ls=":", alpha=0.6, label="V6 gate 1e-4")
    ax.set_xlabel("hardness noise σ_HV")
    ax.set_ylabel("max param relative error")
    ax.set_title("F7 — noise sweep: recovery error vs measurement noise")
    ax.legend(facecolor=CHARCOAL, edgecolor=INK, labelcolor=INK)
    _style(ax)
    _save(fig, out / "F7_noise_sweep.png")


def generate_all(out: Path, seed: int = 0):
    """Generate all figures F1-F10 into ``out``.

    Slow, compute-heavy figures (F6 NUTS posterior, F7 noise sweep, F9 Pareto)
    are scheduled FIRST so a CI time-box never leaves the headline figures
    stale; F1 (hero GIF) and F2 (architecture) are deterministic SVG/plot work.
    Each figure prints its elapsed time so `make figures` is observable.
    """
    np.random.seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    generators = [
        ("F6", fig_f6_posterior),
        ("F7", fig_f7_noise_sweep),
        ("F9", fig_f9_pareto),
        ("F5", fig_f5_cross_ad),
        ("F4", fig_f4_mms),
        ("F8", fig_f8_identifiability),
        ("F3", fig_f3_erfc),
        ("F10", fig_f10_alloys),
        ("F2", fig_f2_architecture),
        ("F1", fig_f1_hero),
    ]
    import time as _time

    for name, fn in generators:
        t0 = _time.time()
        print(f"[figures] {name} ...")
        fn(out)
        print(f"[figures] {name} done in {_time.time() - t0:.1f}s")
