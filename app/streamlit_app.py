"""Virtual Furnace — practitioner-facing Streamlit app for Ferrumizer.

The app is the interactive front end of the Ferrumizer process emulator:
drag a schedule, quench, and (optionally) a custom alloy chemistry; watch
the emulator predict temperature history, carbon diffusion, hardness, and
effective case depth in real time. The same engine powers the CLI, so
anything designed here can be scripted or calibrated from the terminal.

Tabs
----
* Virtual Furnace  — live schedule explorer (the "what-if" surface).
* Cycle Predictor  — NUTS Bayesian calibration from a measured traverse
  (or an ingested PLC log that carries one), with honest runtime warning.
* Log Ingestion    — upload a raw furnace PLC/datalogger export and see
  what the parser extracts before anything is calibrated.

Naming note: this is a *predictor/emulator*, not a "simulator" in the
colloquial fake sense — every curve is the solution of the same physics
model the CLI uses, with parameters documented in the repo.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "components" / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ferrumize.pipeline import FerrumizerPipeline, ProcessParams, Scenario
from ferrumizer_physics.alloys import composition_to_preset, list_alloys

# Brand palette (mirrors app/ferrumize/figures.py)
CHARCOAL = "#1c1b18"
INK = "#0d0c0b"
CREAM = "#efe9dd"
GOLD = "#d6b57c"
EMBER = "#c1502e"

st.set_page_config(page_title="Virtual Furnace — Ferrumizer", page_icon="◉", layout="wide")
st.title("Virtual Furnace")
st.caption(
    "Ferrumizer process emulator · gas carburizing end-to-end · ISO 2639 practice (550 HV) · "
    "finite-rate quench model"
)

# --------------------------------------------------------------------------- #
# Shared plot styling (fixed axes so physics is visible, not auto-scaled away)
# --------------------------------------------------------------------------- #
TEMP_YMAX = 1400.0   # K
CARBON_YMAX = 1.4    # mass-%
HARD_YMAX = 700.0    # HV


def _style_ax(ax, ylabel, ymax, ymin=0.0):
    ax.set_facecolor(CHARCOAL)
    ax.set_xlabel("time (s)", color=CREAM, fontsize=9)
    ax.set_ylabel(ylabel, color=CREAM, fontsize=9)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(colors=CREAM, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GOLD)
        ax.spines[s].set_alpha(0.4)
    ax.grid(color=GOLD, alpha=0.12, lw=0.5)


def dial_gauge(ecd_mm: float, max_mm: float = 2.0, threshold: float = 550.0):
    """Semicircular dial gauge styled as the Case-Depth Dial mark."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0), facecolor=CHARCOAL)
    ax.set_facecolor(CHARCOAL)

    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=CREAM, lw=8, alpha=0.15)

    sectors = [
        (0.0, 0.33, EMBER),
        (0.33, 0.66, GOLD),
        (0.66, 1.0, CREAM),
    ]
    for lo, hi, color in sectors:
        a0 = np.pi * (1 - lo)
        a1 = np.pi * (1 - hi)
        seg = np.linspace(a0, a1, 80)
        ax.plot(r * np.cos(seg), r * np.sin(seg), color=color, lw=8)

    frac = min(max(ecd_mm / max_mm, 0.0), 1.0)
    ang = np.pi * (1 - frac)
    ax.plot([0, 0.72 * np.cos(ang)], [0, 0.72 * np.sin(ang)], color="white", lw=3)
    ax.plot(0, 0, "o", color="white", ms=8, zorder=5)

    ax.text(
        0, -0.28, f"{ecd_mm:.3f} mm", ha="center", va="top",
        color="white", fontsize=18, fontweight="bold",
    )
    ax.text(
        0, -0.52, f"ECD @ {threshold:.0f} HV", ha="center", va="top",
        color=GOLD, fontsize=10,
    )
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.7, 1.15)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def run_emulation(
    preset,
    alloy_label,
    boost_c,
    boost_h,
    diffuse_c,
    diffuse_h,
    carbon_potential,
    emissivity,
    size_mm,
    quench_medium,
    quench_temp_c,
    quench_agitation,
    h_m=1.0e-4,
):
    """Forward emulation with a two-stage boost/diffuse schedule + quench."""
    boost_s = boost_h * 3600.0
    diffuse_s = diffuse_h * 3600.0
    t_total = boost_s + diffuse_s
    scenario = Scenario(
        alloy=alloy_label,
        t_total=t_total,
        schedule_times=(0.0, boost_s, t_total),
        schedule_temps_C=(float(boost_c), float(boost_c), float(diffuse_c)),
        thermal_n=41,
        thermal_sample_every=100,
        carbon_n=81,
        carbon_dt=2.0,
        carbon_sample_every=300,
        size_mm=size_mm,
        # mass_transfer (Robin) BC — SAME as the calibration path, so a
        # calibrated h_m replays identically here instead of being silently
        # dropped (the old dirichlet default made h_m a dead parameter on
        # this tab). h_m is the surface mass-transfer coefficient (m/s).
        carbon_mode="mass_transfer",
        quench_medium=quench_medium,
        quench_temp_K=quench_temp_c + 273.15,
        quench_agitation=quench_agitation,
    )
    params = ProcessParams(C_pot=carbon_potential, eps=emissivity, h_m=h_m)
    return FerrumizerPipeline(scenario, params, preset=preset).forward()


# --------------------------------------------------------------------------- #
# Sidebar: process controls (with tooltips on everything)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Process")
    alloy_choice = st.selectbox(
        "Alloy",
        ["8620", "9310", "5120", "Custom…"],
        help="Steel grade. The three shipped presets carry literature-anchored "
        "diffusion/hardness constants; 'Custom…' builds a preset from the "
        "chemistry you enter below using published correlations (estimates, "
        "not certified constants).",
    )
    preset = None
    alloy_label = alloy_choice
    if alloy_choice == "Custom…":
        st.markdown("**Custom chemistry (wt-%)**")
        c_c = st.number_input("C (carbon)", 0.05, 1.2, 0.20, 0.01, help="Required.")
        mn_c = st.number_input("Mn", 0.0, 2.5, 0.45, 0.05, help="Ms estimate input.")
        cr_c = st.number_input("Cr", 0.0, 2.5, 0.40, 0.05, help="Ms estimate input.")
        ni_c = st.number_input("Ni", 0.0, 4.0, 0.55, 0.05, help="Ms estimate input.")
        mo_c = st.number_input("Mo", 0.0, 1.5, 0.25, 0.05, help="Ms estimate input.")
        preset = composition_to_preset(
            {"C": c_c, "Mn": mn_c, "Cr": cr_c, "Ni": ni_c, "Mo": mo_c}, name="user"
        )
        alloy_label = preset["name"]

    st.markdown("**Schedule**")
    boost_c = st.slider(
        "Boost temperature (°C)", 850, 1050, 950, 5,
        help="First-stage soak setpoint. High carbon-potential, high-temperature "
        "boost drives carbon into the surface fast.",
    )
    boost_h = st.slider(
        "Boost duration (h)", 0.5, 8.0, 2.0, 0.5,
        help="Time at boost temperature. Case depth grows roughly as √t.",
    )
    diffuse_c = st.slider(
        "Diffuse temperature (°C)", 800, 1050, 930, 5,
        help="Second-stage setpoint: lets surface carbon redistribute deeper "
        "and the steep near-surface gradient relax toward the target.",
    )
    diffuse_h = st.slider(
        "Diffuse duration (h)", 0.5, 8.0, 1.0, 0.5,
        help="Time at diffuse temperature. Longer = flatter, deeper profile.",
    )
    carbon_potential = st.slider(
        "Carbon potential (mass-%)", 0.6, 1.2, 1.0, 0.01,
        help="Carbon activity of the furnace atmosphere at the surface. "
        "Higher Cp = steeper gradient, faster case build, more soot risk.",
    )
    emissivity = st.slider(
        "Emissivity", 0.3, 1.0, 0.8, 0.01,
        help="Surface radiation efficiency of the load. Drives the thermal "
        "surrogate's heating rate.",
    )
    size_mm = st.slider(
        "Part size (mm, cross-section)", 6.0, 40.0, 16.0, 1.0,
        help="Characteristic cross-section of the part. Larger parts heat "
        "slower and quench slower — both change the outcome.",
    )

    st.markdown("**Quench**")
    quench_medium = st.selectbox(
        "Quench medium",
        ["oil", "water", "polymer", "air"],
        index=0,
        help="Real quenches are finite-rate. Oil ≈ 900 W/m²K film, water ≈ 3500, "
        "polymer ≈ 1800, air ≈ 50. Slow media form bainite/pearlite where the "
        "local cooling rate drops below the C-curve noses: the surface stays "
        "martensitic, the CORE collapses — a depth-dependent profile, not a "
        "uniform softening.",
    )
    quench_temp_c = st.slider(
        "Quench bath temperature (°C)", 20, 120, 60, 5,
        help="Bath temperature. Higher bath = slower final cooling.",
    )
    quench_agitation = st.slider(
        "Agitation", 0.0, 1.0, 0.5, 0.05,
        help="Scales the effective film coefficient (0 = still, 1 = vigorous). "
        "More agitation = faster quench = more martensite.",
    )

    with st.expander("Advanced — surface mass-transfer coefficient (h_m)"):
        h_m = st.number_input(
            "h_m (m/s, log-scale)",
            min_value=1e-6, max_value=1e-2, value=1.0e-4, step=1e-6,
            format="%.2e",
            help="Surface gas-to-part mass-transfer coefficient. The furnace "
            "tab now uses the SAME mass_transfer (Robin) BC as the calibration "
            "path, so a calibrated h_m replays identically here. At long soaks "
            "the surface saturates near C_pot regardless, so h_m mostly "
            "matters for short boost stages and the approach transient.",
        )

    run = st.button("Run emulation", type="primary", help="Recompute the full forward pass.")

    # Hardenability readout (Grossmann DI) — answers "will it through-harden?"
    try:
        from ferrumizer_physics.alloys import load_alloy, through_hardening_verdict

        th_preset = preset if preset is not None else load_alloy(alloy_choice)
        hv = through_hardening_verdict(th_preset, size_mm)
        st.divider()
        st.markdown("**Hardenability (Grossmann DI)**")
        st.caption(f"DI ≈ {hv['di_mm']:.1f} mm vs section {hv['section_mm']:.0f} mm")
        st.markdown(f"**{hv['verdict']}**")
        st.caption(hv["caveat"])
    except Exception:  # noqa: BLE001
        pass  # hardenability is informational; never block the emulation

# --------------------------------------------------------------------------- #
# Tab 1 — Virtual Furnace (live schedule explorer)
# --------------------------------------------------------------------------- #
tab_furnace, tab_predict, tab_cct, tab_ingest = st.tabs(
    ["Virtual Furnace", "Cycle Predictor", "CCT Diagram", "Log Ingestion"]
)

with tab_furnace:
    st.markdown(
        "Drag the schedule and quench controls, then press **Run emulation**. "
        "Every curve below is the solution of the same physics model the CLI "
        "and calibration use — fixed axes so you see the *magnitude* of each "
        "change, not an auto-rescaled silhouette."
    )
    if run or "result" not in st.session_state:
        with st.spinner("Solving thermal history, carbon diffusion and hardening..."):
            result = run_emulation(
                preset,
                alloy_label,
                boost_c,
                boost_h,
                diffuse_c,
                diffuse_h,
                carbon_potential,
                emissivity,
                size_mm,
                quench_medium,
                quench_temp_c,
                quench_agitation,
                h_m=h_m,
            )
        st.session_state.result = result

    result = st.session_state.result

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Process history")
        thermal = result["thermal"]
        t_s = thermal.get("times_s", np.linspace(0, 1, len(np.asarray(thermal["Ts"]))))
        fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=CHARCOAL)
        _style_ax(ax, "T (K)", TEMP_YMAX)
        ax.plot(np.asarray(t_s), np.asarray(thermal["Ts"]),
                color=EMBER, lw=2, label="surface")
        if "Tcore" in thermal:
            ax.plot(np.asarray(t_s), np.asarray(thermal["Tcore"]),
                    color=GOLD, lw=2, label="core")
        ax.legend(facecolor=CHARCOAL, labelcolor=CREAM, fontsize=8)
        st.pyplot(fig)
        st.caption(
            "Furnace temperature history. The boost segment holds the part near "
            "the high setpoint; the diffuse segment lets the profile redistribute. "
            "Core lags surface on heating (thermal inertia)."
        )

        st.subheader("Carbon profile (end of cycle)")
        fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=CHARCOAL)
        _style_ax(ax, "C (mass-%)", CARBON_YMAX)
        ax.plot(np.asarray(result["x_mm"]), np.asarray(result["carbon"]["C_final"]),
                color=GOLD, lw=2)
        ax.axhline(carbon_potential, color=CREAM, ls="--", lw=0.8, alpha=0.5)
        st.pyplot(fig)
        st.caption(
            "Carbon concentration vs depth at cycle end. The dashed line is the "
            "atmosphere carbon potential. A steeper near-surface gradient is what "
            "short soak times produce; diffuse stages flatten it."
        )

        st.subheader("Hardness profile")
        fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=CHARCOAL)
        _style_ax(ax, "H (HV)", HARD_YMAX)
        ax.plot(np.asarray(result["x_mm"]), np.asarray(result["H"]), color=CREAM, lw=2)
        ax.axhline(550.0, color=EMBER, ls="--", lw=1.0)
        ax.text(0.02, 560, "550 HV (ISO 2639 ECD threshold)", color=EMBER, fontsize=8)
        st.pyplot(fig)
        st.caption(
            "Vickers hardness vs depth. The case-depth threshold at 550 HV is "
            "drawn in red. Where this curve sits relative to the threshold is the "
            "entire product decision: too shallow = early fatigue failure; "
            "too deep = wasted cycle time and distortion."
        )

        if "quench" in result:
            st.subheader("Phase fractions across the section (CCT-style)")
            fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=CHARCOAL)
            _style_ax(ax, "volume fraction", 1.0)
            xq = np.asarray(result["x_mm"])
            q = result["quench"]
            ax.plot(xq, np.asarray(q["f_martensite"]), color=CREAM, lw=2, label="martensite")
            ax.plot(xq, np.asarray(q["X_pearlite"]), color=EMBER, lw=2, label="pearlite")
            ax.plot(xq, np.asarray(q["X_bainite"]), color=GOLD, lw=2, label="bainite")
            ax.legend(facecolor=CHARCOAL, labelcolor=CREAM, fontsize=8)
            st.pyplot(fig)
            st.caption(
                "Each depth has its own cooling curve, so each depth forms a "
                "different phase mix. Fast-cooling surface keeps martensite; "
                "slow-cooling core can form pearlite/bainite. This is the "
                "'CCT-style' answer: the model now distinguishes local cooling "
                "rates instead of one part-average curve."
            )

    with right:
        st.subheader("Case-Depth Dial")
        st.pyplot(dial_gauge(float(result["ecd_mm"])))
        st.metric("Surface hardness", f"{float(result['H'][0]):.0f} HV")
        st.metric("Core hardness", f"{float(result['H'][-1]):.0f} HV")
        if "quench" in result:
            q = result["quench"]
            st.metric("Surface martensite", f"{float(result['f_martensite'][0]) * 100:.0f} %")
            st.metric("Core martensite", f"{float(result['f_martensite'][-1]) * 100:.0f} %")
            st.metric("Surface pearlite", f"{float(q['X_pearlite'][0]) * 100:.1f} %")
            st.caption(
                "With the finite-rate, depth-resolved quench model, each depth "
                "has its own cooling rate: the surface cools fast and keeps "
                "martensite, the core cools slower and can form pearlite/"
                "bainite. Compare oil (slow) vs water (fast) — and note how "
                "the phase fractions vary across the section in the profile "
                "plot below."
            )
        else:
            st.caption("Instant-quench path (legacy). Enable a quench medium to see bainite/pearlite effects.")

# --------------------------------------------------------------------------- #
# Tab 2 — Cycle Predictor (Bayesian calibration)
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Calibrate against a measured hardness traverse")
    st.markdown(
        "Upload a CSV with columns `depth_mm,hardness_HV` (surface first) **or** "
        "a raw PLC/datalogger export containing a traverse — the ingestion "
        "parser will find the depth/hardness columns automatically.\n\n"
        "Ferrumizer runs NumPyro NUTS over {log D0, Q, C_pot, h_m, eps} and "
        "reports the posterior with convergence gates (R̂ < 1.01, bulk ESS > 400)."
    )
    uploaded = st.file_uploader(
        "Traverse CSV or PLC log", type=["csv", "txt", "log"], key="traverse"
    )
    st.markdown("**Quench process** — tell the calibrator what the part actually saw.")
    cal_quench = st.selectbox(
        "Quench medium (calibration)",
        ["none (instant quench)", "oil", "water", "polymer", "air"],
        index=1,
        help="The calibration forward model now uses the SAME spatial quench "
        "physics as the Virtual Furnace. If your traverse came from an "
        "oil-quenched part, select 'oil' — otherwise the posterior will be "
        "biased to compensate for the missing quench physics and will NOT "
        "reproduce your data when replayed in the furnace tab. 'none' is the "
        "legacy instantaneous-quench assumption; use it only for "
        "full-martensite traverses.",
    )
    chains = st.number_input("Chains", 1, 4, 2, 1, help="MCMC chains (CPU cost ×chains).")
    draws = st.number_input("Draws per chain", 50, 500, 150, 10,
                            help="Posterior samples per chain.")
    warmup = st.number_input("Warmup", 50, 500, 100, 10,
                             help="Adaptation draws per chain (discarded).")
    run_cal = st.button("Run calibration", type="secondary")

    if run_cal and uploaded is None:
        st.warning("Upload a traverse CSV first.")
    if run_cal and uploaded is not None:
        depths = None
        H = None
        report = None
        try:
            from ingest.plc_parser import parse_plc_log

            content = uploaded.read().decode("utf-8", errors="replace")
            report = parse_plc_log(uploaded.name, text=content)
            if report.has_traverse and report.traverse is not None:
                trav = report.traverse
                depths = np.asarray(trav["depth_mm"], dtype=np.float64)
                H = np.asarray(trav["hardness_HV"], dtype=np.float64)
                for w in report.warnings:
                    st.info(f"Ingestion: {w}")
            else:
                # fall back to simple CSV
                data = np.genfromtxt(io.StringIO(content), delimiter=",", names=True)
                depths = np.asarray(data["depth_mm"], dtype=np.float64)
                H = np.asarray(data["hardness_HV"], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse input: {exc}")
        if depths is not None and H is not None and report is not None:
            st.info(
                "Running NUTS on a light carbon grid — on CPU this typically "
                "takes **5–20 min** for these settings. The app grid is "
                "deliberately coarser than the CLI default so the demo is "
                "interactive; for research-grade posteriors use `ferrumize "
                "calibrate` with the full grid."
            )
            from calibration.calibrate import run_calibration

            # ------------------------------------------------------------------ #
            # Use the INGESTED trajectory when the log has one (review 3 P0 #3).
            # Before this fix the calibrator hardcoded a 2 h / 950 C scenario
            # and silently ignored what the furnace actually did — a 4 h / 925 C
            # cycle was calibrated as 2 h / 950 C and the D0/Q posterior was
            # biased to compensate for the wrong thermal history.
            # ------------------------------------------------------------------ #
            if report.has_trajectory and report.trajectory is not None:
                from ingest.plc_parser import schedule_from_trajectory

                sched = schedule_from_trajectory(
                    report.trajectory["t_s"], report.trajectory["T_C"]
                )
                scenario = Scenario(
                    alloy=alloy_label,
                    t_total=float(sched["schedule_times"][-1]),
                    schedule_times=tuple(sched["schedule_times"]),
                    schedule_temps_C=tuple(sched["schedule_temps_C"]),
                    thermal_n=21,
                    carbon_n=41,      # light grid — see runtime warning above
                    carbon_dt=8.0,    # light grid
                    carbon_mode="mass_transfer",  # h_m must be exercised (see calibrate.py)
                    quench_medium=None if cal_quench.startswith("none") else cal_quench,
                    quench_temp_K=333.15,
                    quench_agitation=0.5,
                )
                st.info(
                    f"Using the ingested trajectory: {len(sched['schedule_times'])} "
                    f"schedule knots, t_total = {scenario.t_total / 3600:.2f} h, "
                    f"temps = {[round(x) for x in sched['schedule_temps_C']]} °C."
                )
            else:
                scenario = Scenario(
                    alloy=alloy_label,
                    t_total=7200.0,
                    schedule_times=(0.0, 7200.0),
                    schedule_temps_C=(950.0, 950.0),
                    thermal_n=21,
                    carbon_n=41,      # light grid — see runtime warning above
                    carbon_dt=8.0,    # light grid
                    carbon_mode="mass_transfer",  # h_m must be exercised (see calibrate.py)
                    quench_medium=None if cal_quench.startswith("none") else cal_quench,
                    quench_temp_K=333.15,
                    quench_agitation=0.5,
                )
                st.info("No trajectory in the log; using the default 2 h / 950 °C scenario.")
            with st.spinner("Running NUTS calibration (CPU)... do not close this tab."):
                mcmc, summary = run_calibration(
                    depths,
                    H,
                    scenario,
                    num_warmup=int(warmup),
                    num_samples=int(draws),
                    num_chains=int(chains),
                    seed=0,
                )
            samples = mcmc.get_samples(group_by_chain=False)
            st.success("Calibration complete.")
            gate = "GATES PASS" if summary["gates_ok"] else "GATES NOT MET"
            st.markdown(f"### Convergence: **{gate}**")
            rows = []
            for name, s in summary["params"].items():
                rows.append(
                    {
                        "param": name,
                        "mean": f"{s['mean']:.4g}",
                        "sd": f"{s['sd']:.4g}",
                        "r_hat": f"{s['r_hat']:.3f}",
                        "bulk_ess": f"{s['bulk_ess']:.0f}",
                        "gate": "OK" if s["gate_ok"] else "FAIL",
                    }
                )
            st.dataframe(rows)
            st.subheader("Posterior marginals")
            fig, axes = plt.subplots(1, len(samples), figsize=(2.4 * len(samples), 2.6), facecolor=CHARCOAL)
            if len(samples) == 1:
                axes = [axes]
            for ax, (name, vals) in zip(axes, samples.items()):
                ax.hist(np.asarray(vals), bins=20, color=GOLD, alpha=0.85)
                ax.set_title(name, color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=7)
                ax.set_facecolor(CHARCOAL)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption(
                "Posterior distributions over the five process parameters. "
                "A tight, single-peaked histogram means the traverse pins that "
                "parameter; a flat/wide one means it is not identifiable from "
                "this data (see the identifiability analysis: one schedule "
                "leaves D0 and Q tangled)."
            )

            # Posterior predictive check: does the posterior actually
            # reproduce the observed traverse? (review 2: the first thing a
            # scientist checks — overlay predicted vs observed hardness.)
            try:
                from calibration.calibrate import posterior_predictive_hardness

                ppc = posterior_predictive_hardness(mcmc, depths, scenario, n_draws=120)
                fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=CHARCOAL)
                ax.set_facecolor(CHARCOAL)
                ax.plot(ppc["obs_depths"], H, "o", color=GOLD, ms=5, label="observed")
                ax.plot(ppc["obs_depths"], ppc["H_mean"], color=CREAM, lw=2, label="posterior mean")
                ax.fill_between(
                    ppc["obs_depths"], ppc["H_lo"], ppc["H_hi"],
                    color=CREAM, alpha=0.25, label="5-95% credible band",
                )
                ax.set_xlabel("depth (mm)", color="white", fontsize=9)
                ax.set_ylabel("hardness (HV)", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#444444")
                ax.legend(facecolor=CHARCOAL, labelcolor=CREAM, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                resid = np.asarray(H) - ppc["H_mean"]
                st.caption(
                    f"Posterior predictive check — max |residual| = "
                    f"{np.max(np.abs(resid)):.0f} HV. If the observed points "
                    "fall outside the credible band, the model+prior cannot "
                    "explain the traverse: check the quench medium selection, "
                    "the schedule, or the alloy preset."
                )

                # residual structure check (review 2): are residuals random?
                fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=CHARCOAL)
                ax.set_facecolor(CHARCOAL)
                ax.axhline(0.0, color="#444444", lw=1)
                ax.plot(depths, resid, "o-", color=EMBER, ms=4, lw=1)
                ax.set_xlabel("depth (mm)", color="white", fontsize=9)
                ax.set_ylabel("residual (HV)", color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#444444")
                fig.tight_layout()
                st.pyplot(fig)
                slope = np.polyfit(depths, resid, 1)[0]
                st.caption(
                    f"Residuals vs depth — trend slope {slope:+.2f} HV/mm. "
                    "A large nonzero slope means the model systematically "
                    "over/under-predicts with depth (misspecified physics or "
                    "wrong quench medium), not just measurement scatter. "
                    "Random scatter about zero is what a good fit looks like."
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Posterior predictive check unavailable: {exc}")
    elif run_cal:
        st.warning("Upload a traverse CSV first.")

# --------------------------------------------------------------------------- #
# Tab 3 — CCT Diagram (the metallurgist's view: noses + cooling curves)
# --------------------------------------------------------------------------- #
with tab_cct:
    st.subheader("CCT-style diagram: transformation noses + cooling curves")
    st.markdown(
        "The C-curves are the isothermal transformation start lines "
        "(Scheil-JMAK, 1 % transformed) for pearlite and bainite — the same "
        "kinetics the quench model integrates. The overlaid lines are the "
        "actual computed cooling curves at **surface, mid-radius and core** "
        "from the last Virtual Furnace run. Where a cooling curve crosses a "
        "nose, that phase starts to form; martensite forms only where the "
        "curve dives below Ms before crossing a nose."
    )
    if "result" in st.session_state and "quench" in st.session_state.result:
        result = st.session_state.result
        q = result["quench"]
        from ferrumizer_physics.hardening import (
            BAINITE_NOSE_K,
            PEARLITE_NOSE_K,
            ms_andrews,
            ttt_start_times,
        )

        preset = result.get("_preset")
        try:
            th_preset = preset if preset is not None else load_alloy(alloy_choice)
        except Exception:  # noqa: BLE001
            th_preset = load_alloy("8620")
        curve = ttt_start_times(th_preset, X=0.01)

        fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=CHARCOAL)
        ax.set_facecolor(CHARCOAL)
        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(200, 1200)
        ax.set_xlabel("time (s, log)", color="white", fontsize=10)
        ax.set_ylabel("temperature (K)", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")

        T = np.asarray(curve["T"])
        ax.plot(np.asarray(curve["t_pearlite_s"]), T, color=EMBER, lw=2.2,
                label=f"pearlite start (nose {PEARLITE_NOSE_K - 273.15:.0f} °C)")
        ax.plot(np.asarray(curve["t_bainite_s"]), T, color=GOLD, lw=2.2,
                label=f"bainite start (nose {BAINITE_NOSE_K - 273.15:.0f} °C)")

        # Ms lines (surface/core carbon)
        C_final = np.asarray(result["carbon"]["C_final"])
        C_surf = float(C_final[0])
        C_core = float(C_final[-1])
        Ms_surf = float(ms_andrews(C_surf, th_preset["ms"]["A"], th_preset["ms"]["b_carbon"]))
        Ms_core = float(ms_andrews(C_core, th_preset["ms"]["A"], th_preset["ms"]["b_carbon"]))
        ax.axhline(Ms_surf, color="#8aa07f", ls=":", lw=1.4,
                   label=f"Ms (surface C={C_surf:.2f})")
        ax.axhline(Ms_core, color="#8aa07f", ls="--", lw=1.4,
                   label=f"Ms (core C={C_core:.2f})")

        # cooling curves at surface / mid / core from the spatial quench
        Thist = np.asarray(q["cooling_history"])
        ts = np.asarray(q["cooling_times_s"])
        n_therm = Thist.shape[1]
        mid = n_therm // 2
        core = n_therm - 1
        ax.plot(ts, Thist[:, 0], color=CREAM, lw=2.0, label="cooling: surface")
        ax.plot(ts, Thist[:, mid], color="#b0b8c0", lw=2.0, ls="--", label="cooling: mid-radius")
        ax.plot(ts, Thist[:, core], color="#6f7780", lw=2.0, ls="-.", label="cooling: core")

        ax.legend(facecolor=CHARCOAL, labelcolor=CREAM, fontsize=8, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Read it like a metallurgist: a cooling curve that stays right of a "
            "nose for a long time accumulates diffusional fraction (soft); one "
            "that plunges through before the nose has time to act lands in "
            "martensite (hard). The surface curve usually dives steeply; the "
            "core curve crosses the pearlite nose — exactly the depth-dependent "
            "softening the profile plots show."
        )
        st.markdown(
            "**Honest caveat:** the noses are Gaussian surrogates fitted to "
            "order-of-magnitude CCT kinetics and validated against the 8620 "
            "Jominy hardenability band (verification gate V8). They are "
            "engineering approximations, not measured TTT diagrams — use them "
            "for ranking quenches, not for certification."
        )
    else:
        st.info(
            "Run an emulation in the Virtual Furnace tab first — the cooling "
            "curves are taken from the last computed quench."
        )

# --------------------------------------------------------------------------- #
# Tab 4 — Log Ingestion (PLC/datalogger parser preview)
# --------------------------------------------------------------------------- #
with tab_ingest:
    st.subheader("Ingest a furnace PLC / datalogger export")
    st.markdown(
        "Paste or upload a raw export (company banners, mixed units, quoted "
        "cells, junk rows are all handled). The parser auto-detects the "
        "delimiter, header row, column roles and deg C / deg F, and compresses "
        "the trajectory into soak segments. Nothing here is faked: it is the "
        "same `ferrumize ingest` code path."
    )
    log_file = st.file_uploader("PLC log (CSV/TXT/LOG)", type=["csv", "txt", "log"], key="plc")
    if log_file is not None:
        from ingest.plc_parser import parse_plc_log, schedule_from_trajectory

        content = log_file.read().decode("utf-8", errors="replace")
        report = parse_plc_log(log_file.name, text=content)
        st.info(f"Rows used: {report.rows_used}/{report.rows_total} · temp unit: {report.temperature_unit}")
        for w in report.warnings:
            st.warning(w)
        if report.has_trajectory:
            traj = report.trajectory
            assert traj is not None
            st.subheader("Extracted trajectory")
            st.line_chart(
                {
                    "T (°C)": traj["T_C"],
                }
            )
            sched = schedule_from_trajectory(traj["t_s"], traj["T_C"])
            st.subheader("Compressed schedule (RDP knots)")
            st.dataframe(
                {
                    "time (s)": [round(x) for x in sched["schedule_times"]],
                    "setpoint (°C)": [round(x) for x in sched["schedule_temps_C"]],
                }
            )
            st.caption(
                "Ramer-Douglas-Peucker line simplification preserves heating "
                "and cooling ramps as diagonal segments (linear interpolation "
                "reproduces them exactly) instead of chopping them into a "
                "staircase of flat soaks. These knots feed directly into the "
                "Scenario — the Cycle Predictor tab now calibrates against "
                "what the furnace actually did, not a hardcoded 2 h / 950 °C "
                "default."
            )
        if report.has_traverse and report.traverse is not None:
            trav_ingest = report.traverse
            st.subheader("Extracted traverse")
            st.dataframe(
                {
                    "depth (mm)": trav_ingest["depth_mm"],
                    "hardness (HV)": trav_ingest["hardness_HV"],
                }
            )
        if not report.has_trajectory and not report.has_traverse:
            st.error("No recognizable time/temperature or depth/hardness columns found.")
