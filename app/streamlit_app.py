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
        quench_medium=quench_medium,
        quench_temp_K=quench_temp_c + 273.15,
        quench_agitation=quench_agitation,
    )
    params = ProcessParams(C_pot=carbon_potential, eps=emissivity)
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
        "polymer ≈ 1800, air ≈ 50. Slow media let bainite/pearlite form and "
        "case hardness collapses — the model now predicts that.",
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

    run = st.button("Run emulation", type="primary", help="Recompute the full forward pass.")

# --------------------------------------------------------------------------- #
# Tab 1 — Virtual Furnace (live schedule explorer)
# --------------------------------------------------------------------------- #
tab_furnace, tab_predict, tab_ingest = st.tabs(
    ["Virtual Furnace", "Cycle Predictor", "Log Ingestion"]
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

    with right:
        st.subheader("Case-Depth Dial")
        st.pyplot(dial_gauge(float(result["ecd_mm"])))
        st.metric("Surface hardness", f"{float(result['H'][0]):.0f} HV")
        st.metric("Core hardness", f"{float(result['H'][-1]):.0f} HV")
        if "quench" in result:
            q = result["quench"]
            st.metric("Surface martensite", f"{float(result['f_martensite'][0]) * 100:.0f} %")
            st.metric("Diffusional phases", f"{float(q['X_diffusional']) * 100:.1f} %")
            st.caption(
                "With the finite-rate quench model, a slow quench converts "
                "austenite to bainite/pearlite and the dial drops — the real "
                "production failure mode. Compare oil (slow) vs water (fast)."
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
        try:
            from ingest.plc_parser import parse_plc_log

            content = uploaded.read().decode("utf-8", errors="replace")
            report = parse_plc_log(uploaded.name, text=content)
            if report.has_traverse:
                trav = report.traverse
                assert trav is not None
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
        if depths is not None and H is not None:
            st.info(
                "Running NUTS on a light carbon grid — on CPU this typically "
                "takes **5–20 min** for these settings. The app grid is "
                "deliberately coarser than the CLI default so the demo is "
                "interactive; for research-grade posteriors use `ferrumize "
                "calibrate` with the full grid."
            )
            from calibration.calibrate import run_calibration

            scenario = Scenario(
                alloy=alloy_label,
                t_total=7200.0,
                schedule_times=(0.0, 7200.0),
                schedule_temps_C=(950.0, 950.0),
                thermal_n=21,
                carbon_n=41,      # light grid — see runtime warning above
                carbon_dt=8.0,    # light grid
            )
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
    elif run_cal:
        st.warning("Upload a traverse CSV first.")

# --------------------------------------------------------------------------- #
# Tab 3 — Log Ingestion (PLC/datalogger parser preview)
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
            st.subheader("Compressed schedule (soak segments)")
            st.dataframe(
                {
                    "time (s)": [round(x) for x in sched["schedule_times"]],
                    "setpoint (°C)": [round(x) for x in sched["schedule_temps_C"]],
                }
            )
            st.caption(
                "These segments feed directly into the Scenario schedule knots "
                "— compare a planned cycle against what the furnace actually did."
            )
        if report.has_traverse:
            trav = report.traverse
            assert trav is not None
            st.subheader("Extracted traverse")
            st.dataframe(
                {
                    "depth (mm)": trav["depth_mm"],
                    "hardness (HV)": trav["hardness_HV"],
                }
            )
        if not report.has_trajectory and not report.has_traverse:
            st.error("No recognizable time/temperature or depth/hardness columns found.")
