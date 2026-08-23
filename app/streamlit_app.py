"""Practitioner-facing Streamlit demo for Ferrumizer (PRODUCT_SPEC §10).

Sidebar: alloy selector, boost/diffuse/quench schedule editors, unknown
sliders. Main: live T(t), C(x), H(x) + ECD shown on a dial gauge styled as
the Case-Depth Dial mark (brand<->UI loop). Calibrate tab accepts an uploaded
hardness traverse and renders the NUTS posterior.
"""

from __future__ import annotations

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

# Brand palette (mirrors app/ferrumize/figures.py)
CHARCOAL = "#1c1b18"
INK = "#0d0c0b"
CREAM = "#efe9dd"
GOLD = "#d6b57c"
EMBER = "#c1502e"

st.set_page_config(page_title="ferrumizer", page_icon="◉", layout="wide")
st.title("ferrumizer — the differentiable heat-treatment engine")
st.caption("Case-Depth Dial · gas carburizing end-to-end · ISO 2639 practice (550 HV)")


def dial_gauge(ecd_mm: float, max_mm: float = 2.0, threshold: float = 550.0):
    """Semicircular dial gauge styled as the Case-Depth Dial mark."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0), facecolor=CHARCOAL)
    ax.set_facecolor(CHARCOAL)

    # gauge arc: -90 (left) -> +90 (right)
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=CREAM, lw=8, alpha=0.15)

    # colored sectors: shallow (ember), mid (gold), deep (cream)
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

    # needle
    frac = min(max(ecd_mm / max_mm, 0.0), 1.0)
    ang = np.pi * (1 - frac)
    ax.plot([0, 0.72 * np.cos(ang)], [0, 0.72 * np.sin(ang)], color="white", lw=3)
    ax.plot(0, 0, "o", color="white", ms=8, zorder=5)

    # ECD value + caption
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


def run_simulation(alloy, soak_c, soak_h, carbon_potential, emissivity, boost_c, diffuse_c):
    scenario = Scenario(
        alloy=alloy,
        t_total=soak_h * 3600.0,
        schedule_times=(0.0, soak_h * 3600.0),
        schedule_temps_C=(float(soak_c), float(soak_c)),
        thermal_n=41,
        thermal_sample_every=100,
        carbon_n=81,
        carbon_dt=2.0,
        carbon_sample_every=300,
    )
    params = ProcessParams(C_pot=carbon_potential, eps=emissivity)
    with st.spinner("Solving thermal history, carbon diffusion and hardening..."):
        result = FerrumizerPipeline(scenario, params).forward()
    return scenario, result


tab_sim, tab_cal = st.tabs(["Furnace Simulator", "Calibrate"])

with tab_sim:
    with st.sidebar:
        st.subheader("Process")
        alloy = st.selectbox("Alloy", ["8620", "9310", "5120"])
        soak_c = st.slider("Soak temperature (°C)", 850, 1050, 950, 5)
        soak_h = st.slider("Soak duration (h)", 0.5, 8.0, 2.0, 0.5)
        carbon_potential = st.slider("Carbon potential (mass-%)", 0.6, 1.2, 1.0, 0.01)
        emissivity = st.slider("Emissivity", 0.3, 1.0, 0.8, 0.01)
        run = st.button("Run simulation", type="primary")

    if run or "result" not in st.session_state:
        scenario, result = run_simulation(
            alloy, soak_c, soak_h, carbon_potential, emissivity, soak_c, soak_c
        )
        st.session_state.result = result

    result = st.session_state.result
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Process history")
        thermal = result["thermal"]
        st.line_chart(
            {
                "surface K": np.asarray(thermal["Ts"]),
                "core K": np.asarray(thermal["Tcore"]),
            }
        )
        st.subheader("Profiles")
        st.line_chart(
            {
                "carbon mass-%": np.asarray(result["carbon"]["C_final"]),
                "hardness HV": np.asarray(result["H"]),
            }
        )
    with right:
        st.subheader("Case-Depth Dial")
        st.pyplot(dial_gauge(float(result["ecd_mm"])))
        st.metric("Surface hardness", f"{float(result['H'][0]):.0f} HV")
        st.metric("Core hardness", f"{float(result['H'][-1]):.0f} HV")
        st.caption("ECD threshold: 550 HV, ISO 2639 practice")

with tab_cal:
    st.subheader("Calibrate against a measured hardness traverse")
    st.markdown(
        "Upload a CSV with columns `depth_mm,hardness_HV` (surface first). "
        "Ferrumizer runs NumPyro NUTS over {log D0, Q, C_pot, h_m, eps} and "
        "reports the posterior with convergence gates."
    )
    uploaded = st.file_uploader("Traverse CSV", type=["csv"], key="traverse")
    chains = st.number_input("Chains", 1, 4, 2, 1)
    draws = st.number_input("Draws per chain", 50, 500, 150, 10)
    warmup = st.number_input("Warmup", 50, 500, 100, 10)
    run_cal = st.button("Run calibration", type="secondary")

    if run_cal and uploaded is None:
        st.warning("Upload a traverse CSV first.")
    if run_cal and uploaded is not None:
        depths = None
        H = None
        try:
            data = np.genfromtxt(uploaded, delimiter=",", names=True)
            depths = np.asarray(data["depth_mm"], dtype=np.float64)
            H = np.asarray(data["hardness_HV"], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse CSV: {exc}")
        if depths is not None and H is not None:
            from calibration.calibrate import run_calibration

            scenario = Scenario(
                alloy=alloy,
                t_total=7200.0,
                schedule_times=(0.0, 7200.0),
                schedule_temps_C=(950.0, 950.0),
                thermal_n=21,
                carbon_n=81,
                carbon_dt=2.0,
            )
            with st.spinner("Running NUTS calibration (CPU)..."):
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
    elif run_cal:
        st.warning("Upload a traverse CSV first.")
