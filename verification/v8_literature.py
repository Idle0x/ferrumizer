"""V8 — Literature anchor: model ECD vs the reference 8620 traverse.

Loads the reference hardness traverse (data/literature/aisi_8620/traverse.csv,
provenance documented alongside), reads its ECD at the 550 HV threshold by
linear interpolation, then runs the full forward chain with the
literature-consistent parameters recorded in PROVENANCE.md and compares the
model-predicted ECD.

Gate: |ECD_model - ECD_data| <= stated digitization error (0.1 mm).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy

REPO = Path(__file__).resolve().parents[1]
DATA_CSV = REPO / "data" / "literature" / "aisi_8620" / "traverse.csv"

# Parameters exactly as recorded in data/literature/aisi_8620/PROVENANCE.md
LIT = dict(
    log_D0=float(np.log(2.0e-5)),
    Q_kJ=140.0,
    C_pot=1.0,
    h_m=1e-4,
    eps=0.8,
    t_total=4 * 3600.0,
    T_soak_C=925.0,
)
ECD_ERROR_BAR_MM = 0.1  # stated digitization uncertainty (PROVENANCE.md)


def ecd_from_traverse(
    depth_mm: np.ndarray, hardness: np.ndarray, threshold: float = 550.0
) -> float:
    """Linear-interpolation ECD from a measured traverse (surface -> core)."""
    above = hardness >= threshold
    if not above.any():
        return 0.0
    if above.all():
        return float(depth_mm[-1])
    i = int(np.max(np.where(above)[0]))
    if i >= len(depth_mm) - 1:
        return float(depth_mm[-1])
    h0, h1 = hardness[i], hardness[i + 1]
    x0, x1 = depth_mm[i], depth_mm[i + 1]
    frac = (threshold - h1) / (h0 - h1)
    return float(x1 - frac * (x1 - x0))


def run_v8() -> dict:
    data = np.genfromtxt(DATA_CSV, delimiter=",", names=True)
    depth = np.asarray(data["depth_mm"], dtype=np.float64)
    hard = np.asarray(data["hardness_HV"], dtype=np.float64)
    ecd_data = ecd_from_traverse(depth, hard)

    preset = load_alloy("8620")
    th = preset["thermal"]
    knots = jnp.array(
        [[0.0, LIT["t_total"]], [LIT["T_soak_C"], LIT["T_soak_C"]]], dtype=jnp.float64
    )
    out = fast_forward(
        jnp.float64(LIT["log_D0"]),
        jnp.float64(LIT["Q_kJ"]),
        jnp.float64(LIT["C_pot"]),
        jnp.float64(LIT["h_m"]),
        jnp.float64(LIT["eps"]),
        schedule_knots=knots,
        t_total=LIT["t_total"],
        T_init_K=298.15,
        T_quench=298.15,
        h_conv=25.0,
        k=th["k"],
        rho_cp=th["rho"] * th["cp"],
        half_thickness_m=12.5 / 1000.0,
        x_half_mm=12.5,
        carbon_n=81,
        carbon_dt=10.0,
        carbon_mode="dirichlet",
        preset=preset,
        n_T_samples=200,
    )
    ecd_model = float(out["ecd_mm"])
    delta = abs(ecd_model - ecd_data)
    return {
        "ecd_data_mm": ecd_data,
        "ecd_model_mm": ecd_model,
        "delta_mm": delta,
        "error_bar_mm": ECD_ERROR_BAR_MM,
        "passed": delta <= ECD_ERROR_BAR_MM,
    }


if __name__ == "__main__":
    r = run_v8()
    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"V8 [{status}] ECD_data={r['ecd_data_mm']:.3f} mm  "
        f"ECD_model={r['ecd_model_mm']:.3f} mm  |delta|={r['delta_mm']:.3f} mm "
        f"(error bar ±{r['error_bar_mm']:.2f} mm)"
    )
