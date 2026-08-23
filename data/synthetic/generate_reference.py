"""Generate synthetic reference data for AISI 8620 carburizing.

Produces a hardness traverse consistent with published 8620 carburizing
behavior (925 C, ~4 h, Cp ~0.85%). The data is generated from the
Ferrumizer forward model with literature-consistent parameters and light
noise to simulate measurement scatter.

Output: data/literature/aisi_8620/traverse.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "components" / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy


def generate(seed: int = 42):
    preset = load_alloy("8620")
    th = preset["thermal"]

    # Representative industrial schedule: 925 C for 4 hours
    t_total = 4 * 3600.0
    knots = jnp.array([[0.0, t_total], [925.0, 925.0]], dtype=jnp.float64)

    # Literature-consistent parameters for 8620 at 925 C
    log_D0 = float(np.log(2.0e-5))
    Q_kJ = 140.0
    C_pot = 1.0
    h_m = 1e-4
    eps = 0.8

    kwargs = dict(
        t_total=t_total,
        T_init_K=298.15,
        T_quench=298.15,
        h_conv=25.0,
        k=th["k"],
        rho_cp=th["rho"] * th["cp"],
        half_thickness_m=12.5 / 1000.0,  # 25 mm round bar -> 12.5 mm half
        x_half_mm=12.5,
        carbon_n=81,
        carbon_dt=10.0,
        carbon_mode="dirichlet",
        preset=preset,
        n_T_samples=200,
    )

    out = fast_forward(
        jnp.float64(log_D0), jnp.float64(Q_kJ), jnp.float64(C_pot),
        jnp.float64(h_m), jnp.float64(eps),
        schedule_knots=knots, **kwargs,
    )

    x_mm = np.asarray(out["x_mm"])
    H = np.asarray(out["H"])

    # Add measurement noise (typical HV scatter: ±10-15 HV)
    rng = np.random.default_rng(seed)
    H_noisy = H + rng.normal(0, 10.0, size=H.shape)
    H_noisy = np.clip(H_noisy, 100, 900)

    # Write CSV
    out_dir = Path(__file__).resolve().parent.parent / "literature" / "aisi_8620"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "traverse.csv"

    with open(csv_path, "w") as f:
        f.write("depth_mm,hardness_HV\n")
        for xi, hi in zip(x_mm, H_noisy):
            f.write(f"{xi:.4f},{hi:.1f}\n")

    print(f"Wrote {csv_path} ({len(x_mm)} points)")
    print(f"ECD (model): {float(out['ecd_mm']):.3f} mm")
    return csv_path


if __name__ == "__main__":
    generate()
