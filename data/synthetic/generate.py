"""Generate deterministic synthetic hardness traverses for calibration demos."""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "components" / "shared"))
sys.path.insert(0, str(ROOT / "app"))

from ferrumize.models import fast_forward
from ferrumizer_physics.alloys import load_alloy


def generate(seed: int = 20260822) -> None:
    out = Path(__file__).resolve().parent
    preset = load_alloy("8620")
    th = preset["thermal"]
    rng = np.random.default_rng(seed)
    planted = {"D0": 2.2e-5, "Q_kJ": 137.0, "C_pot": 1.0, "h_m": 1e-4, "eps": 0.8}
    depths = np.linspace(0.0, 2.0, 21)
    rows = []
    schedule_temps = {}
    for label, temp in (("low", 900.0), ("high", 1000.0)):
        total = 7200.0
        knots = jnp.array([[0.0, total], [temp, temp]], dtype=jnp.float64)
        result = fast_forward(
            jnp.log(planted["D0"]), planted["Q_kJ"], planted["C_pot"], planted["h_m"], planted["eps"],
            schedule_knots=knots, t_total=total, T_init_K=298.15, T_quench=298.15,
            h_conv=20.0, k=th["k"], rho_cp=th["rho"] * th["cp"],
            half_thickness_m=0.008, x_half_mm=8.0, carbon_n=81, carbon_dt=2.0,
            carbon_mode="mass_transfer", preset=preset, n_T_samples=120,
        )
        hardness = np.interp(depths, np.asarray(result["x_mm"]), np.asarray(result["H"]))
        hardness += rng.normal(0.0, 10.0, size=hardness.shape)
        path = out / f"traverse_{label}.csv"
        with path.open("w") as handle:
            handle.write("depth_mm,hardness_HV\n")
            for depth, hv in zip(depths, hardness):
                handle.write(f"{depth:.6f},{hv:.6f}\n")
        rows.append((path, temp))
        schedule_temps[label] = temp
    # The YAML must describe the data it references: two schedules at 900 C
    # and 1000 C (the identifiability protocol that breaks D0-Q collinearity).
    # A single-schedule 950 C line would contradict the generated CSVs.
    yaml = out / "calibration_data.yaml"
    yaml.write_text(
        "alloy: 8620\n"
        "geometry: slab\n"
        "size_mm: 16.0\n"
        "t_total: 7200.0\n"
        "# Two-schedule identifiability protocol (see ADR-002 / figure F8):\n"
        "# a single schedule leaves D0 and Q collinear; the second temperature\n"
        "# breaks the degeneracy. traverse_low.csv is the 900 C schedule,\n"
        "# traverse_high.csv the 1000 C schedule.\n"
        "schedule:\n  times: [0.0, 7200.0]\n  temps_C: [900.0, 900.0]\n"
        "schedule2:\n  times: [0.0, 7200.0]\n  temps_C: [1000.0, 1000.0]\n"
        "thermal:\n  n: 21\n  sample_every: 20\n"
        "carbon:\n  n: 41\n  dt: 1.0\n  sample_every: 300\n  mode: mass_transfer\n"
        "params:\n  D0: 2.2e-5\n  Q_kJ: 137.0\n  C_pot: 1.0\n  h_m: 1.0e-4\n  eps: 0.8\n"
        "observations:\n"
        "  traverse_csv: traverse_low.csv\n"
        "  traverse_csv2: traverse_high.csv\n"
    )
    sums = []
    for path in sorted(out.glob("*.csv")):
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    sums.append(f"{hashlib.sha256(yaml.read_bytes()).hexdigest()}  {yaml.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    generate()
