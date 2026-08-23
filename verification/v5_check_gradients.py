"""V5 — check_gradients on all autodiff-native boxes.

Runs the Tesseract runtime's built-in gradient checker against the thermal
and hardening stages (the two JAX/autodiff boxes). The carburizing stage uses
finite-difference parameter gradients by design and is covered by V4 instead.

Gate: zero gradient-check failures across all AD boxes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

COMPONENTS = Path(__file__).resolve().parents[1] / "components"


def _load_api(stage: str):
    path = COMPONENTS / stage / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"{stage}_api", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_v5() -> dict:
    from tesseract_core.runtime.testing.finite_differences import check_gradients

    results = {}
    all_passed = True

    # --- thermal stage ---
    thermal_api = _load_api("thermal-stage")
    thermal_inputs = {
        "geometry": "slab",
        "size_mm": 16.0,
        "n": 11,
        "dt": 0.1,
        "t_total": 60.0,
        "schedule_times": np.array([0.0, 60.0]),
        "schedule_temps_C": np.array([950.0, 950.0]),
        "k": 42.0,
        "rho": 7800.0,
        "cp": 700.0,
        "eps": 0.8,
        "h": 20.0,
        "convective_model": "constant_h",
        "T_init_K": 298.15,
        "sample_every": 10,
    }
    thermal_failures = 0
    for endpoint, failures, _ in check_gradients(
        thermal_api, {"inputs": thermal_inputs}, max_evals=20, seed=42, show_progress=False
    ):
        thermal_failures += len(failures)
    results["thermal-stage"] = thermal_failures
    all_passed = all_passed and thermal_failures == 0

    # --- hardening stage ---
    hardening_api = _load_api("hardening-stage")
    n = 21
    hardening_inputs = {
        "C_profile": np.linspace(1.0, 0.2, n),
        "x_mm": np.linspace(0.0, 8.0, n),
        "T_quench": 298.15,
        "ms_A": 833.0,
        "ms_b_carbon": 240.0,
        "km_alpha": 0.011,
        "Hcore": 230.0,
        "Hmax": 650.0,
        "Cmin": 0.5,
        "Cideal": 1.0,
        "ecd_threshold_hv": 550.0,
    }
    hardening_failures = 0
    for endpoint, failures, _ in check_gradients(
        hardening_api, {"inputs": hardening_inputs}, max_evals=20, seed=42, show_progress=False
    ):
        hardening_failures += len(failures)
    results["hardening-stage"] = hardening_failures
    all_passed = all_passed and hardening_failures == 0

    return {"failures": results, "passed": all_passed}


if __name__ == "__main__":
    r = run_v5()
    status = "PASS" if r["passed"] else "FAIL"
    print(f"V5 [{status}]")
    for box, count in r["failures"].items():
        print(f"  {box}: {count} failure(s)")
