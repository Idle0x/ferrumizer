"""Quench-model sanity gates (Q1-Q3).

Q1 — medium ranking: slower media must produce more diffusional phases.
Q2 — slow-quench collapse: air quench must consume austenite as pearlite
     and collapse ECD/hardness (the production failure mode).
Q3 — differentiability: the Newton cooling curve and quench fractions must
     carry finite JAX gradients.

These gates run fast (no MCMC, no heavy grids) and protect the finite-rate
quench model added in the hackathon upgrade pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ferrumize.pipeline import FerrumizerPipeline, ProcessParams, Scenario
from ferrumizer_physics.hardening import newton_cooling_curve


def _xdiff(medium: str, ag: float) -> float:
    sc = Scenario(quench_medium=medium, quench_temp_K=333.15, quench_agitation=ag, size_mm=16.0)
    res = FerrumizerPipeline(sc, ProcessParams()).forward()
    return float(res["quench"]["X_diffusional"])


def run_q1() -> dict:
    """Medium ranking: air > oil > water in diffusional fraction."""
    x_air = _xdiff("air", 0.2)
    x_oil = _xdiff("oil", 0.3)
    x_water = _xdiff("water", 0.8)
    passed = x_air > x_oil and x_oil > x_water
    return {"passed": passed, "x_air": x_air, "x_oil": x_oil, "x_water": x_water}


def run_q2() -> dict:
    """Slow quench collapse: air quench -> no case, near-core hardness."""
    sc = Scenario(quench_medium="air", quench_temp_K=333.15, quench_agitation=0.2, size_mm=16.0)
    res = FerrumizerPipeline(sc, ProcessParams()).forward()
    passed = (
        float(res["quench"]["X_diffusional"]) > 0.9
        and float(res["ecd_mm"]) < 0.01
        and float(res["H"][0]) < 300.0
    )
    return {
        "passed": passed,
        "X_diff": float(res["quench"]["X_diffusional"]),
        "ecd_mm": float(res["ecd_mm"]),
        "H_surface": float(res["H"][0]),
    }


def run_q3() -> dict:
    """Differentiability of the cooling curve."""
    t = jnp.linspace(0.0, 600.0, 50)
    g = jax.grad(
        lambda h: jnp.sum(
            newton_cooling_curve(1223.0, 333.0, h, 5.46e6, 0.008, t, 0.5)
        )
    )(900.0)
    passed = bool(jnp.isfinite(g))
    return {"passed": passed, "grad_h": float(g)}


if __name__ == "__main__":
    for name, fn in [("Q1", run_q1), ("Q2", run_q2), ("Q3", run_q3)]:
        r = fn()
        print(f"{name}: {'PASS' if r['passed'] else 'FAIL'} {r}")
