"""Regression tests for the fixes: adaptive sub-stepping stability (N5) and
container composition gradients (G1).

These encode the two defects found during the 2026-08-22 audit:
  1. NUTS exploration of the prior corners used to produce NaN profiles
     (stability guard was skipped for tracers) -> the JAX twin now sub-steps.
  2. The container composition gradient used to crash with a
     "non-differentiable output 'x_mm'" transpose error -> x_mm is now
     declared Differentiable and gradients are verified finite and non-zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.carbon import CarburizeConfig, run_carburize, run_carburize_numpy
from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    km_fraction,
    ms_andrews,
)


def _corner_cfg() -> CarburizeConfig:
    return CarburizeConfig(n=81, dt=2.0, t_total=7200.0, mode="dirichlet", sample_every=300)


def test_jax_twin_stable_at_prior_corners():
    """Tracer path (what NUTS sees) must stay finite for unstable-prior draws."""
    cfg = _corner_cfg()
    T = np.full(64, 1223.15)

    @jax.jit
    def f(D0, Q):
        out = run_carburize(T, cfg, 0.2, D0, Q * 1000.0, 1.0, 1e-4, 8.0)
        return jnp.sum(out["C_final"])

    for D0, Q in [(9.9e-5, 100.0), (1e-4, 90.0), (2.2e-5, 137.0)]:
        v = float(f(jnp.array(D0), jnp.array(Q)))
        assert np.isfinite(v)
        g = jax.grad(f, argnums=(0, 1))(jnp.array(D0), jnp.array(Q))
        assert all(np.isfinite([float(x) for x in g]))


def test_stable_path_parity_with_numpy():
    """Stable draws stay bit-identical to the legacy NumPy stepper."""
    T = np.full(64, 1223.15)
    cfg = CarburizeConfig(n=21, dt=2.0, t_total=3600.0, mode="dirichlet", sample_every=300)
    jx = run_carburize(T, cfg, 0.2, 2.2e-5, 137000.0, 1.0, 1e-4, 8.0)
    np_out = run_carburize_numpy(T, cfg, 0.2, 2.2e-5, 137000.0, 1.0, 1e-4, 8.0)
    assert np.max(np.abs(np.asarray(jx["C_final"]) - np_out["C_final"])) < 1e-12


def test_numpy_path_still_raises_on_unstable_config():
    """The legacy box's identity is to refuse loudly (N5)."""
    T = np.full(64, 1223.15)
    cfg = CarburizeConfig(n=81, dt=2.0, t_total=7200.0, mode="dirichlet", sample_every=300)
    with pytest.raises(ValueError):
        run_carburize_numpy(T, cfg, 0.2, 9.9e-5, 100000.0, 1.0, 1e-4, 8.0)


def test_container_composition_gradient_flows():
    """Gradients must cross the composed Tesseract boundaries (G1)."""
    from ferrumize.config import load_config, scenario_from_config
    from ferrumize.pipeline import FerrumizerPipeline, ProcessParams

    cfg = load_config("data/synthetic/calibration_data.yaml")
    sc = scenario_from_config(cfg)
    pipe = FerrumizerPipeline(scenario=sc)
    p = pipe.params

    def ecd_from_eps(eps):
        p2 = ProcessParams(D0=p.D0, Q_kJ=p.Q_kJ, C_pot=p.C_pot, h_m=p.h_m, eps=eps)
        return pipe.forward_containers(p2)["hardening"]["ecd_mm"]

    eps_val = jnp.asarray(p.eps, jnp.float64)
    g_ad = float(jax.grad(ecd_from_eps)(eps_val))
    assert np.isfinite(g_ad)
    assert abs(g_ad) > 1e-12  # gradients do real work, not a silent zero

    e0 = float(ecd_from_eps(eps_val))
    e1 = float(ecd_from_eps(eps_val + 0.05))
    g_fd = (e1 - e0) / 0.05
    assert abs(g_ad - g_fd) / max(abs(g_ad), abs(g_fd), 1e-30) < 0.2
