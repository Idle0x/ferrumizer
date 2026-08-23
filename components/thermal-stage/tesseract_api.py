"""Thermal stage Tesseract: 1-D heat conduction with radiative Robin BC.

Solves dT/dt = alpha * d2T/dx2 on a slab or cylinder cross-section with
boundary condition  -k dT/dx = h (T_furn - Ts) + eps*sigma*(T_furn^4 - Ts^4).
Fully differentiable via JAX (autodiff-native box).
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent / "shared"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

from ferrumizer_physics.thermal import ThermalConfig, run_thermal

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    geometry: Literal["slab", "cylinder"] = Field(default="slab")
    size_mm: float = Field(default=16.0, gt=0)
    n: int = Field(default=161, ge=3)
    dt: float = Field(default=5.0e-4, gt=0)
    t_total: float = Field(default=3600.0, gt=0)
    schedule_times: Array[(None,), Float64] = Field(
        description="Schedule knot times, s (first must be 0)."
    )
    schedule_temps_C: Array[(None,), Float64] = Field(
        description="Schedule setpoint temperatures, deg C."
    )
    k: float = Field(default=42.0, gt=0)
    rho: float = Field(default=7800.0, gt=0)
    cp: float = Field(default=700.0, gt=0)
    eps: Differentiable[Float64] = Field(default=0.8)
    h: float = Field(default=20.0, ge=0)
    convective_model: Literal["constant_h", "h_T_correlation"] = Field(default="constant_h")
    T_init_K: float = Field(default=298.15)
    sample_every: int = Field(default=200, ge=1)


class OutputSchema(BaseModel):
    times_s: Array[(None,), Float64]
    T_surface: Differentiable[Array[(None,), Float64]]
    T_core: Differentiable[Array[(None,), Float64]]
    T_field_final: Differentiable[Array[(None,), Float64]]
    cooling_rate_surface: Differentiable[Array[(None,), Float64]]


def _effective_h(h, convective_model: str, T_furn_K):
    if convective_model == "h_T_correlation":
        dT = jnp.maximum(jnp.abs(T_furn_K - 298.15), 1.0)
        return h * (dT / 925.0) ** 0.25
    return h


def _forward_dict(d: dict) -> dict:
    schedule_knots = jnp.stack(
        [jnp.asarray(d["schedule_times"]), jnp.asarray(d["schedule_temps_C"])]
    )
    alpha = d["k"] / (d["rho"] * d["cp"])
    h_eff = _effective_h(d["h"], d["convective_model"], d["schedule_temps_C"][-1] + 273.15)
    cfg = ThermalConfig(
        geometry=d["geometry"],
        size_mm=d["size_mm"],
        n=d["n"],
        dt=d["dt"],
        t_total=d["t_total"],
        alpha=alpha,
        h=h_eff,
        eps=d["eps"],
        k=d["k"],
        T_init_K=d["T_init_K"],
        sample_every=d["sample_every"],
    )
    out = run_thermal(schedule_knots, cfg)
    Ts = out["Ts"]
    dt_sample = cfg.dt * cfg.sample_every
    cooling_rate = jnp.gradient(Ts, dt_sample)
    return {
        "times_s": out["times_s"],
        "T_surface": Ts,
        "T_core": out["Tcore"],
        "T_field_final": out["T_final"],
        "cooling_rate_surface": cooling_rate,
    }


@eqx.filter_jit
def _forward_jit(inputs: dict) -> dict:
    return _forward_dict(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the thermal stage forward simulation."""
    out = jax_apply(_forward_jit, inputs)
    return OutputSchema(**out)


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(_forward_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return jax_jvp(_forward_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return jax_vjp(_forward_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(_forward_jit, abstract_inputs)
