"""Hardening stage Tesseract: JMAK/Scheil, Koistinen-Marburger, hardness, ECD.

Fully differentiable via JAX (autodiff-native box). Computes martensite-start
temperature (Andrews), martensite fraction (Koistinen-Marburger), hardness
via smoothstep mixing rule, and effective case depth (ECD at 550 HV, ISO 2639).
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent / "shared"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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

from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    jmak_scheil_fraction,
    km_fraction,
    ms_andrews,
)

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    C_profile: Differentiable[Array[(None,), Float64]] = Field(
        description="Carbon profile, mass-%, ordered surface -> core."
    )
    x_mm: Differentiable[Array[(None,), Float64]] = Field(
        description="Depth from surface, mm, ordered surface -> core."
    )
    T_quench: Differentiable[Float64] = Field(
        default=298.15, description="Quench / room temperature, K."
    )
    # Alloy preset scalars (static, not differentiated)
    ms_A: float = Field(default=833.0, description="Andrews Ms intercept, K.")
    ms_b_carbon: float = Field(default=240.0, description="Andrews Ms carbon slope, K/wt%.")
    km_alpha: float = Field(default=0.011, description="Koistinen-Marburger alpha, 1/K.")
    Hcore: float = Field(default=230.0, description="Core hardness, HV.")
    Hmax: float = Field(default=650.0, description="Max surface hardness, HV.")
    Cmin: float = Field(default=0.5, description="Carbon for hardness onset, mass-%.")
    Cideal: float = Field(default=1.0, description="Carbon for hardness saturation, mass-%.")
    ecd_threshold_hv: float = Field(default=550.0, description="ECD hardness threshold, HV.")
    # Optional JMAK diffusional transformation inputs
    T_cooling_history: Array[(None,), Float64] | None = Field(
        default=None, description="Optional cooling curve (K) for JMAK Scheil fraction."
    )
    dt_cooling: float = Field(default=1.0, description="Time step of cooling curve, s.")
    jmak_n: float = Field(default=2.0, description="JMAK exponent.")
    jmak_k_ref: float = Field(default=8.5e-9, description="JMAK peak rate constant, 1/s.")
    jmak_T_nose: float = Field(default=823.15, description="JMAK C-curve nose temperature, K.")
    jmak_width: float = Field(default=80.0, description="JMAK C-curve Gaussian width, K.")


class OutputSchema(BaseModel):
    Ms: Differentiable[Array[(None,), Float64]] = Field(
        description="Martensite-start temperature profile, K."
    )
    f_martensite: Differentiable[Array[(None,), Float64]] = Field(
        description="Martensite volume fraction profile."
    )
    H: Differentiable[Array[(None,), Float64]] = Field(
        description="Hardness profile, HV, surface -> core."
    )
    ecd_mm: Differentiable[Float64] = Field(
        description="Effective case depth at threshold hardness, mm."
    )
    X_diffusional: Differentiable[Float64] = Field(
        description="JMAK/Scheil diffusional transformed fraction (0 if no cooling curve)."
    )


def _preset_from_dict(d: dict) -> dict:
    return {
        "ms": {"A": d["ms_A"], "b_carbon": d["ms_b_carbon"]},
        "km_alpha": d["km_alpha"],
        "hardness": {
            "Hcore": d["Hcore"],
            "Hmax": d["Hmax"],
            "Cmin": d["Cmin"],
            "Cideal": d["Cideal"],
        },
        "ecd_threshold_hv": d["ecd_threshold_hv"],
        "jmak": {
            "n": d["jmak_n"],
            "k_pearlite": d["jmak_k_ref"],
            "T_nose": d["jmak_T_nose"],
            "width": d["jmak_width"],
        },
    }


def _forward_dict(d: dict) -> dict:
    C = jnp.asarray(d["C_profile"], dtype=jnp.float64)
    x_mm = jnp.asarray(d["x_mm"], dtype=jnp.float64)
    Tq = jnp.asarray(d["T_quench"], dtype=jnp.float64)
    preset = _preset_from_dict(d)

    Ms = ms_andrews(C, preset["ms"]["A"], preset["ms"]["b_carbon"])
    f_mart = km_fraction(Ms, Tq, preset["km_alpha"])
    H = hardness_profile(C, preset, f_mart)
    ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

    # JMAK diffusional fraction (optional)
    T_hist = d.get("T_cooling_history")
    if T_hist is not None:
        jmak = preset["jmak"]
        X_diff = jmak_scheil_fraction(
            jnp.asarray(T_hist, dtype=jnp.float64),
            d["dt_cooling"],
            n_exp=jmak["n"],
            k_ref=jmak["k_pearlite"],
            T_nose=jmak["T_nose"],
            width=jmak["width"],
        )
    else:
        X_diff = jnp.asarray(0.0, dtype=jnp.float64)

    return {
        "Ms": Ms,
        "f_martensite": f_mart,
        "H": H,
        "ecd_mm": ecd,
        "X_diffusional": X_diff,
    }


@eqx.filter_jit
def _forward_jit(inputs: dict) -> dict:
    return _forward_dict(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the hardening stage forward computation."""
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
