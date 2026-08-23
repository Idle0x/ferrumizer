"""Carburizing stage Tesseract: 1-D carbon diffusion with Arrhenius D(T).

LEGACY-BY-DESIGN: the ``apply`` endpoint runs the original explicit
finite-difference stepper in plain NumPy, mimicking a legacy process
simulator that cannot be autodiffed.

Gradient strategy (declared in tesseract_config.yaml metadata):
  * ``jacobian`` endpoint: parameter-space central finite differences over
    the scalar process parameters {D0, Q_kJ_per_mol, C_pot, h_m} (<= 6 params).
    This stands in for the gradient interface of un-autodiffable commercial
    process software.
  * ``jvp``/``vjp`` endpoints: computed by differentiating the numerically
    identical JAX twin of the FD scheme (verification/cross_ad proves
    agreement to < 1e-3 relative error, gate V4). This is what lets
    end-to-end gradients cross this container boundary.
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
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64

from ferrumizer_physics.carbon import (
    CarburizeConfig,
    D_of_T,
    run_carburize,
    run_carburize_numpy,
)
from ferrumizer_physics.constants import RGAS

jax.config.update("jax_enable_x64", True)

# Parameter-space FD step fractions for the jacobian endpoint (N4: documented).
_REL_STEP = {
    "D0": 1e-5,
    "Q_kJ_per_mol": 1e-5,
    "C_pot": 1e-5,
    "h_m": 1e-4,
}


class InputSchema(BaseModel):
    T_surface_history: Differentiable[Array[(None,), Float64]] = Field(
        description="Surface temperature history (K), uniformly sampled over [0, t_total]."
    )
    x_half_mm: float = Field(default=8.0, gt=0, description="Core-to-surface distance, mm.")
    n: int = Field(default=81, ge=3, description="Nodes on the half-domain.")
    dt: float = Field(default=2.0, gt=0, description="Time step, s.")
    t_total: float = Field(default=3600.0, gt=0, description="Total time, s.")
    mode: Literal["dirichlet", "mass_transfer"] = Field(default="dirichlet")
    C0: float = Field(default=0.2, description="Initial core carbon, mass-%.")
    D0: Differentiable[Float64] = Field(default=2.2e-5, description="Diffusion prefactor, m^2/s.")
    Q_kJ_per_mol: Differentiable[Float64] = Field(
        default=137.0, description="Activation energy, kJ/mol."
    )
    C_pot: Differentiable[Float64] = Field(default=1.0, description="Carbon potential, mass-%.")
    h_m: Differentiable[Float64] = Field(
        default=1e-4, description="Mass-transfer coefficient, m/s."
    )
    sample_every: int = Field(default=1, ge=1)


class OutputSchema(BaseModel):
    C_final: Differentiable[Array[(None,), Float64]] = Field(
        description="Final carbon profile, surface -> core, mass-%."
    )
    C_hist: Array[(None, None), Float64] = Field(
        description="Sampled carbon snapshots, surface -> core."
    )
    x_mm: Differentiable[Array[(None,), Float64]] = Field(
        description="Depth from surface, mm. A fixed grid (zero Jacobian) but marked "
        "Differentiable so downstream composition (hardening ECD depends on x) can "
        "route cotangents through this boundary."
    )
    surface_flux_hist: Array[(None,), Float64] = Field(
        description="Surface carbon flux history, mass-%/s (approx)."
    )


def _cfg_from_dict(d: dict) -> CarburizeConfig:
    return CarburizeConfig(
        n=d["n"],
        dt=d["dt"],
        t_total=d["t_total"],
        mode=d["mode"],
        sample_every=d["sample_every"],
    )


def _x_mm_array(d: dict):
    return jnp.linspace(0.0, d["x_half_mm"], d["n"])


def _compute_flux(C_hist, T_samples, D0, Q_J, dx):
    """Surface flux approximation: D(T_s) * (C_surf - C_next) / dx."""
    D_s = D_of_T(D0, Q_J, T_samples)
    return D_s * (C_hist[:, 0] - C_hist[:, 1]) / dx


def _forward_jax(d: dict) -> dict:
    """JAX twin of the FD scheme — identical discretization (V4-verified)."""
    cfg = _cfg_from_dict(d)
    out = run_carburize(
        jnp.asarray(d["T_surface_history"], dtype=jnp.float64),
        cfg,
        C0=jnp.asarray(d["C0"], dtype=jnp.float64),
        D0=jnp.asarray(d["D0"], dtype=jnp.float64),
        Q_J=jnp.asarray(d["Q_kJ_per_mol"], dtype=jnp.float64) * 1000.0,
        C_pot=jnp.asarray(d["C_pot"], dtype=jnp.float64),
        hm=jnp.asarray(d["h_m"], dtype=jnp.float64),
        x_half_mm=float(d["x_half_mm"]),
    )
    n = cfg.n
    m = cfg.sample_every
    N = int(np.ceil(cfg.t_total / cfg.dt))
    N = ((N + m - 1) // m) * m
    n_out = N // m + 1

    # Sample temperatures at the same cadence as C_hist
    Ts_flat = jnp.ravel(jnp.asarray(d["T_surface_history"], dtype=jnp.float64))
    M = Ts_flat.shape[0]
    sample_indices = jnp.clip(jnp.arange(n_out) * m * M // N, 0, M - 1)
    T_samples = Ts_flat[sample_indices]

    flux = _compute_flux(out["C_hist"], T_samples, d["D0"], d["Q_kJ_per_mol"] * 1000.0, out["dx"])

    return {
        "C_final": out["C_final"],
        "C_hist": out["C_hist"],
        "x_mm": _x_mm_array(d),
        "surface_flux_hist": flux,
    }


@eqx.filter_jit
def _forward_jit(inputs: dict) -> dict:
    return _forward_jax(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the carburizing stage using the legacy NumPy FD stepper."""
    d = inputs.model_dump()
    cfg = _cfg_from_dict(d)
    out = run_carburize_numpy(
        np.asarray(d["T_surface_history"], dtype=np.float64),
        cfg,
        C0=float(d["C0"]),
        D0=float(d["D0"]),
        Q_J=float(d["Q_kJ_per_mol"]) * 1000.0,
        C_pot=float(d["C_pot"]),
        hm=float(d["h_m"]),
        x_half_mm=float(d["x_half_mm"]),
    )
    n = cfg.n
    m = cfg.sample_every
    N = int(np.ceil(cfg.t_total / cfg.dt))
    N = ((N + m - 1) // m) * m
    n_out = N // m + 1

    Ts_flat = np.ravel(np.asarray(d["T_surface_history"], dtype=np.float64))
    M = Ts_flat.shape[0]
    sample_indices = np.clip(np.arange(n_out) * m * M // N, 0, M - 1)
    T_samples = Ts_flat[sample_indices]

    D_s = float(d["D0"]) * np.exp(-float(d["Q_kJ_per_mol"]) * 1000.0 / (RGAS * T_samples))
    flux = D_s * (out["C_hist"][:, 0] - out["C_hist"][:, 1]) / out["dx"]

    return OutputSchema(
        C_final=out["C_final"],
        C_hist=out["C_hist"],
        x_mm=np.linspace(0.0, d["x_half_mm"], n),
        surface_flux_hist=flux,
    )


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Parameter-space central finite differences for scalar params.

    For array inputs (T_surface_history), falls back to the JAX twin.
    Documented step sizes in _REL_STEP (N4 compliance).
    """
    result = {out: {} for out in jac_outputs}
    base_out = apply(inputs)

    for inp in jac_inputs:
        if inp in _REL_STEP:
            val = float(getattr(inputs, inp))
            step = abs(val) * _REL_STEP[inp] if val != 0 else _REL_STEP[inp]
            for out in jac_outputs:
                fwd = dict(inputs.model_dump())
                fwd[inp] = val + step
                bwd = dict(inputs.model_dump())
                bwd[inp] = val - step
                out_fwd = apply(InputSchema(**fwd))
                out_bwd = apply(InputSchema(**bwd))
                result[out][inp] = (
                    np.asarray(getattr(out_fwd, out)) - np.asarray(getattr(out_bwd, out))
                ) / (2.0 * step)
        else:
            # Array input: use JAX twin jacrev
            from tesseract_core.runtime.jax_recipes import jax_jacobian

            return jax_jacobian(_forward_jit, inputs, jac_inputs, jac_outputs)

    return result


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    from tesseract_core.runtime.jax_recipes import jax_jvp

    return jax_jvp(_forward_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    from tesseract_core.runtime.jax_recipes import jax_vjp

    return jax_vjp(_forward_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    from tesseract_core.runtime.jax_recipes import jax_abstract_eval

    return jax_abstract_eval(_forward_jit, abstract_inputs)
