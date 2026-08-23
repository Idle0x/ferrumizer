"""ferrumizer_physics — shared pure-physics library for the Ferrumizer
differentiable heat-treatment engine.

This package is the single source of truth for all physical models. It is
bundled into each stage Tesseract via ``build_config.package_data`` and is
also imported directly by the app layer (calibration, design, verification).

Modules
-------
constants   physical constants (R, Stefan-Boltzmann)
alloys      alloy preset loader (8620 / 9310 / 5120)
thermal     1-D heat conduction with radiative Robin BC (JAX)
carbon      1-D carbon diffusion with Arrhenius D(T) (JAX twin + NumPy FD)
hardening   JMAK/Scheil, Koistinen-Marburger, hardness mixing, ECD (JAX)
"""

import jax

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.alloys import list_alloys, load_alloy, ms_temperature
from ferrumizer_physics.carbon import (
    CarburizeConfig,
    D_of_T,
    D_of_T_np,
    erfc_reference,
    run_carburize,
    run_carburize_numpy,
    stability_check_carbon,
)
from ferrumizer_physics.constants import RGAS, SIGMA
from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    jmak_scheil_fraction,
    km_fraction,
    ms_andrews,
    run_hardening,
    smoothstep,
)
from ferrumizer_physics.thermal import (
    ThermalConfig,
    cyl_grid,
    face_temperature,
    furnace_T,
    grid,
    lumped_surface_T,
    run_thermal,
    slab_grid,
    stability_dt,
)

__version__ = "0.1.0"

__all__ = [
    "RGAS",
    "SIGMA",
    "list_alloys",
    "load_alloy",
    "ms_temperature",
    "ThermalConfig",
    "slab_grid",
    "cyl_grid",
    "grid",
    "stability_dt",
    "furnace_T",
    "face_temperature",
    "run_thermal",
    "lumped_surface_T",
    "CarburizeConfig",
    "D_of_T",
    "D_of_T_np",
    "stability_check_carbon",
    "run_carburize",
    "run_carburize_numpy",
    "erfc_reference",
    "ms_andrews",
    "km_fraction",
    "smoothstep",
    "hardness_profile",
    "ecd_from_hardness",
    "jmak_scheil_fraction",
    "run_hardening",
    "__version__",
]
