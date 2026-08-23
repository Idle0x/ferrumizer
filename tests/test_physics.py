from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.alloys import list_alloys, load_alloy, ms_temperature
from ferrumizer_physics.carbon import (
    CarburizeConfig,
    D_of_T_np,
    erfc_reference,
    run_carburize,
    run_carburize_numpy,
    stability_check_carbon,
)
from ferrumizer_physics.hardening import (
    ecd_from_hardness,
    hardness_profile,
    jmak_scheil_fraction,
    km_fraction,
    ms_andrews,
    smoothstep,
)
from ferrumizer_physics.thermal import (
    ThermalConfig,
    furnace_T,
    grid,
    run_thermal,
    slab_grid,
    stability_dt,
)


def test_alloys_and_ms():
    assert list_alloys() == ["5120", "8620", "9310"]
    p = load_alloy("aisi-8620")
    assert p["alloy"] == "aisi_8620"
    assert ms_temperature(p, 1.0) == pytest.approx(593.0)
    with pytest.raises(KeyError):
        load_alloy("9999")


def test_thermal_limits_and_schedule():
    with pytest.raises(ValueError):
        stability_dt(0.0, 1e-3)
    with pytest.raises(ValueError):
        stability_dt(1.0, 0.0)
    with pytest.raises(ValueError):
        stability_dt(1.0, 1e-3, safety=0.0)
    with pytest.raises(ValueError):
        stability_dt(1.0, 1e-3, safety=1.1)
    assert grid("slab", 16.0, 11)[0].shape == (11,)
    assert grid("cylinder", 16.0, 11)[0][0] == 0
    with pytest.raises(ValueError):
        grid("triangle", 16.0, 11)
    knots = jnp.array([[0.0, 10.0], [900.0, 1000.0]])
    assert float(furnace_T(knots, 20.0)) == pytest.approx(1273.15)


def test_thermal_forward_slab_and_cylinder():
    alpha = 42.0 / (7800.0 * 700.0)
    _, dx = slab_grid(16.0, 11)
    dt = stability_dt(alpha, dx)
    knots = jnp.array([[0.0, 60.0], [950.0, 950.0]])
    for geometry in ("slab", "cylinder"):
        _, geometry_dx = grid(geometry, 16.0, 11)
        geometry_dt = stability_dt(alpha, geometry_dx) / (4.0 if geometry == "cylinder" else 1.0)
        cfg = ThermalConfig(
            geometry=geometry,
            size_mm=16.0,
            n=11,
            dt=geometry_dt,
            t_total=60.0,
            alpha=alpha,
            h=20.0,
            eps=0.8,
            k=42.0,
            T_init_K=298.15,
            sample_every=10,
        )
        out = run_thermal(knots, cfg)
        assert out["T_final"].shape == (11,)
        assert np.isfinite(np.asarray(out["Ts"])).all()
    with pytest.raises(ValueError):
        ThermalConfig(
            geometry="slab",
            size_mm=16.0,
            n=11,
            dt=1.0,
            t_total=1.0,
            alpha=alpha,
            h=20.0,
            eps=0.8,
            k=42.0,
            T_init_K=298.15,
        )
        run_thermal(
            knots,
            ThermalConfig(
                geometry="slab",
                size_mm=16.0,
                n=11,
                dt=1.0,
                t_total=1.0,
                alpha=alpha,
                h=20.0,
                eps=0.8,
                k=42.0,
                T_init_K=298.15,
            ),
        )


def test_carbon_reference_and_parity():
    T = np.full(16, 1223.15)
    cfg = CarburizeConfig(n=21, dt=2.0, t_total=3600.0, sample_every=300)
    jx = run_carburize(T, cfg, 0.2, 2.2e-5, 137000.0, 1.0, 1e-4, 8.0)
    np_out = run_carburize_numpy(T, cfg, 0.2, 2.2e-5, 137000.0, 1.0, 1e-4, 8.0)
    assert np.max(np.abs(np.asarray(jx["C_final"]) - np_out["C_final"])) < 1e-12
    assert erfc_reference(np.array([0.0]), 3600.0, D_of_T_np(2.2e-5, 137000.0, 1223.15), 1.0, 0.2)[
        0
    ] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stability_check_carbon(1e-5, 1e-4, 1.0, "mass_transfer", 1e-3)
    with pytest.raises(ValueError):
        stability_check_carbon(1e-5, 1e-4, 1.0, "dirichlet", 0.0, safety=0.0)
    assert cfg.replace(dt=1.0).dt == 1.0


def test_hardening_functions_and_ecd_edges():
    p = load_alloy("8620")
    C = jnp.array([1.0, 0.8, 0.2])
    Ms = ms_andrews(C, p["ms"]["A"], p["ms"]["b_carbon"])
    f = km_fraction(Ms, 298.15, p["km_alpha"])
    H = hardness_profile(C, p, f)
    x = jnp.array([0.0, 1.0, 2.0])
    assert float(smoothstep(0.5)) == pytest.approx(0.5)
    assert float(f[0]) < float(f[-1])
    assert float(ecd_from_hardness(H, x, 550.0)) >= 0.0
    assert float(ecd_from_hardness(jnp.array([200.0, 200.0]), jnp.array([0.0, 1.0]))) == 0.0
    assert float(
        ecd_from_hardness(jnp.array([650.0, 650.0]), jnp.array([0.0, 1.0]))
    ) == pytest.approx(1.0)
    assert (
        0.0
        <= float(
            jmak_scheil_fraction(jnp.linspace(600.0, 1000.0, 20), 1.0, 2.0, 1e-3, 823.15, 80.0)
        )
        <= 1.0
    )
    full = __import__("ferrumizer_physics.hardening", fromlist=["run_hardening"])
    result = full.run_hardening(C, x, 298.15, p, T_history=jnp.linspace(600.0, 1000.0, 20))
    assert "X_diffusional" in result
