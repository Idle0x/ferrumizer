"""Tests for the cycle-prediction review fixes: h_m/Robin, SBC integrity,
V6 gate, profile likelihoods, PPC, sigma inference, hardenability."""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from ferrumizer_physics.alloys import (
    ideal_critical_diameter_mm,
    load_alloy,
    through_hardening_verdict,
)


# --------------------------------------------------------------------------- #
# Hardenability (Grossmann DI)
# --------------------------------------------------------------------------- #
class TestHardenability:
    def test_di_positive_and_reasonable(self):
        """8620 should have a DI on the order of tens of mm."""
        preset = load_alloy("8620")
        di = ideal_critical_diameter_mm(preset)
        assert 10.0 < di < 80.0

    def test_alloying_raises_di(self):
        """More alloying (Mn, Cr, Mo) must raise hardenability."""
        di_8620 = ideal_critical_diameter_mm(load_alloy("8620"))
        di_9310 = ideal_critical_diameter_mm(load_alloy("9310"))
        assert di_9310 > di_8620  # 9310 has more Ni/Cr -> more hardenable

    def test_verdict_scale(self):
        """Small section -> through-hardening; large section -> soft core."""
        preset = load_alloy("8620")
        small = through_hardening_verdict(preset, section_mm=6.0)
        large = through_hardening_verdict(preset, section_mm=200.0)
        assert "through" in small["verdict"].lower()
        assert "soft core" in large["verdict"].lower()
        assert small["di_to_section_ratio"] > large["di_to_section_ratio"]


# --------------------------------------------------------------------------- #
# Calibration boundary condition (h_m must be exercised)
# --------------------------------------------------------------------------- #
class TestCalibrationBC:
    def test_run_calibration_rejects_dirichlet(self):
        """Calibration samples h_m; Dirichlet mode makes it dead — must fail."""
        from calibration.calibrate import run_calibration
        from ferrumize.pipeline import Scenario

        sc = Scenario(carbon_mode="dirichlet")
        depths = np.array([0.0, 0.5, 1.0])
        H = np.array([600.0, 450.0, 300.0])
        with pytest.raises(ValueError, match="mass_transfer"):
            run_calibration(depths, H, sc, num_warmup=2, num_samples=2, num_chains=1)

    def test_run_calibration_initializes_and_samples(self):
        """Regression: the app's Cycle Predictor path must not crash.

        Two production bugs were caught in the final sweep:
          (1) `_predict_hardness` returned only H while the model unpacked
              `(H, ecd)` — the likelihood crashed instantly.
          (2) NUTS used numpyro's default init_to_uniform, which failed with
              "Cannot find valid initial parameters" on the R2 physics.
        This test runs a tiny single-schedule calibration end-to-end (prior
        init) and asserts we get samples with finite statistics.
        """
        from calibration.calibrate import run_calibration
        from ferrumize.pipeline import Scenario
        from verification.v6_recovery import generate_planted_data

        data = generate_planted_data(noise_sigma=12.0, seed=0)
        sc = Scenario(carbon_mode="mass_transfer", carbon_n=21, carbon_dt=8.0)
        mcmc, summary = run_calibration(
            data["depths_mm"],
            data["schedules"][0]["H_obs"],
            sc,
            sigma_hv=12.0,
            infer_sigma=False,
            num_warmup=20,
            num_samples=20,
            num_chains=1,
            seed=0,
        )
        samples = mcmc.get_samples(group_by_chain=False)
        assert "log_D0" in samples and "Q_kJ" in samples
        for k in ("log_D0", "Q_kJ", "C_pot", "eps"):
            v = np.asarray(samples[k])
            assert np.all(np.isfinite(v)), f"{k} has non-finite samples"
        assert summary["gates_ok"] is not None

    def test_config_two_schedule_parses(self):
        """The generated calibration YAML must describe both schedules."""
        from ferrumize.config import (
            load_config,
            scenario2_from_config,
            scenario_from_config,
            validate_config,
        )

        cfg = load_config("data/synthetic/calibration_data.yaml")
        assert validate_config(cfg) == []
        sc = scenario_from_config(cfg)
        sc2 = scenario2_from_config(cfg)
        assert sc2 is not None
        assert sc.schedule_temps_C == (900.0, 900.0)
        assert sc2.schedule_temps_C == (1000.0, 1000.0)
        assert sc.carbon_mode == "mass_transfer"
        assert cfg["observations"]["traverse_csv"] == "traverse_low.csv"
        assert cfg["observations"]["traverse_csv2"] == "traverse_high.csv"


# --------------------------------------------------------------------------- #
# V6: gate threshold and configurable noise
# --------------------------------------------------------------------------- #
class TestV6Gate:
    def test_gate_constant(self):
        from verification.v6_recovery import GATE_REL_ERR

        assert GATE_REL_ERR >= 5e-3  # 1e-4 was numerically naive

    def test_planted_data_has_h_m(self):
        """h_m must be recovered (mass_transfer), not dead."""
        from verification.v6_recovery import PLANTED, generate_planted_data

        assert "h_m" in PLANTED
        data = generate_planted_data()
        assert set(data["planted"]) == {"log_D0", "Q_kJ", "C_pot", "h_m", "eps"}


# --------------------------------------------------------------------------- #
# V7: SBC must not cheat
# --------------------------------------------------------------------------- #
class TestV7Integrity:
    def test_n_sim_is_real(self):
        from verification.v7_sbc_tarp import N_SIM

        assert N_SIM >= 200  # the former 4 was statistically void

    def test_init_is_from_prior(self):
        """Init must be from the prior, never the true planted values."""
        import inspect

        from verification.v7_sbc_tarp import _run_inference

        src = inspect.getsource(_run_inference)
        assert "init_to_sample" in src
        assert "init_to_value" not in src

    def test_no_silent_nan_mask(self):
        """NaN must become a hard penalty (1e6), never a flat 230 HV line."""
        import inspect

        from verification.v7_sbc_tarp import _predict_H

        src = inspect.getsource(_predict_H)
        assert "nan_to_num" not in src
        assert "1e6" in src


# --------------------------------------------------------------------------- #
# Profile likelihoods
# --------------------------------------------------------------------------- #
class TestProfileLikelihood:
    def test_grid_shapes(self):
        from ferrumize.pipeline import ProcessParams, Scenario
        from identifiability.analyze import profile_likelihood_grid

        import dataclasses

        # light grid (as the identifiability CLI uses): the profile surface
        # is a diagnostic of D0-Q degeneracy shape, not a production calc
        sc = dataclasses.replace(
            Scenario(carbon_mode="mass_transfer"),
            carbon_n=21,
            carbon_dt=8.0,
            carbon_mode="dirichlet",
        )
        params = ProcessParams()
        pv = np.array([np.log(params.D0), params.Q_kJ, params.C_pot, params.h_m, params.eps])
        depths = np.linspace(0.0, 2.0, 9)
        H = 600.0 - 200.0 * depths
        pl = profile_likelihood_grid(
            pv,
            depths,
            H,
            sc,
            log_D0_range=(-11.5, -10.0, 5),
            Q_range=(110.0, 165.0, 5),
            n_nuisance_iters=2,
        )
        assert pl["neg_log_lik"].shape == (5, 5)
        assert np.isfinite(pl["neg_log_lik"]).all()

    def test_two_schedule_lowest_at_truth(self):
        """The two-schedule profile likelihood should be minimized near the
        planted parameter point (not on the D0-Q ridge).

        Tolerance is intentionally grid-coarse: the surface uses a light
        scenario (fast diagnostic) with log_D0 cells of ~0.33, so 'near' is
        within ~2 cells. The TIGHT recovery standard is the V6 gate (L-BFGS
        on the full two-schedule likelihood, planted params to ~1e-3), not
        the diagnostic grid.
        """
        import dataclasses

        from ferrumize.pipeline import ProcessParams, Scenario
        from identifiability.analyze import profile_likelihood_grid
        from verification.v6_recovery import PLANTED, _kwargs_for, _predict_H

        sc = dataclasses.replace(
            Scenario(carbon_mode="mass_transfer"),
            carbon_n=21,
            carbon_dt=8.0,
            carbon_mode="dirichlet",
        )
        kwargs = _kwargs_for()
        p = [
            jnp_float(PLANTED["log_D0"]),
            jnp_float(PLANTED["Q_kJ"]),
            jnp_float(PLANTED["C_pot"]),
            jnp_float(np.log(PLANTED["h_m"])),
            jnp_float(PLANTED["eps"]),
        ]
        from verification.v6_recovery import OBS_DEPTHS_MM

        depths = OBS_DEPTHS_MM.copy()
        H = np.asarray(_predict_H(p, (1000.0, 1000.0), kwargs))

        pv = np.array(
            [PLANTED["log_D0"], PLANTED["Q_kJ"], PLANTED["C_pot"], PLANTED["h_m"], PLANTED["eps"]]
        )
        pl = profile_likelihood_grid(
            pv,
            depths,
            H,
            sc,
            log_D0_range=(float(PLANTED["log_D0"]) - 1.0, float(PLANTED["log_D0"]) + 1.0, 7),
            Q_range=(float(PLANTED["Q_kJ"]) - 15.0, float(PLANTED["Q_kJ"]) + 15.0, 7),
            n_nuisance_iters=3,
        )
        # best grid point should be near planted (within ~2 grid cells;
        # tight recovery is the V6 gate's job)
        assert abs(pl["best_log_D0"] - PLANTED["log_D0"]) < 0.7
        assert abs(pl["best_Q"] - PLANTED["Q_kJ"]) < 12.0


def jnp_float(x):
    import jax.numpy as jnp

    return jnp.float64(x)
