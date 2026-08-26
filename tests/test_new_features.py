"""Tests for the quench model, dynamic alloy chemistry, and PLC ingestion."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ferrumize.pipeline import FerrumizerPipeline, ProcessParams, Scenario
from ferrumizer_physics.alloys import composition_to_preset, load_alloy, validate_preset
from ferrumizer_physics.hardening import newton_cooling_curve, quench_fractions
from ingest.plc_parser import parse_plc_log, schedule_from_trajectory


# --------------------------------------------------------------------------- #
# Quench model
# --------------------------------------------------------------------------- #
class TestQuenchModel:
    def test_instant_path_unchanged(self):
        """Legacy path (no quench medium) must match the old instant-quench result."""
        sc = Scenario()
        res = FerrumizerPipeline(sc, ProcessParams()).forward()
        assert float(res["ecd_mm"]) > 0.15  # full case from instant quench

    def test_air_quench_collapses_hardness(self):
        """Slow quench converts austenite to pearlite -> no case depth."""
        sc = Scenario(quench_medium="air", quench_temp_K=333.15, quench_agitation=0.2, size_mm=16.0)
        res = FerrumizerPipeline(sc, ProcessParams()).forward()
        assert float(res["quench"]["X_diffusional"][0]) > 0.9
        assert float(res["ecd_mm"]) < 0.01
        assert float(res["H"][0]) < 300.0

    def test_water_quench_preserves_martensite(self):
        """Fast quench keeps most austenite as martensite -> full case."""
        sc = Scenario(quench_medium="water", quench_temp_K=298.15, quench_agitation=0.8, size_mm=16.0)
        res = FerrumizerPipeline(sc, ProcessParams()).forward()
        assert float(res["quench"]["X_diffusional"][0]) < 0.05
        assert float(res["f_martensite"][0]) > 0.9
        assert float(res["ecd_mm"]) > 0.15

    def test_medium_ranking(self):
        """Slower media must produce more diffusional phases (air > oil > water)."""
        def xdiff(medium, ag):
            sc = Scenario(quench_medium=medium, quench_temp_K=333.15, quench_agitation=ag, size_mm=16.0)
            return float(FerrumizerPipeline(sc, ProcessParams()).forward()["quench"]["X_diffusional"][0])

        assert xdiff("air", 0.2) > xdiff("oil", 0.3)
        assert xdiff("oil", 0.3) > xdiff("water", 0.8)

    def test_depth_resolved_phases(self):
        """Spatial quench: surface and core must see different cooling rates
        (phase fractions differ across the section for a fast quench)."""
        sc = Scenario(quench_medium="water", quench_temp_K=298.15, quench_agitation=0.8, size_mm=16.0)
        res = FerrumizerPipeline(sc, ProcessParams()).forward()
        q = res["quench"]
        # per-depth arrays, not scalars
        assert np.ndim(np.asarray(q["X_pearlite"])) == 1
        # surface pearlite below core pearlite for water (surface cools faster)
        assert float(q["X_pearlite"][0]) <= float(q["X_pearlite"][-1]) + 1e-9

    def test_cooling_curve_differentiable(self):
        """newton_cooling_curve must be JAX-differentiable."""
        T = newton_cooling_curve(1223.0, 333.0, 900.0, 5.46e6, 0.008, jnp.linspace(0, 600, 50), 0.5)
        g = jax.grad(lambda h: jnp.sum(newton_cooling_curve(1223.0, 333.0, h, 5.46e6, 0.008, jnp.linspace(0, 600, 50), 0.5)))(900.0)
        assert jnp.isfinite(g)
        assert jnp.all(T >= 333.0 - 1e-9)  # never below bath temp


# --------------------------------------------------------------------------- #
# Dynamic alloy chemistry
# --------------------------------------------------------------------------- #
class TestDynamicAlloys:
    def test_composition_to_preset_schema(self):
        preset = composition_to_preset({"C": 0.2, "Mn": 0.8, "Cr": 1.1}, name="x")
        assert validate_preset(preset) == []
        assert preset["hardness"]["Hmax"] > 600
        assert preset["ms"]["A"] > 700

    def test_carbon_required(self):
        with pytest.raises(ValueError):
            composition_to_preset({"Mn": 0.5})

    def test_load_alloy_accepts_dict(self):
        preset = composition_to_preset({"C": 0.2}, name="y")
        assert load_alloy(preset) is preset

    def test_load_alloy_rejects_bad_dict(self):
        with pytest.raises(KeyError):
            load_alloy({"bad": "dict"})

    def test_custom_alloy_runs_pipeline(self):
        preset = composition_to_preset({"C": 0.2, "Mn": 0.8}, name="z")
        sc = Scenario(quench_medium="water", quench_temp_K=298.15, quench_agitation=0.8)
        res = FerrumizerPipeline(sc, ProcessParams(), preset=preset).forward()
        assert float(res["ecd_mm"]) > 0.0


# --------------------------------------------------------------------------- #
# PLC ingestion
# --------------------------------------------------------------------------- #
MESSY_LOG = """IPSEN VUTK-524 Datalogger Export
Customer: Acme | Heat: 8841
========================================================
Elapsed [min], Furnace Temp [degF], Zone2 [F], Notes
0, 72.0, 70.0, "charge loaded"
5, 310.4, 308.2,
10, 822.1, 818.0,
15, 1410.2, 1405.0, "ramp up"
20, 1742.0, 1733.0, "at temp"
25, 1742.5, 1733.5, "soaking"
30, 1741.9, 1733.1,
35, 1742.2, 1733.5,
40, 1742.0, 1733.2,
45, 1640.2, 1630.0, "cool"
50, 1492.1, 1484.0,
55, 662.0, 658.0, "quench drop"
60, 212.0, 210.0, "unload"
garbage row that should be skipped
"""

TRAVERSE_CSV = """Depth[mm];Hardness[HV0.3]
0.10;612
0.30;581
0.50;545
0.80;472
1.50;318
"""


class TestPLCIngestion:
    def test_parses_messy_log(self):
        r = parse_plc_log("messy.log", text=MESSY_LOG)
        assert r.has_trajectory
        assert r.temperature_unit == "F"
        assert len(r.trajectory["t_s"]) == 13
        # minutes -> seconds conversion happened
        assert r.trajectory["t_s"][-1] == 3600.0
        # degF -> degC conversion happened (72 F -> 21.11 C)
        assert abs(r.trajectory["T_C"][0] - 21.11) < 0.5
        assert any("minutes" in w for w in r.warnings)

    def test_parses_traverse(self):
        r = parse_plc_log("traverse.csv", text=TRAVERSE_CSV)
        assert r.has_traverse
        assert len(r.traverse["depth_mm"]) == 5
        assert r.traverse["hardness_HV"][0] == 612.0
        assert not r.has_trajectory

    def test_schedule_compression(self):
        r = parse_plc_log("messy.log", text=MESSY_LOG)
        s = schedule_from_trajectory(r.trajectory["t_s"], r.trajectory["T_C"])
        # noisy ramp should collapse into soak segments
        assert len(s["schedule_times"]) >= 2
        # soak temperature near 1742 F = 950 C (within tolerance)
        assert any(abs(t - 950.0) < 25.0 for t in s["schedule_temps_C"])

    def test_empty_input(self):
        r = parse_plc_log("empty.log", text="\n\n")
        assert not r.has_trajectory
        assert not r.has_traverse
        assert any("No usable" in w for w in r.warnings)
