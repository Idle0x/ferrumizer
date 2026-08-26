"""Review-3 regression tests: log ingestion (plc_parser) + trajectory wiring.

Covers the five genuinely-open items from the master synthesis:

* P0 #1 — time normalization must NEVER guess units from the median step
  (a 10 s datalogger log was silently multiplied by 60 before).
* P0 #3 — calibration must use the ingested trajectory, not a hardcoded
  2 h / 950 C scenario (covered at the app level; here we test the
  schedule_from_trajectory + parse_plc_log contract it relies on).
* P2 #11 — schedule compression preserves ramps (RDP), not a staircase.
* P2 #12 — rows_used accounting is exact.
* P2 #13 — range validation warns on impossible values.
"""

from __future__ import annotations

import numpy as np
import pytest

from ingest.plc_parser import parse_plc_log, schedule_from_trajectory


def _csv(headers: str, rows: list[str]) -> str:
    return headers + "\n" + "\n".join(rows) + "\n"


class TestTimeNormalization:
    """P0 #1: units come from headers/clock strings, never the median."""

    def test_10s_log_with_s_header_is_not_scaled(self):
        t = np.arange(0.0, 2 * 3600.0, 10.0)
        T = 950.0 + 3.0 * np.sin(t / 300.0)
        text = _csv("time [s],temperature [C]",
                    [f"{a:.0f},{b:.3f}" for a, b in zip(t, T)])
        r = parse_plc_log("10s.csv", text=text)
        assert r.has_trajectory
        tr = r.trajectory
        assert tr is not None
        span_h = (tr["t_s"][-1] - tr["t_s"][0]) / 3600.0
        assert span_h == pytest.approx(2.0, abs=0.01)
        # and crucially: no 'interpreted as minutes' warning was emitted
        assert not any("minutes" in w for w in r.warnings)

    def test_no_unit_defaults_to_seconds_with_warning(self):
        t = np.arange(0.0, 3600.0, 10.0)
        T = np.full_like(t, 950.0)
        text = _csv("time,temperature", [f"{a:.0f},{b:.2f}" for a, b in zip(t, T)])
        r = parse_plc_log("nou.csv", text=text)
        tr = r.trajectory
        assert tr is not None
        span_h = (tr["t_s"][-1] - tr["t_s"][0]) / 3600.0
        assert span_h == pytest.approx(1.0, abs=0.01)
        assert any("ASSUMED seconds" in w for w in r.warnings)

    def test_minutes_header_scales_by_60(self):
        t = np.arange(0.0, 120.0, 1.0)  # 119-minute span, 1-min sampling
        T = 900.0 + np.sin(t)
        text = _csv("time [min],temp", [f"{a:.0f},{b:.3f}" for a, b in zip(t, T)])
        r = parse_plc_log("min.csv", text=text)
        tr = r.trajectory
        assert tr is not None
        span_min = (tr["t_s"][-1] - tr["t_s"][0]) / 60.0
        assert span_min == pytest.approx(119.0, abs=0.01)
        assert any("minutes" in w for w in r.warnings)

    def test_hours_header_scales_by_3600(self):
        t = np.arange(0.0, 3.0, 0.05)  # 3 hours, 3-min sampling
        T = 925.0 + np.sin(t * 4.0)
        text = _csv("elapsed (h),temp", [f"{a:.3f},{b:.3f}" for a, b in zip(t, T)])
        r = parse_plc_log("h.csv", text=text)
        tr = r.trajectory
        assert tr is not None
        span_h = (tr["t_s"][-1] - tr["t_s"][0]) / 3600.0
        assert span_h == pytest.approx(2.95, abs=0.01)
        assert any("hours" in w for w in r.warnings)

    def test_clock_hhmmss_parsed_to_elapsed_seconds(self):
        clock = [f"{8 + i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
                 for i in range(0, 7200, 30)]
        T = 950.0 + np.sin(np.arange(len(clock)) / 10.0)
        text = _csv("timestamp,temp", [f"{a},{b:.3f}" for a, b in zip(clock, T)])
        r = parse_plc_log("clock.csv", text=text)
        tr = r.trajectory
        assert tr is not None
        span_h = (tr["t_s"][-1] - tr["t_s"][0]) / 3600.0
        assert span_h == pytest.approx(1.99, abs=0.01)
        assert any("HH:MM:SS" in w for w in r.warnings)


class TestScheduleCompression:
    """P2 #11: RDP preserves ramps; no staircase of flat soaks."""

    def test_heatup_ramp_stays_a_ramp(self):
        # 40-minute heatup 25 -> 925 C, then a 2 h soak at 925 C
        t = np.concatenate([np.linspace(0, 2400, 241), np.linspace(2400, 9600, 721)])
        T = np.concatenate([np.linspace(25, 925, 241), np.full(721, 925.0)])
        sched = schedule_from_trajectory(t.tolist(), T.tolist())
        assert len(sched["schedule_times"]) <= 4, "ramp must not become a staircase"
        # first two knots should span the ramp (25 -> ~925)
        assert sched["schedule_temps_C"][0] == pytest.approx(25.0, abs=8.0)
        assert sched["schedule_temps_C"][-1] == pytest.approx(925.0, abs=1.0)

    def test_soak_plateau_is_flat(self):
        t = np.linspace(0, 7200, 721)
        T = np.full(721, 930.0) + 2.0 * np.sin(np.arange(721) / 20.0)
        sched = schedule_from_trajectory(t.tolist(), T.tolist())
        # a flat soak with noise should compress to ~2 knots
        assert len(sched["schedule_times"]) <= 4
        assert all(abs(x - 930.0) < 6.0 for x in sched["schedule_temps_C"])


class TestRowsAndValidation:
    """P2 #12 rows_used exact; P2 #13 range validation warns."""

    def test_rows_used_traverse_only_is_exact(self):
        text = _csv("depth_mm,hardness_hv",
                    ["0.0,650", "0.5,620", "1.0,580", "1.5,540", "2.0,500"])
        r = parse_plc_log("trav.csv", text=text)
        assert r.rows_total == 6
        assert r.rows_used == 5  # the five data rows, not rows_total - start

    def test_rows_used_with_malformed_rows(self):
        text = _csv("time [s],temp",
                    ["0,950", "10,951", "junk", "20,949", "30,952", "banner,row"])
        r = parse_plc_log("junk.csv", text=text)
        assert r.rows_used == 4
        assert any("malformed" in w for w in r.warnings)

    def test_negative_depth_and_bad_hardness_warn(self):
        text = _csv("depth_mm,hardness_hv",
                    ["-0.5,2500", "0.0,620", "1.0,40"])
        r = parse_plc_log("bad.csv", text=text)
        assert any("Negative depths" in w for w in r.warnings)
        assert any("Hardness outside" in w for w in r.warnings)

    def test_impossible_temperature_warns(self):
        t = np.arange(0.0, 600.0, 10.0)
        T = np.full_like(t, 5000.0)
        text = _csv("time [s],temp", [f"{a:.0f},{b:.1f}" for a, b in zip(t, T)])
        r = parse_plc_log("hot.csv", text=text)
        assert any("plausible furnace range" in w for w in r.warnings)


class TestTrajectoryToScenarioContract:
    """P0 #3: the calibration wiring contract — schedule knots feed Scenario."""

    def test_schedule_knots_are_scenario_compatible(self):
        # 4 h cycle at 925 C with a heatup ramp: the exact Maria scenario
        t = np.concatenate([np.linspace(0, 2400, 241), np.linspace(2400, 14400, 1201)])
        T = np.concatenate([np.linspace(25, 925, 241), np.full(1201, 925.0)])
        sched = schedule_from_trajectory(t.tolist(), T.tolist())
        # Scenario requires t_total == last knot, times monotonic, same length
        assert sched["schedule_times"][-1] == pytest.approx(14400.0, abs=10.0)
        assert len(sched["schedule_times"]) == len(sched["schedule_temps_C"])
        assert np.all(np.diff(sched["schedule_times"]) > 0)
        # the furnace sees a real ramp: interpolate at t=1200s (mid-heatup).
        # furnace_T takes knots shaped (2, K) — row 0 time(s), row 1 °C —
        # and returns KELVIN.
        from ferrumizer_physics.thermal import furnace_T

        knots = np.array([sched["schedule_times"], sched["schedule_temps_C"]])
        T_mid = float(furnace_T(knots, 1200.0)) - 273.15
        assert 400.0 < T_mid < 500.0  # ~halfway up the ramp, not a step
