"""Ingestion parser for messy furnace PLC / datalogger exports.

Real furnace logs are not clean CSV. They arrive with company banners,
multi-line headers, mixed units (deg C / deg F), timestamps in several
formats, quoted fields, separator chaos, and occasional sensor dropout
rows. This module is deliberately defensive: it sniffs the structure
instead of assuming it, and reports what it had to guess.

Two outputs are produced from one file when possible:

* ``trajectory`` — time (s) vs temperature (deg C) history from the
  furnace's own datalogger; used for reconstructing a cycle, comparing
  against a designed schedule, or validating a virtual-furnace run.
* ``traverse``  — depth (mm) vs hardness (HV) from a metallography cut;
  the exact input the calibration stage consumes.

Column detection uses a synonym table so the parser works on exports from
different PLCs without configuration. Unknown columns are ignored with a
warning, never silently dropped data.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Column synonym tables
# --------------------------------------------------------------------------- #
TIME_SYNONYMS = {
    "time", "t", "time_s", "timestamp", "time_sec", "seconds", "sec", "elapsed",
    "elapsed_s", "s", "time_since_start", "dt",
}
TEMP_SYNONYMS = {
    "temp", "temperature", "temp_c", "temperature_c", "t_c", "tc", "temp_degc",
    "temperature_degc", "degc", "temp_f", "temperature_f", "degf", "temp_degf",
    "furnace_temp", "furnace_temperature", "zone_temp", "soak_temp", "atmos_temp",
}
DEPTH_SYNONYMS = {
    "depth", "depth_mm", "x", "x_mm", "dist", "distance", "distance_mm",
    "case_depth", "depth_from_surface",
}
HARDNESS_SYNONYMS = {
    "hardness", "hardness_hv", "hv", "hv0.3", "hv0_3", "h", "h_hv",
    "microhardness", "vickers", "vickers_hv",
}

_UNIT_RE = re.compile(r"(?i)(deg\s*c|degc|celsius|°c|\bc\b)")
_FAHRENHEIT_RE = re.compile(r"(?i)(deg\s*f|degf|fahrenheit|°f|\bf\b)")

_TIME_RE = re.compile(r"(?i)(time|sec|elapsed|timestamp)")
_TEMP_RE = re.compile(r"(?i)(temp|deg|°|furnace|zone|atmos)")
_DEPTH_RE = re.compile(r"(?i)(depth|dist|x_mm|d_mm)")
_HARD_RE = re.compile(r"(?i)(hardness|hv|vickers)")


def _detect_time_unit(header_cell: str) -> tuple[str, str]:
    """Detect the time unit from the column header text.

    Returns ``(unit, source)`` where unit is one of ``"s"``, ``"min"``,
    ``"h"``, ``"clock"`` (HH:MM:SS), or ``"unknown"``.

    This REPLACES the old median-difference heuristic, which silently
    multiplied 5-30 s datalogger logs by 60 (a 10 s sampling rate became a
    300-hour cycle). Furnace dataloggers (Eurotherm, Super Systems, ...)
    commonly log at 5-30 s for a 20-hour cycle — the median step is *no
    information* about the unit. The header is.
    """
    h = header_cell.strip().lower()
    h = h.replace("°", "deg")
    # clock-time formats: 'hh:mm:ss' / 'hh:mm' mentioned in the header
    if re.search(r"(?i)hh\s*:\s*mm|clock|timestamp|hh:mm", h):
        return "clock", "header"
    # explicit unit tokens, longest-first so 'min' isn't matched by 'm'
    for unit, pat in (
        ("h", r"(?i)\b(hrs?|hours?|h)\b|\bh\b|\[h\]|\(h\)|_h\b|\bh$"),
        ("min", r"(?i)\b(mins?|minutes?)\b|\[min\]|\(min\)|\bmin\b|_min"),
        ("s", r"(?i)\b(secs?|seconds?)\b|\[s\]|\(s\)|\bsec\b|_s\b|\bs$"),
    ):
        if re.search(pat, h):
            return unit, "header"
    # bare 'm' is ambiguous — treat as minutes ONLY if 'min' spelled out;
    # a bare 's' alone is seconds (SI convention)
    if re.search(r"(?i)\bm\b", h) and re.search(r"(?i)min", h):
        return "min", "header"
    return "unknown", "none"


def _parse_clock_cell(cell: str) -> float | None:
    """Parse 'HH:MM:SS' or 'HH:MM' elapsed/clock time into seconds."""
    m = re.match(r"^\s*(\d{1,3}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*$", cell)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    s = float(m.group(3)) if m.group(3) else 0.0
    if mi > 59 or s >= 60.0:
        return None
    return h * 3600.0 + mi * 60.0 + s


def _classify_column(name: str) -> str | None:
    """Map a column header to a canonical role (time/temp/depth/hardness)."""
    norm = re.sub(r"[^a-z0-9_]", "", name.lower())
    if norm in TIME_SYNONYMS:
        return "time"
    if norm in TEMP_SYNONYMS:
        return "temp"
    if norm in DEPTH_SYNONYMS:
        return "depth"
    if norm in HARDNESS_SYNONYMS:
        return "hardness"
    # fallback regex (handles 'Time [s]', 'T (degC)', 'HV0.3' etc.)
    if _TIME_RE.search(name) and not _TEMP_RE.search(name):
        return "time"
    if _TEMP_RE.search(name):
        return "temp"
    if _DEPTH_RE.search(name):
        return "depth"
    if _HARD_RE.search(name):
        return "hardness"
    return None


@dataclass
class IngestReport:
    """What the parser found, what it guessed, and the extracted data."""

    source: str
    rows_total: int = 0
    rows_used: int = 0
    trajectory: dict | None = None  # {"t_s": [...], "T_C": [...]}
    traverse: dict | None = None    # {"depth_mm": [...], "hardness_HV": [...]}
    temperature_unit: str = "C"
    warnings: list[str] = field(default_factory=list)

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory is not None and len(self.trajectory["t_s"]) > 1

    @property
    def has_traverse(self) -> bool:
        return self.traverse is not None and len(self.traverse["depth_mm"]) > 1

    def as_dict(self) -> dict:
        out = {
            "source": self.source,
            "rows_total": self.rows_total,
            "rows_used": self.rows_used,
            "temperature_unit": self.temperature_unit,
            "warnings": self.warnings,
        }
        if self.has_trajectory:
            out["trajectory"] = self.trajectory
        if self.has_traverse:
            out["traverse"] = self.traverse
        return out


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # fall back by counting
        counts = {d: sample.count(d) for d in ",;\t|"}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","


def _find_header_row(rows: list[list[str]]) -> int | None:
    """Locate the row whose cells look like column headers.

    Returns None if no recognizable header is found (data assumed headerless).
    """
    for i, row in enumerate(rows[:30]):
        if not row:
            continue
        n_hits = sum(1 for cell in row if _classify_column(cell) is not None)
        if n_hits >= 2:
            return i
        # numeric-ish cells that mention time/temp in the same row also count
        n_time_temp = sum(1 for cell in row if _TIME_RE.search(cell) or _TEMP_RE.search(cell))
        if n_time_temp >= 2 and n_hits >= 1:
            return i
    return None


def _f_to_c(v: float) -> float:
    return (v - 32.0) * 5.0 / 9.0


def _to_float(cell: str) -> float | None:
    cell = cell.strip().replace(",", "").replace('"', "").strip()
    if not cell:
        return None
    # strip stray units like '950 C' or '930°C'
    m = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", cell)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_plc_log(source: str | Path, text: str | None = None) -> IngestReport:
    """Parse a PLC/datalogger export into a normalized report.

    Parameters
    ----------
    source : path or name of the file (used in the report + error msgs)
    text   : optional file contents; if None, ``source`` is read from disk.

    Returns
    -------
    IngestReport with trajectory and/or traverse + warnings.
    """
    src = str(source)
    if text is None:
        text = Path(source).read_text(encoding="utf-8", errors="replace")

    report = IngestReport(source=src)
    delim = _sniff_delimiter(text)
    raw_rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    for row in reader:
        # drop fully-empty rows
        if any(cell.strip() for cell in row):
            raw_rows.append(row)

    report.rows_total = len(raw_rows)
    header_idx = _find_header_row(raw_rows)
    start = 0
    colmap: dict[str, int] = {}

    if header_idx is not None:
        header = raw_rows[header_idx]
        for j, cell in enumerate(header):
            role = _classify_column(cell)
            if role is not None:
                colmap[role] = j
        start = header_idx + 1
        if "temp" in colmap:
            u = header[colmap["temp"]]
            report.temperature_unit = "F" if _FAHRENHEIT_RE.search(u) else "C"
    else:
        # headerless: assume the first two columns are time, temperature
        report.warnings.append("No recognizable header row found; assumed columns = time,temperature")
        colmap = {"time": 0, "temp": 1}

    # Parse rows into role arrays
    arrays: dict[str, list[float]] = {"time": [], "temp": [], "depth": [], "hardness": []}
    time_col_clock = False
    if "time" in colmap and header_idx is not None:
        unit, _src = _detect_time_unit(raw_rows[header_idx][colmap["time"]])
        time_col_clock = unit == "clock"
    skipped = 0
    used_rows = 0
    for row in raw_rows[start:]:
        if len(row) <= max(colmap.values(), default=-1):
            skipped += 1
            continue
        vals: dict[str, float] = {}
        ok = False
        for role, idx in colmap.items():
            cell = row[idx]
            if role == "time" and time_col_clock:
                v = _parse_clock_cell(cell)
                if v is None:
                    continue
            else:
                v = _to_float(cell)
                if v is None:
                    # tolerate a single bad cell by skipping just that column
                    continue
            if role == "temp" and report.temperature_unit == "F":
                v = _f_to_c(v)
            vals[role] = v
            ok = True
        if not ok:
            skipped += 1
            continue
        used_rows += 1
        for role, v in vals.items():
            arrays[role].append(v)

    report.rows_used = used_rows
    if skipped:
        report.warnings.append(f"Skipped {skipped} malformed row(s) (non-numeric or short).")

    # Build outputs
    if len(arrays["time"]) >= 2 and len(arrays["temp"]) >= 2:
        t = np.asarray(arrays["time"], dtype=np.float64)
        T = np.asarray(arrays["temp"], dtype=np.float64)
        # ------------------------------------------------------------------ #
        # Time normalization — see ADR: NEVER guess units from the median
        # time step. A 5-30 s datalogger log would be silently multiplied
        # by 60 (10 s -> 300-hour cycle). The unit comes from the header,
        # from HH:MM:SS parsing, or defaults to seconds with a warning.
        # ------------------------------------------------------------------ #
        if time_col_clock:
            # HH:MM:SS clock timestamps: convert to elapsed seconds
            t0 = t[0]
            t = t - t0
            report.warnings.append(
                "Time column parsed as HH:MM:SS clock time; converted to "
                f"elapsed seconds (t0 = {t0:.0f} s)."
            )
        else:
            unit, src = "unknown", "none"
            if header_idx is not None and "time" in colmap:
                unit, src = _detect_time_unit(raw_rows[header_idx][colmap["time"]])
            if unit == "min":
                t = t * 60.0
                report.warnings.append("Time column header says minutes; converted to seconds (×60).")
            elif unit == "h":
                t = t * 3600.0
                report.warnings.append("Time column header says hours; converted to seconds (×3600).")
            elif unit == "s":
                pass  # already seconds
            else:
                # no unit in header: default to seconds, warn loudly
                report.warnings.append(
                    "Time column unit not stated in header; ASSUMED seconds. "
                    "If your log is in minutes, re-export with a unit in the "
                    "column name (e.g. 'time [min]') to avoid 60× errors."
                )
        span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if span > 0:
            dt = np.diff(t)
            med = float(np.median(dt)) if len(dt) else 0.0
            # sanity: a cycle longer than 30 days at a median step < 1 s is
            # almost certainly a unit error — flag, do NOT auto-rescale.
            if med < 1.0 and span > 30 * 86400.0:
                report.warnings.append(
                    "Suspicious time span: >30 days at sub-second sampling. "
                    "The time unit may be wrong — check your export."
                )
        order = np.argsort(t)
        report.trajectory = {
            "t_s": [float(x) for x in t[order]],
            "T_C": [float(x) for x in T[order]],
        }
        # range validation (P2 #13): impossible furnace temperatures
        T_arr = np.asarray(report.trajectory["T_C"])
        if np.any(T_arr < -50.0) or np.any(T_arr > 1400.0):
            report.warnings.append(
                "Temperature outside a plausible furnace range (-50..1400 °C) "
                "in the trajectory — check units or sensor calibration."
            )

    if len(arrays["depth"]) >= 2 and len(arrays["hardness"]) >= 2:
        d = np.asarray(arrays["depth"], dtype=np.float64)
        H = np.asarray(arrays["hardness"], dtype=np.float64)
        order = np.argsort(d)
        report.traverse = {
            "depth_mm": [float(x) for x in d[order]],
            "hardness_HV": [float(x) for x in H[order]],
        }
        # warn if depths are not monotonically spread (common export bug)
        if len(d) > 3 and np.median(np.diff(np.sort(d))) <= 0:
            report.warnings.append("Duplicate depth values found; sorted by depth.")
        # range validation (P2 #13): negative depths, impossible hardness
        if np.any(d < 0.0):
            report.warnings.append("Negative depths found in traverse — check the depth column.")
        if np.any(H < 50.0) or np.any(H > 1200.0):
            report.warnings.append(
                "Hardness outside a plausible HV range (50..1200) — check "
                "units (MPa/HRC mistaken for HV?) or bad rows."
            )

    if not report.has_trajectory and not report.has_traverse:
        report.warnings.append(
            "No usable trajectory or traverse columns found. Expected time+temperature "
            "and/or depth+hardness columns."
        )

    return report


def _rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker line simplification on (t, T) points.

    Preserves controlled heating/cooling RAMPS as diagonal segments instead
    of chopping them into a staircase of flat soaks (the old change-point
    detector fired a new 'segment' every min_hold_s during any continuous
    ramp, and encoded furnace overshoot as a deliberate setpoint). Returns
    the simplified points, endpoints always kept.
    """
    if len(points) < 3:
        return points

    def _simplify(pts: np.ndarray) -> np.ndarray:
        if len(pts) < 3:
            return pts
        start, end = pts[0], pts[-1]
        seg = end - start
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len < 1e-12:
            return np.array([start, end])
        # perpendicular distance of each interior point from the chord
        d = np.abs((end[0] - start[0]) * (start[1] - pts[:, 1])
                   - (start[0] - pts[:, 0]) * (end[1] - start[1])) / seg_len
        idx = int(np.argmax(d[1:-1])) + 1
        if d[idx] > epsilon:
            left = _simplify(pts[: idx + 1])
            right = _simplify(pts[idx:])
            return np.vstack([left[:-1], right])
        return np.array([start, end])

    return _simplify(points)


def schedule_from_trajectory(
    t_s: list[float], T_C: list[float], tol_C: float = 12.0, min_hold_s: float = 300.0
) -> dict:
    """Compress a noisy datalogger trajectory into schedule knots.

    Uses Ramer-Douglas-Peucker line simplification on the (time, temp)
    points (after light smoothing), so genuine setpoint changes become
    knots and controlled RAMPS are preserved as diagonal segments — the
    piecewise-linear :func:`furnace_T` interpolates them exactly as the
    furnace behaved. This replaces the old change-point detector that
    turned every 40-minute heatup ramp into 8 flat soak segments and
    recorded overshoot as a deliberate setpoint.

    Returns ``{"schedule_times": [...], "schedule_temps_C": [...]}`` —
    time knots in seconds, temperature knots in °C.
    """
    t = np.asarray(t_s, dtype=np.float64)
    T = np.asarray(T_C, dtype=np.float64)
    if len(t) < 4:
        return {"schedule_times": t.tolist(), "schedule_temps_C": T.tolist()}

    # light smoothing to keep the RDP fit from chasing sensor noise.
    # np.convolve(mode='same') dilutes the endpoints (a 1% window drags the
    # first knot from 25 C to 18 C and fabricates a corner in a flat soak),
    # so we use np.pad(edge) before convolving and trim back — the padded
    # samples keep the edges honest.
    win = max(3, int(len(T) * 0.01))
    kernel = np.ones(win) / win
    pad = win // 2
    T_pad = np.pad(T, (pad, pad), mode="edge")
    Ts = np.convolve(T_pad, kernel, mode="valid")
    # convolve(valid) length can drift by 1 from len(T) when win is even;
    # trim to the time axis exactly
    if len(Ts) > len(T):
        Ts = Ts[: len(T)]
    elif len(Ts) < len(T):
        Ts = np.concatenate([Ts, [T[-1]] * (len(T) - len(Ts))])

    pts = np.column_stack([t, Ts])
    simplified = _rdp_simplify(pts, epsilon=tol_C)

    # RDP keeps the smoothed endpoints; snap them to the RAW trajectory so
    # the schedule starts at the true initial temperature.
    simplified[0] = np.array([simplified[0][0], T[0]])
    simplified[-1] = np.array([simplified[-1][0], T[-1]])

    # merge plateau knots closer than min_hold_s (a real soak has many
    # near-identical samples; RDP keeps the plateau endpoints). If a knot
    # lands within min_hold_s of the previous one, drop it: the two are
    # within tol_C by construction, so the previous knot already represents
    # the temperature.
    kept_pts = [simplified[0]]
    for p in simplified[1:]:
        if p[0] - kept_pts[-1][0] >= min_hold_s:
            kept_pts.append(p)
    kept = np.asarray(kept_pts)

    times: list[float] = [float(x) for x in kept[:, 0]]
    temps: list[float] = [float(x) for x in kept[:, 1]]
    if times[-1] < t[-1]:
        times.append(float(t[-1]))
        temps.append(temps[-1])
    return {"schedule_times": times, "schedule_temps_C": temps}
