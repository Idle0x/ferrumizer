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
    skipped = 0
    for row in raw_rows[start:]:
        if len(row) <= max(colmap.values(), default=-1):
            skipped += 1
            continue
        vals: dict[str, float] = {}
        ok = False
        for role, idx in colmap.items():
            v = _to_float(row[idx])
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
        for role, v in vals.items():
            arrays[role].append(v)

    report.rows_used = len(arrays["time"]) if arrays["time"] else report.rows_total - start
    if skipped:
        report.warnings.append(f"Skipped {skipped} malformed row(s) (non-numeric or short).")

    # Build outputs
    if len(arrays["time"]) >= 2 and len(arrays["temp"]) >= 2:
        t = np.asarray(arrays["time"], dtype=np.float64)
        T = np.asarray(arrays["temp"], dtype=np.float64)
        # normalize time to seconds if it looks like minutes or hours
        span = float(t[-1] - t[0])
        if span > 0:
            dt = np.diff(t)
            med = float(np.median(dt)) if len(dt) else 0.0
            if med > 90.0:  # hours
                t = t * 3600.0
                report.warnings.append("Time column interpreted as hours; converted to seconds.")
            elif med > 4.0:  # minutes (a furnace soak at >4 s sampling is a minute-scale log)
                t = t * 60.0
                report.warnings.append("Time column interpreted as minutes; converted to seconds.")
        order = np.argsort(t)
        report.trajectory = {
            "t_s": [float(x) for x in t[order]],
            "T_C": [float(x) for x in T[order]],
        }

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
            report.warnings.append("Depth column not strictly increasing; sorted by depth.")

    if not report.has_trajectory and not report.has_traverse:
        report.warnings.append(
            "No usable trajectory or traverse columns found. Expected time+temperature "
            "and/or depth+hardness columns."
        )

    return report


def schedule_from_trajectory(
    t_s: list[float], T_C: list[float], tol_C: float = 12.0, min_hold_s: float = 300.0
) -> dict:
    """Compress a noisy datalogger trajectory into piecewise-constant soak segments.

    Real furnace setpoints are stepped (heat, soak, diffuse) but the logged
    temperature wanders around the setpoint. This segments the trajectory by
    change points: when the smoothed temperature moves by more than ``tol_C``
    for longer than ``min_hold_s``, a new segment begins. Returns the
    piecewise-constant schedule (time knots + temperature knots) suitable for
    :class:`ferrumize.pipeline.Scenario`.
    """
    t = np.asarray(t_s, dtype=np.float64)
    T = np.asarray(T_C, dtype=np.float64)
    if len(t) < 4:
        return {"schedule_times": t.tolist(), "schedule_temps_C": T.tolist()}

    # simple moving average (window ~ 1% of samples, min 3)
    win = max(3, int(len(T) * 0.01))
    kernel = np.ones(win) / win
    Ts = np.convolve(T, kernel, mode="same")

    seg_starts = [0]
    for i in range(win, len(Ts) - 1):
        if abs(Ts[i] - Ts[seg_starts[-1]]) > tol_C and (t[i] - t[seg_starts[-1]]) > min_hold_s:
            seg_starts.append(i)
    seg_starts.append(len(T) - 1)

    times: list[float] = []
    temps: list[float] = []
    for k, s in enumerate(seg_starts[:-1]):
        e = seg_starts[k + 1]
        times.append(float(t[s]))
        temps.append(float(np.median(T[s:e + 1])))
    times.append(float(t[-1]))
    temps.append(temps[-1] if temps else float(T[-1]))

    return {"schedule_times": times, "schedule_temps_C": temps}
