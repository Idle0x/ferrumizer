# Ingestion

Ferrumizer ships a defensive parser for **messy furnace PLC / datalogger
exports**. Real logs arrive with company banners, multi-line headers, mixed
units, quoted fields, separator chaos, and occasional junk rows. The parser
sniffs the structure instead of assuming it:

- **Delimiter auto-detection** (comma, semicolon, tab, pipe)
- **Header-row detection** via a column-role synonym table
  (time / temperature / depth / hardness)
- **deg C ↔ deg F conversion**
- **Time-unit normalization from the HEADER, never from the data**
- **HH:MM:SS clock timestamps** parsed to elapsed seconds
- **Malformed-row skipping** with explicit warnings (never silent data loss)
- **Schedule compression via Ramer-Douglas-Peucker** line simplification:
  heating/cooling ramps stay diagonal segments instead of being chopped
  into a staircase of flat soaks
- **Range validation** on temperatures, depths, and hardness values

## Time units: the rule that prevents 60× errors

The parser **never guesses the time unit from the median time step**. That
heuristic was removed after a review found it silently multiplied
5–30 s datalogger logs by 60 (a 10-second sampling rate became a 300-hour
cycle, and the diffusion model — case depth ~ √t — then over-predicted by
~11×). The unit now comes from, in order:

1. **The column header** — `time [s]`, `elapsed (min)`, `t [h]`, `sec`,
   `minutes`, `hours`, etc. are read from the header text.
2. **HH:MM:SS clock timestamps** — `timestamp` columns with
   `09:15:03`-style cells are converted to elapsed seconds.
3. **Explicit user confirmation** — in the app, the warning banner tells you
   what was assumed.
4. **Default to seconds** with a prominent warning when the header has no
   unit. The parser never auto-rescales ambiguous data.

## Schedule compression: RDP, not change-points

The old compressor fired a new "soak segment" every `min_hold_s` whenever
the temperature deviated from the previous segment start — so a 40-minute
heatup ramp became 8 flat segments, and a 20 °C overshoot was recorded as a
deliberate setpoint. The new compressor applies Ramer-Douglas-Peucker line
simplification (after light edge-honest smoothing): genuine setpoint changes
become knots, ramps are preserved as diagonal segments, and the
piecewise-linear `furnace_T` reproduces the logged behavior exactly.

## CLI

```bash
ferrumize ingest /path/to/plc_export.log --out results/ingested
```

> **`ferrumize: command not found`?** Install it once from the repo root:
> `uv tool install -e . --with streamlit`. Already installed but still not
> found? Make sure `~/.local/bin` is on your `PATH`, or run
> `source .venv/bin/activate` before the command (requires `uv sync`).

Writes `results/ingested/ingested.json` containing the normalized trajectory,
any extracted hardness traverse, the compressed schedule, and the warnings
the parser emitted.

## App

The **Log Ingestion** tab of the Virtual Furnace app (`ferrumize app`)
previews exactly this parser, and the **Cycle Predictor** tab accepts either a
clean `depth_mm,hardness_HV` CSV *or* a raw PLC log that contains a traverse —
the parser finds the right columns.

When a PLC log contains a **trajectory** (time + temperature), the Cycle
Predictor uses it: the compressed schedule replaces the old hardcoded
2 h / 950 °C scenario, so calibration reflects what the furnace actually did
instead of being biased by an assumed thermal history.

## Example

A furnace export with a banner, deg F temperatures, minute-scale timestamps,
quoted cells, and a trailing garbage row is parsed to:

```json
{
  "trajectory": { "t_s": [0.0, ..., 6000.0], "T_C": [21.1, ..., 100.0] },
  "schedule":   { "schedule_times": [0, 900, 1500, 4200, 4800, 5400, 6000],
                  "schedule_temps_C": [295, 929, 945, 888, 807, 165, 165] },
  "warnings": ["Time column header says minutes; converted to seconds (×60).",
               "Skipped 1 malformed row(s) (non-numeric or short)."]
}
```

Implementation: `app/ingest/plc_parser.py`.
