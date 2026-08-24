# Ingestion

Ferrumizer ships a defensive parser for **messy furnace PLC / datalogger
exports**. Real logs arrive with company banners, multi-line headers, mixed
units, quoted fields, separator chaos, and occasional junk rows. The parser
sniffs the structure instead of assuming it:

- **Delimiter auto-detection** (comma, semicolon, tab, pipe)
- **Header-row detection** via a column-role synonym table
  (time / temperature / depth / hardness)
- **deg C ↔ deg F conversion**
- **Time-unit normalization** (seconds / minutes / hours → seconds)
- **Malformed-row skipping** with explicit warnings (never silent data loss)
- **Schedule compression**: a noisy datalogger trajectory is segmented into
  piecewise-constant soak segments ready for a `Scenario`

## CLI

```bash
ferrumize ingest /path/to/plc_export.log --out results/ingested
```

Writes `results/ingested/ingested.json` containing the normalized trajectory,
any extracted hardness traverse, the compressed schedule, and the warnings
the parser emitted.

## App

The **Log Ingestion** tab of the Virtual Furnace app (`ferrumize app`)
previews exactly this parser, and the **Cycle Predictor** tab accepts either a
clean `depth_mm,hardness_HV` CSV *or* a raw PLC log that contains a traverse —
the parser finds the right columns.

## Example

A furnace export with a banner, deg F temperatures, minute-scale timestamps,
quoted cells, and a trailing garbage row is parsed to:

```json
{
  "trajectory": { "t_s": [0.0, ..., 6000.0], "T_C": [21.1, ..., 100.0] },
  "schedule":   { "schedule_times": [0, 900, 1500, 4200, 4800, 5400, 6000],
                  "schedule_temps_C": [295, 929, 945, 888, 807, 165, 165] },
  "warnings": ["Time column interpreted as minutes; converted to seconds.",
               "Skipped 1 malformed row(s) (non-numeric or short)."]
}
```

Implementation: `app/ingest/plc_parser.py`.
