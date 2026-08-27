# CLI Reference

`ferrumize` is a [Typer](https://typer.tiangolo.com/) app. Run
`ferrumize --help` for the complete surface, `ferrumize CMD --help` for any
single command's options. Exit code is `0` on success and `1` on failure or
a blocked gate.

Global options (before the command):

| Flag | Meaning |
|---|---|
| `--seed N` | Global RNG seed (deterministic runs). Default `0`. |
| `--verbose` | Debug logging. |

## Commands

| Command | Purpose | Example |
|---|---|---|
| `validate` | Validate a run-config YAML before spending compute | `ferrumize validate data/synthetic/calibration_data.yaml` |
| `simulate` | Run the forward pipeline for a scenario, write T/C/H/ECD | `ferrumize simulate CONFIG --out results/simulate` |
| `calibrate` | NUTS Bayesian calibration against a measured traverse (gates: R̂ < 1.01, bulk ESS > 400) | `ferrumize calibrate DATA.yaml --chains 4 --draws 1000` |
| `design` | Inverse-design a schedule to hit a target ECD; add `--penalty energy` for the Pareto front | `ferrumize design 0.15 --alloy 8620 --penalty energy` |
| `identifiability` | Fisher/correlation analysis: single- vs two-schedule identifiability | `ferrumize identifiability CONFIG` |
| `ingest` | Parse a messy PLC/datalogger export into normalized trajectory + traverse | `ferrumize ingest /path/to/plc.log --out results/ingested` |
| `verify` | Run the full V1–V8 + Q1–Q3 gate suite; nonzero exit on any FAIL | `ferrumize verify` |
| `figures` | Regenerate the F1–F10 figures deterministically (seeded) | `ferrumize figures --only F3,F8` |
| `app` | Launch the Virtual Furnace Streamlit app | `ferrumize app` |

## Notes

Run times and memory costs per command, plus requirements (minimum vs
recommended) and the pitfalls of running the long jobs, are in
[reproducing.md](reproducing.md).

- `calibrate` output is **blocked from release** unless the convergence gates
  pass — a gate failure is a nonzero exit, not a warning.
- `design` reports the physically reachable ECD range when the target is
  outside bounds instead of silently grinding toward an impossible value.
- `ingest` emits every assumption it made as a warning (unit conversions,
  skipped rows) — nothing is silently dropped. See
  [docs/ingestion.md](ingestion.md).
- `figures` is deterministic: the same seed produces byte-identical images on
  the same matplotlib version.

Implementation: [`app/ferrumize/cli.py`](../app/ferrumize/cli.py).
