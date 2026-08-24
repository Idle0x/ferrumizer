# Changelog

All notable changes to Ferrumizer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic
versioning. Entries are grouped by **Added / Changed / Fixed / Removed**.

## [0.1.1] — 2026-08-23

### Added

- **Finite-rate quench model** — real quenches are no longer assumed
  instantaneous: lumped-Newton cooling driven by quench medium
  (oil/water/polymer/air), bath temperature, agitation, and part size, with
  Scheil-additivity JMAK integration for pearlite/bainite and KM martensite on
  the surviving austenite. Slow quenches now correctly collapse surface
  hardness and ECD (the production failure mode). New verification gates
  Q1–Q3 wired into `ferrumize verify`.
- **PLC / datalogger ingestion** (`app/ingest/plc_parser.py`) — defensive
  parser for messy furnace exports: delimiter/header/column-role sniffing,
  deg C ↔ deg F conversion, time-unit normalization, malformed-row warnings,
  and trajectory → soak-segment compression. New `ferrumize ingest` command;
  new `docs/ingestion.md` page; Log Ingestion tab in the app.
- **Dynamic alloy chemistry** — `composition_to_preset()` builds a full
  physics preset from bare wt-% composition (Andrews multi-element Ms,
  case-hardness plateau with hardenability bump); `load_alloy()` accepts
  dicts; `validate_preset()` guards schema. The app's Custom… alloy option
  uses this path.
- **Two-stage boost/diffuse editing** in the app (previously the app ran a
  single-stage schedule regardless of the underlying model's capability).
- **Virtual Furnace app overhaul** — renamed from "Furnace Simulator",
  fixed-axis plots so physics changes are visible, numeric deltas on key
  metrics, tooltips on every control, explainer paragraphs under every chart,
  and a runtime/accuracy warning for the light-grid calibration path.
- Quench verification gates Q1–Q3 (`verification/q_quench.py`).
- Tests: `tests/test_new_features.py` (14 tests covering quench, dynamic
  alloys, PLC ingestion).

### Changed

- **README rebuilt** (~100 → ~460 lines): all 10 figures embedded with
  plain-language explanations and regenerate commands, architecture diagram,
  full CLI table, why-Tesseract / why-Track-04 rationale, extension guide,
  honest limitations, repository layout with hyperlinks.
- **`ferrumize design` output** now serializes loss/ECD traces as JSON-safe
  lists (see Fixed).
- Alloy JMAK rate constants corrected from placeholder magnitudes (dead code
  path) to physically realistic C-curve magnitudes; documented in each preset.
- ADR-001 / verification scripts / calibration docstrings: internal plan
  references replaced with public doc links (docs/ADR, docs/verification).
- `docs/cli.md`, `docs/api.md`, `docs/gallery.md`, `docs/verification.md`,
  `docs/roadmap.md`, `mkdocs.yml` updated for the new surface.

### Fixed

- **`ferrumize design` JSON crash** — `TypeError: only length-1 arrays can be
  converted to Python scalars` when writing `design.json` (NumPy arrays in
  `loss_trace`/`ecd_trace`; converted to lists before dump). The optimizer
  itself was already converging correctly (verified ECD 0.1500, loss 3.5e-10).
- **App "Process history" plot** — `KeyError: 't_s'` (thermal output key is
  `times_s`).
- **Streamlit not installed after `uv sync`** — documented that extras are
  exclusive: use `uv sync --extra app --extra dev --extra docs`.
- **Wheel packaging** — `app/ingest` added to the hatch wheel packages so the
  PLC parser ships in distributions.
- **`brand/gen_assets.py`** — hardcoded absolute output path replaced with the
  script's own directory (path-portable).
- **CITATION.cff** — removed placeholder `repository-code` URL.
- **Time-unit heuristic** in PLC parser: minute-scale logs (median dt > 4 s)
  are now correctly converted to seconds (previously only dt > 10 s was
  treated as minutes, missing common 5-minute datalog sampling).

## [0.1.0] — 2026-08-22

### Added

- Initial Ferrumizer pipeline: thermal → carburizing → hardening → ECD, CLI,
  calibration/design scaffolding, and V1–V8 verification harnesses.
- AISI 8620, 9310, and 5120 presets (literature-anchored constants with
  per-field provenance).
- Tesseract component schemas and local composition path
  (`forward_containers`).
