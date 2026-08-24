# API Reference

Ferrumizer is organized in three layers: the **component contracts**
(per-Tesseract schemas), the **shared physics library** (the math), and the
**application layer** (pipeline, calibration, design, ingestion, figures).

## Component contracts (`components/*/tesseract_api.py`)

Each Tesseract exposes the same three things:

- `InputSchema` / `OutputSchema` — validated (JSON) schemas for its boundary.
- `apply(...)` — the pure computation.
- `test` cases — regression fixtures in `test_cases/`.

| Component | Inputs → Outputs |
|---|---|
| `components/thermal-stage` | schedule + thermal props → `T_surface` history, `Tcore`, final T field |
| `components/carburizing-stage` | `T_surface_history` + diffusion params → `C_final` profile |
| `components/hardening-stage` | `C_profile` + phase constants → hardness `H`, `ecd_mm` |

## Shared physics (`components/shared/ferrumizer_physics/`)

| Module | Contents |
|---|---|
| `thermal.py` | 1-D conduction, Robin BC, lumped-capacitance surrogate (`lumped_surface_T`), stability-enforced time step |
| `carbon.py` | Fick diffusion with Arrhenius D(T), Dirichlet / mass-transfer BC, adaptive sub-stepping |
| `hardening.py` | Andrews Ms, Koistinen–Marburger, Scheil-additivity JMAK, **finite-rate quench model** (`quench_fractions`, `QUENCH_MEDIA_H`), smoothstep hardness mixing, ISO 2639 ECD |
| `alloys.py` | Preset loader (`load_alloy`, `list_alloys`), **dynamic chemistry** (`composition_to_preset`), schema validation (`validate_preset`) |
| `alloys/*.yaml` | Shipped presets: AISI 8620, 9310, 5120 |

## Application layer (`app/`)

| Module | Contents |
|---|---|
| `app/ferrumize/pipeline.py` | `Scenario`, `ProcessParams`, `FerrumizerPipeline` — pure-JAX `forward()` and container `forward_containers()` |
| `app/ferrumize/models.py` | `fast_forward` — the lumped-surrogate forward model used by calibration/design (gradients flow end-to-end) |
| `app/ferrumize/figures.py` | Deterministic F1–F10 figure generation |
| `app/calibration/calibrate.py` | NumPyro NUTS + convergence gates + posterior summary |
| `app/design/optimize.py` | Gradient schedule design + Pareto front |
| `app/identifiability/analyze.py` | Fisher/correlation identifiability analysis |
| `app/ingest/plc_parser.py` | PLC/datalogger ingestion: `parse_plc_log`, `schedule_from_trajectory` |
| `app/streamlit_app.py` | Virtual Furnace Streamlit app (3 tabs) |

## Verification layer (`verification/`)

Each `V*` / `Q*` module exposes a `run_vN()` / `run_qN()` callable returning
`{"passed": bool, ...}`; see [docs/verification.md](verification.md).
