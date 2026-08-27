# Architecture

Ferrumizer is three independently buildable Tesseracts composed into one
differentiable function.

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│    thermal-stage     │   │   carburizing-stage  │   │   hardening-stage    │
│  JAX · FTCS + Robin  │──►│ legacy NumPy FD box  │──►│  Andrews Ms · KM ·   │
│  T(x,t)              │   │ + JAX twin, Scheil   │   │  JMAK · mixing · ECD │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
        ▲                       ▲                            ▲
        └───────────────── end-to-end gradients ────────────┘
```

## The components

Each lives in `components/<name>/` with:

- `tesseract_api.py` — the contract: `InputSchema` / `OutputSchema` / `apply`.
- `tesseract_config.yaml` — container build spec.
- `tesseract_requirements.txt` — container dependencies.
- `test_cases/*.json` — regression fixtures for the container `test` endpoint.

| Component | Computation | Gradient strategy |
|---|---|---|
| `thermal-stage` | 1-D conduction, convective+radiative Robin BC (JAX, explicit FTCS) | autodiff (JAX) |
| `carburizing-stage` | Fick diffusion, Arrhenius D(T), Dirichlet or mass-transfer BC | **finite differences on the legacy NumPy forward** + exact autodiff on the JAX twin |
| `hardening-stage` | Andrews Ms, KM martensite, Scheil-JMAK (quench), smoothstep mixing, ECD | autodiff (JAX) |

The deliberate boundary: the carburizing box keeps its legacy NumPy FD
forward path with a finite-difference parameter Jacobian, while a numerically
identical JAX twin supplies composition derivatives. The composition crosses
that implementation boundary — the whole point of the Tesseract framing.

## Two execution paths

| Path | How | Used by |
|---|---|---|
| `FerrumizerPipeline.forward()` | pure-JAX over the shared physics library | app, calibration, design, figures |
| `FerrumizerPipeline.forward_containers()` | `tesseract_jax.apply_tesseract` through the three real components | cross-AD verification (V4c), composition demo |

Both paths share the same discretization and constants, so their outputs
agree to floating-point tolerance; V4c verifies that gradients flow through
**two real container boundaries** and match finite differences within 20%.

Local development uses `Tesseract.from_tesseract_api` with the same schemas
the container builds use; set the `FERRUMIZER_<STAGE>_IMAGE` environment
variable to run against built images instead.

## The application layer

- `app/ferrumize/pipeline.py` — `Scenario`, `ProcessParams`,
  `FerrumizerPipeline`.
- `app/ferrumize/models.py` — `fast_forward`: lumped-surrogate forward model
  for calibration/design (gradients flow end-to-end).
- `app/ferrumize/figures.py` — deterministic F1–F10 generation.
- `app/calibration/` — NumPyro NUTS + convergence gates.
- `app/design/` — gradient schedule design + Pareto front.
- `app/identifiability/` — Fisher/correlation identifiability analysis.
- `app/ingest/` — PLC/datalogger ingestion.
- `app/streamlit_app.py` — Virtual Furnace (3 tabs).

## Why this architecture (the short version)

The hackathon asks for composition across a **real boundary**, gradients doing
**real work**, and a problem where Tesseract is **load-bearing**. A monolithic
JAX reimplementation of all three stages would be simpler but would delete the
boundary; a commercial-FE wrapper would be heavy and unverifiable. Keeping the
legacy FD box first-class — and proving the composition gradients against it —
is the honest middle: the boundary is real, the gradient claims are checked,
and the whole thing reproduces on a laptop. See the README's
[Tesseract integration](../README.md#tesseract-integration) section.
