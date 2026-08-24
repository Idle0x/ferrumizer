# ferrumizer

[![Python](https://img.shields.io/badge/python-3.12%2B-23262A)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-D6B57C)](LICENSE)
[![Verification](https://img.shields.io/badge/verification-V1--V8%2BQ-5D3320)](docs/verification.md)
[![Repo](https://img.shields.io/badge/github-Idle0x%2Fferrumizer-23262A)](https://github.com/Idle0x/ferrumizer)

> **Gradients through the furnace.**
> The differentiable heat-treatment engine — gas carburizing modeled end-to-end
> as one differentiable pipeline: furnace schedule → thermal history → carbon
> diffusion → phase transformation → hardness → effective case depth.

Ferrumizer is a **Tesseract Hackathon 2026** submission.
**Track 04 — Differentiable inference & UQ**, cross-track with **Track 02 —
Multi-physics & coupled systems**. It composes three independently buildable
Tesseracts across a real differentiation boundary, then uses end-to-end
gradients to (a) **calibrate** process parameters against measured hardness
traverses with full Bayesian uncertainty, and (b) **design** furnace schedules
that hit a target effective case depth (ECD) with an energy trade-off.

---

## Table of contents

1. [What this project is](#what-this-project-is)
2. [Why differentiable heat treatment](#why-differentiable-heat-treatment)
3. [Architecture: three Tesseracts, one gradient](#architecture-three-tesseracts-one-gradient)
4. [Why Tesseract, and why Track 04](#why-tesseract-and-why-track-04)
5. [Quickstart](#quickstart)
6. [What you can do with it](#what-you-can-do-with-it)
7. [The figures: what each one proves](#the-figures-what-each-one-proves)
8. [Physics model reference](#physics-model-reference)
9. [Verification](#verification)
10. [Extending Ferrumizer](#extending-ferrumizer)
11. [Honest limitations](#honest-limitations)
12. [Future work](#future-work)
13. [Repository layout](#repository-layout)
14. [Documentation](#documentation)
15. [Citation & license](#citation--license)

---

## What this project is

Gas carburizing is how industry hardens the *surface* of low-carbon steel
parts (gears, shafts, bearings) without hardening the core: the part is held
in a carbon-rich furnace atmosphere at ~900–1050 °C, carbon diffuses into the
surface, and a quench transforms the carbon-enriched austenite into hard
martensite. The engineering quantity that matters is the **effective case
depth (ECD)** — the depth at which hardness crosses 550 HV (ISO 2639) — because
it is the specification the customer pays for.

Ferrumizer treats that whole chain as **one differentiable function**:

```
furnace schedule → T(x,t) → C(x,t) → H(x) → ECD
```

Every stage is differentiable in its parameters (emissivity, diffusivity
prefactor D₀, activation energy Q, carbon potential C_pot, mass-transfer
coefficient h_m), so gradients flow from the final hardness curve *backward
through all three stages*. That is what makes two things possible that trial
and-error cannot do:

1. **Calibration**: given a measured hardness traverse, recover the furnace's
   actual effective parameters — with an uncertainty estimate, not a point
   guess.
2. **Inverse design**: given a target ECD, solve for the schedule that produces
   it, with an optional energy penalty for the gas bill.

The project is deliberately honest about what it is: a **process emulator** —
a fast, transparent model of the physics — not a commercial FE/CFD package,
and not a claim that heat treatment can be reduced to a slider. Every
approximation is documented in [docs/physics.md](docs/physics.md) and the
[ADRs](docs/architecture.md).

---

## Why differentiable heat treatment

The conventional workflow for developing a carburizing cycle is empirical:
run a furnace, cut a part, measure a traverse, adjust, repeat. Each cycle
costs a part, furnace time, and lab time. The problems with that loop:

- **The schedule is a high-dimensional knob.** Soak temperature, soak time,
  carbon potential, boost/diffuse split, quench severity — ECD depends on all
  of them, nonlinearly and jointly. Trial-and-error searches one direction at
  a time.
- **Parameters are furnace-specific.** The textbook diffusion constants are
  not your furnace's constants. Radiative emissivity, atmosphere behavior, and
  mass-transfer conditions vary by furnace, load geometry, and age. A model
  with fixed constants silently drifts from reality.
- **Measurement is expensive.** A traverse is ~10 points of metallography.
  Making the most of each measurement (Bayesian inference) beats averaging.

Differentiability changes the loop: instead of "guess, measure, adjust",
Ferrumizer solves **"what schedule produces this depth?"** as an optimization,
and "what are my furnace's parameters, given the data I have?" as an inference
problem — the two questions practitioners actually pay to answer.

The carbon diffusion stage is deliberately dual:

- a **legacy NumPy finite-difference box** whose parameter Jacobian is computed
  by central finite differences (the way a 1990s solver would do it), and
- a numerically **identical JAX twin** whose composition derivatives are exact
  autodiff.

This is not redundancy for its own sake: it is the *boundary*. The pipeline
keeps the legacy box as a first-class citizen (forward computation is NumPy,
Jacobian is FD) and uses the JAX twin only as a gradient reference. The
composition of the two — gradients crossing a real implementation boundary —
is exactly what Tesseract is for. See [V4/V4c](#verification).

---

## Architecture: three Tesseracts, one gradient

```
┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│      thermal-stage       │      │     carburizing-stage    │      │     hardening-stage      │
│   (JAX, explicit FTCS)   │      │  (legacy NumPy FD box +  │      │ (JAX: Andrews Ms, KM,    │
│   conduction + Robin BC  │ ───► │   JAX twin, Scheil/JMAK) │ ───► │  JMAK, hardness mixing,  │
│   T(x,t)                 │ T_s  │   C(x,t)                 │ C_f  │  ECD @ 550 HV            │
└──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
        ▲                                    ▲                                    ▲
        └───────────── end-to-end gradients (∂ECD/∂θ) ─────────────┘            │
```

Each stage lives in its own directory under
[`components/`](components/) with a `tesseract_api.py` (the contract:
InputSchema / OutputSchema / apply), a `tesseract_config.yaml` (container
build), and test cases:

1. **`components/thermal-stage`** — 1-D conduction with convective + radiative
   Robin boundary conditions (JAX, explicit FTCS, stability-enforced).
2. **`components/carburizing-stage`** — carbon diffusion (Fick with Arrhenius
   D(T)), Dirichlet or mass-transfer boundary, legacy NumPy FD forward with FD
   parameter Jacobian + JAX twin.
3. **`components/hardening-stage`** — Andrews (1965) martensite-start, KM
   martensite fraction, Scheil-additivity JMAK for diffusional phases
   (bainite/pearlite), smoothstep hardness mixing, ISO 2639 ECD.

The composition happens in
[`app/ferrumize/pipeline.py`](app/ferrumize/pipeline.py), with two execution
paths:

- `FerrumizerPipeline.forward()` — pure-JAX fast path (used by calibration,
  design, figures, the app).
- `FerrumizerPipeline.forward_containers()` — the **real container path**:
  routes the same computation through the three Tesseract components via
  `tesseract_jax.apply_tesseract`, so gradients provably cross **two container
  boundaries**. This is what V4c verifies.

The two paths agree to floating-point tolerance.

---

## Why Tesseract, and why Track 04

The hackathon asks for three things, in order: composition across a real
boundary, gradients doing real work, and a real problem where Tesseract is
load-bearing.

**Why Tesseract for this project:**

- The three stages have genuinely different compute characters: a legacy
  NumPy FD box (deliberately not autodiffed), a JAX tensor pipeline, and a
  small phase-transformation kernel. Tesseract's contract-first isolation
  keeps them independently buildable and testable, and lets the composition
  cross the FD↔AD boundary *without hiding the implementation trade-off*.
- Dependency isolation: the carburizing box runs in its own container with its
  own environment; calibration/design never import its internals, only its
  schema.
- The verification suite and the container path share the same schemas, so
  local dev and containerized execution stay in lockstep.

**Why Track 04 (Differentiable inference & UQ):** the rubric explicitly names
this track as "an expensive or black-box solver wrapped as a Tesseract and
dropped into a probabilistic workflow for Bayesian calibration... the solver
may expose its Jacobian by autodiff or by finite differences; the composition
with the inference engine is the contribution." That is precisely the
architecture here: NumPyro NUTS over a Tesseract-wrapped solver that exposes
its Jacobian both ways. Cross-track 02 because the pipeline is inherently
multi-physics (thermal → carbon → hardening) and the inverse problem spans
all three stages.

**Why not another approach:** a monolithic JAX reimplementation of all three
stages would be simpler to write but would *remove the boundary* — the thing
the competition rewards. A commercial-FE wrapper (e.g. PyMAPDL/ANSYS, which
won 2025) would be impressive but closed, heavy, and unverifiable in a
hackathon window. The chosen design keeps the legacy box honest, the gradients
provable, and the whole thing reproducible on a laptop.

---

## Quickstart

```bash
git clone https://github.com/Idle0x/ferrumizer.git
cd ferrumizer
uv sync --extra app --extra dev --extra docs   # extras are exclusive — include them all
make data
ferrumize verify      # full verification suite (V1–V8 + Q1–Q3), ~25 min
ferrumize figures     # regenerate all 10 figures into figures/
ferrumize app         # launch the Virtual Furnace app
```

Requirements: Python 3.12+, `uv` (or pip with the same extras), JAX on CPU is
sufficient; the verification suite runs in ~25 minutes on a modern laptop.

## Ways to run it (three surfaces, one engine)

Everything below runs the **same physics engine** — the CLI, the app, and the
Python API are different front ends to the identical pipeline, so a schedule
designed in one reproduces exactly in the others.

**1. Terminal (CLI)** — scriptable, headless, the reproducibility surface:

```bash
ferrumize validate CONFIG
ferrumize simulate CONFIG --out results/simulate
ferrumize calibrate DATA.yaml --chains 4 --draws 1000
ferrumize design 0.15 --alloy 8620 --penalty energy
ferrumize ingest /path/to/plc.log --out results/ingested
ferrumize verify
ferrumize figures
```

**2. Browser (Streamlit app)** — interactive "what-if" surface:

```bash
ferrumize app          # opens http://localhost:8501
```

Three tabs: **Virtual Furnace** (schedule/quench/alloy sliders → live
T/C/H/ECD), **Cycle Predictor** (upload a traverse or raw PLC log → NUTS
posterior), **Log Ingestion** (preview what the parser extracts from a messy
furnace export).

**3. Python API** — embed Ferrumizer in your own tooling (a Jupyter notebook,
a CI check, another app, a scheduling service):

```python
from ferrumize.pipeline import FerrumizerPipeline, Scenario, ProcessParams

res = FerrumizerPipeline(
    Scenario(quench_medium="oil", quench_temp_K=333.15, size_mm=16.0),
    ProcessParams(C_pot=1.0),
).forward()
print(res["ecd_mm"])  # 0.2134 (or your number)
```

For the containerized Tesseract composition (the hackathon path), use
`FerrumizerPipeline(...).forward_containers()` — it routes the same
computation through the three real components via `tesseract_jax`, and V4c
verifies gradients cross those container boundaries.

**Reproducing results:** every figure, verification gate, and dataset is
regenerated by a documented command (`ferrumize figures`, `ferrumize verify`,
`make data`) — all seeded, so outputs are byte-stable on the same dependency
versions. The README's [figures section](#the-figures-what-each-one-proves)
gives the per-figure regeneration command.

---

## What you can do with it

| Command | What it does |
|---|---|
| `ferrumize simulate CONFIG` | Run the forward pipeline for a scenario and write T/C/H/ECD results. |
| `ferrumize calibrate DATA.yaml --chains 4 --draws 1000` | NUTS Bayesian calibration against a measured traverse; gates on R̂ < 1.01 and bulk ESS > 400. |
| `ferrumize design TARGET_ECD_MM --alloy 8620` | Inverse-design a schedule hitting a target ECD; add `--penalty energy` for the Pareto front. |
| `ferrumize identifiability CONFIG` | Fisher/correlation analysis: why one schedule leaves D₀–Q tangled and two don't. |
| `ferrumize ingest PLC_LOG` | Parse a messy furnace PLC/datalogger export into normalized trajectory + traverse. |
| `ferrumize verify` | Run the V1–V8 + quench gate table. |
| `ferrumize figures` | Regenerate all figures deterministically (seeded). |
| `ferrumize app` | Launch the Streamlit Virtual Furnace. |

The app (`ferrumize app`) is the interactive surface: drag schedule, quench,
and (optionally) custom alloy chemistry; watch temperature, carbon, hardness,
and the Case-Depth Dial update live; upload a traverse **or a raw PLC log**
to run calibration; and inspect what the PLC ingestion parser extracts. It is
a *predictor/emulator*, not a toy — every curve is the same physics the CLI
solves.

### PLC log ingestion

Real furnace logs arrive as company banners, mixed units, quoted cells, junk
rows, and random delimiters. `ferrumize ingest` (and the app's Log Ingestion
tab) sniffs the structure: auto-detected delimiter and header row, column-role
mapping (time/temperature/depth/hardness via synonym table), deg C ↔ deg F
conversion, time-unit normalization, malformed-row skipping with warnings, and
compression of a noisy trajectory into piecewise-constant soak segments ready
for a `Scenario`. Implementation:
[`app/ingest/plc_parser.py`](app/ingest/plc_parser.py).

### Dynamic alloy chemistry

The three shipped presets (8620, 9310, 5120) are literature-anchored YAML in
[`components/shared/ferrumizer_physics/alloys/`](components/shared/ferrumizer_physics/alloys/).
You are not limited to them: `composition_to_preset()` builds a full physics
preset from bare composition (wt-%) using published correlations — Andrews'
multi-element Ms line, case-hardness plateau anchored at ~0.9% C with a
hardenability bump, gamma-iron diffusion defaults — and the pipeline accepts
it directly (`FerrumizerPipeline(scenario, params, preset=preset)`). The app's
**Custom…** alloy option uses exactly this path. Estimation rules are
documented as estimates, not certified constants:
[`components/shared/ferrumizer_physics/alloys.py`](components/shared/ferrumizer_physics/alloys.py).

### Finite-rate quench model

The default forward path previously assumed an **instantaneous** quench to
298 K (a common modeling shortcut). Ferrumizer now ships a finite-rate quench
model: a lumped-Newton cooling curve whose rate depends on quench medium
(oil/water/polymer/air), bath temperature, agitation, and part size, with
Scheil-additivity JMAK integration over the cooling curve for pearlite and
bainite. The consequence is real and visible: a slow quench converts austenite
to diffusional phases, surface hardness collapses, and ECD drops to zero —
the actual production failure mode that an instant-quench model can never
predict. Comparison: air quench → 100% pearlite / 0 HV case; water quench →
~96% martensite / full case. See
[`components/shared/ferrumizer_physics/hardening.py`](components/shared/ferrumizer_physics/hardening.py)
and the figures below.

---

## The figures: what each one proves

All figures are generated deterministically by
[`app/ferrumize/figures.py`](app/ferrumize/figures.py) — the RNG seed is fixed,
so anyone can reproduce them bit-for-bit with `ferrumize figures` (or
`ferrumize figures --only F3,F8` for a subset). They are *not* screenshots of
the app: they are the same physics, rendered as publication artifacts.

### F1 — The hero loop: one heat treatment, animated

![F1](figures/F1_hero_loop.gif)

What it shows: a single carburizing cycle, animated. Furnace schedule → part
temperature → carbon soaking into the surface → hardness profile → ECD number
counting up. This is the ten-second answer to "what does this tool do?"

How to regenerate: `ferrumize figures --only F1`.

### F2 — Architecture

![F2](figures/F2_architecture.png)

What it shows: the three Tesseract stages and the end-to-end gradient flow
through the composition.

How to regenerate: `ferrumize figures --only F2`.

### F3 — Analytic validation: erfc overlay

![F3](figures/F3_erfc_overlay.png)

What it shows: for a semi-infinite slab with constant surface carbon, carbon
diffusion has an exact analytic solution (the error function). This overlays
the solver's numerical C(x) against the analytic curve; the two lines coincide
(normalized L2 ≈ 2e-4, printed in the title). Meaning: the diffusion math is
not hand-waving — it matches the textbook truth.

How to regenerate: `ferrumize figures --only F3`.

### F4 — Convergence: method of manufactured solutions

![F4](figures/F4_mms_convergence.png)

What it shows: a solution we know exactly is planted, then the solver is run
at successively finer grids. Error vs grid spacing on a log-log scale falls
at the predicted order (≥1.85). Meaning: the numerical discretization
converges the way numerical analysis says it must.

How to regenerate: `ferrumize figures --only F4`.

### F5 — The boundary proof: FD vs autodiff

![F5](figures/F5_cross_ad.png)

What it shows: the same gradients computed two ways — finite differences
through the legacy NumPy box, autodiff through the JAX twin. The bars agree to
~10 decimal places (relative ∞-norm ≈ 6e-10 in the title). Meaning: crossing
the implementation boundary loses nothing.

How to regenerate: `ferrumize figures --only F5`.

### F6 — Calibration posterior

![F6](figures/F6_posterior.png)

What it shows: after NUTS calibration against a synthetic traverse, the
posterior distributions over {log D₀, Q, C_pot, h_m, ε}. Tight peaks mean the
data pins the parameter; wide/flat marginals mean it does not (see F8). The
convergence gates (R̂, ESS) are enforced separately by the CLI.

How to regenerate: `ferrumize figures --only F6` (needs a calibration run —
see `docs/calibration.md`).

### F7 — Robustness: recovery vs measurement noise

![F7](figures/F7_noise_sweep.png)

What it shows: the parameter-recovery error as measurement noise increases
from 0 to 20 HV. The gentle upward trend means the method degrades gracefully
with realistic, noisy traverses rather than breaking.

How to regenerate: `ferrumize figures --only F7`.

### F8 — Identifiability: why two schedules beat one

![F8](figures/F8_identifiability.png)

What it shows: with a single temperature schedule, D₀ and Q are statistically
tangled (collinear); with two different schedules, the correlation structure
collapses and the parameters become identifiable. This is the figure that tells
a practitioner **how to use the tool correctly**: run two furnace cycles, not
one.

How to regenerate: `ferrumize figures --only F8`.

### F9 — Design: the ECD-vs-energy Pareto front

![F9](figures/F9_pareto.png)

What it shows: every point is the best schedule found under a given energy
penalty. X = energy proxy (time-integral of setpoint above ambient), Y =
achieved ECD. Meaning: how much case depth must be given up to save gas — the
trade-off a process engineer takes to a meeting.

How to regenerate: `ferrumize figures --only F9`.

### F10 — Alloy comparison strip

![F10](figures/F10_alloy_strip.png)

What it shows: the same recipe applied to the three shipped alloys, side by
side, each labeled with its resulting ECD — the comparison view across 8620 /
9310 / 5120.

How to regenerate: `ferrumize figures --only F10`.

---

## Physics model reference

Detailed derivations and constants live in [docs/physics.md](docs/physics.md)
and the two ADRs ([ADR-001](docs/adr/ADR-001-carbon-diffusion-prefactor.md),
[ADR-002](docs/adr/ADR-002-calibration-surrogate.md)). Summary:

| Stage | Model | Key assumptions |
|---|---|---|
| Thermal | 1-D explicit FTCS conduction, convective + radiative Robin BC (JAX) | slab/axisymmetric radial; no CAD, stress, distortion |
| Thermal surrogate (calibration) | Lumped capacitance with surface-T sampling (ADR-002) | validated V1 ≤ 0.5% |
| Carbon | Fick with Arrhenius D(T) = D₀·exp(−Q/RT), Dirichlet or mass-transfer BC | 1-D; no grain-boundary/geometry effects |
| Hardening | Andrews Ms, Koistinen–Marburger, Scheil-additivity JMAK, smoothstep hardness mixing, ISO 2639 ECD | carbon-proxy hardness + rule of mixtures; JMAK is a documented approximation |
| Quench (new) | Lumped-Newton cooling curve (medium, bath T, agitation, part size) + Scheil-JMAK pearlite/bainite; KM on surviving austenite | single characteristic cooling curve per part; C-curve constants are order-of-magnitude, not certified TTT data |

Every free parameter in the shipped presets carries provenance in the YAML
comments or an ADR; nothing is fit-to-look-nice.

---

## Verification

Run everything with `ferrumize verify` (each gate is an independent script
under `verification/`).

| Gate | Check | Contract |
|---|---|---|
| V1 | Lumped capacitance | max relative error < 0.5% |
| V2 | Semi-infinite erfc | normalized L2 < 1e-3 |
| V3 | MMS operators | order >= 1.85 |
| V4 | FD box vs JAX twin | relative infinity norm < 1e-3 |
| V4c | Container composition gradient (through 2 boundaries) | finite, non-zero, within 20% of FD |
| V5 | Runtime gradient checks | zero failures on AD boxes |
| V6 | Two-schedule recovery | max relative error < 1e-4 |
| V7 | SBC/TARP | SBC p > 0.05; coverage band |
| V8 | Synthetic 8620 traverse reconstruction (literature-anchored params) | within stated reconstruction error bar |
| Q1–Q3 | Quench model sanity | medium ranking (air > oil > water), collapse on slow quench, differentiability |

Tests: `pytest tests/` (unit + regression, including the new quench, dynamic
alloy, and PLC-ingestion tests). Lint: `ruff`. Types: `mypy` clean on 15
source files.

---

## Extending Ferrumizer

Beyond the shipped commands, the pieces are designed to be reused:

- **New alloy**: drop a `aisi_XXXX.yaml` in
  `components/shared/ferrumizer_physics/alloys/` following the 8620 schema, or
  use `composition_to_preset()` for a bare chemistry at runtime (CLI/app).
- **PLC/datalogger ingestion**: `app/ingest/plc_parser.py` returns normalized
  trajectory + traverse; feed the traverse straight into `calibrate`.
- **New stage**: implement a `tesseract_api.py` with the Input/Output schema,
  add it to `FerrumizerPipeline.forward_containers`, and it composes.
- **Different quench media**: the film coefficients live in
  `QUENCH_MEDIA_H` in `hardening.py`; the C-curve constants in the alloy YAMLs.
- **Other paths worth exploring** (not shipped): full 3-D geometry via a
  mesher Tesseract (the 2025 winner's pattern); Enzyme-differentiated Fortran
  legacy ports (the forum's HMC showcase pattern); PyTorch front-ends through
  `tesseract-torch`; field-level HMC over the C(x) field instead of scalar
  parameters.

---

## Honest limitations

- **1-D geometry only.** No CAD, meshing, stress, distortion, or 3-D effects.
  Part size enters as a characteristic length.
- **The literature traverse is a synthetic reconstruction** anchored to
  published parameter ranges — explicitly *not* a digitization of a specific
  furnace dataset. The method is validated against synthetic ground truth;
  the *industrial* claim would require a real measured traverse.
- **JMAK/Scheil C-curve constants are order-of-magnitude**, representative of
  low-alloy carburizing steels, not certified TTT data for a specific heat.
  The quench model is qualitative-to-semiquantitative: it predicts failure
  modes and rankings, not certified phase fractions.
- **The energy proxy is a relative penalty axis**, not an absolute energy
  figure.
- **Hardness is a carbon-proxy mixing rule**, not a microstructure-resolved
  model.
- **D₀/Q defaults** carry documented uncertainty (ADR-001); calibration
  exists precisely because fixed constants drift from real furnaces.

---

## Future work

Realistic, not hype:

1. **Real furnace data**: calibrate against a genuinely measured traverse
   (the single highest-value next step; the machinery is ready).
2. **Grossmann quench-severity inversion** in the quench model.
3. **3-D geometry** via a meshing Tesseract (CAD→mesh→FEM pattern).
4. **Hall–Petch grain-size coupling** to hardness.
5. **Uncertainty-aware design**: propagate the calibration posterior into the
   design objective instead of point estimates.
6. **Enzyme/Fortran legacy-port experiments** across a real language boundary.
7. **Digital-twin integration layer** for furnace telemetry (PLC ingestion is
   the first step).

---

## Repository layout

| Path | Purpose |
|---|---|
| [`components/`](components/) | The three Tesseracts (thermal / carburizing / hardening), each with API, config, requirements, test cases |
| [`components/shared/ferrumizer_physics/`](components/shared/ferrumizer_physics/) | Physics library (thermal, carbon, hardening, alloys) + alloy presets |
| [`app/ferrumize/`](app/ferrumize/) | CLI, pipeline composition, fast differentiable model, figures |
| [`app/calibration/`](app/calibration/) | NumPyro NUTS calibration + convergence gates |
| [`app/design/`](app/design/) | Gradient schedule design + Pareto front |
| [`app/identifiability/`](app/identifiability/) | Fisher/correlation identifiability analysis |
| [`app/ingest/`](app/ingest/) | PLC/datalogger ingestion parser |
| [`app/streamlit_app.py`](app/streamlit_app.py) | The Virtual Furnace app |
| [`verification/`](verification/) | V1–V8 + quench gate scripts |
| [`tests/`](tests/) | Unit + regression tests |
| [`data/`](data/) | Synthetic traverse generators + literature reconstruction (provenance in `PROVENANCE.md`) |
| [`docs/`](docs/) | MkDocs site: physics, architecture, calibration, verification, gallery, roadmap, ADRs |
| [`figures/`](figures/) | Generated figures (regenerate with `ferrumize figures`) |
| [`brand/`](brand/) | Visual identity assets (mark, lockups, favicon) |

---

## Documentation

The MkDocs site ([docs/](docs/)) is the full reference:

- [docs/index.md](docs/index.md) — overview
- [docs/physics.md](docs/physics.md) — physics derivations and constants
- [docs/architecture.md](docs/architecture.md) — pipeline and container composition
- [docs/calibration.md](docs/calibration.md) — Bayesian calibration workflow + runtime expectations
- [docs/design.md](docs/design.md) — schedule design and the Pareto front
- [docs/verification.md](docs/verification.md) — the gate table
- [docs/gallery.md](docs/gallery.md) — figure gallery
- [docs/roadmap.md](docs/roadmap.md) — future work
- [docs/adr/ADR-001-carbon-diffusion-prefactor.md](docs/adr/ADR-001-carbon-diffusion-prefactor.md)
- [docs/adr/ADR-002-calibration-surrogate.md](docs/adr/ADR-002-calibration-surrogate.md)

---

## Citation & license

Apache-2.0. See [CITATION.cff](CITATION.cff) for the citation metadata and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

**Track: 04 — Differentiable inference & UQ** (cross-track with 02 —
Multi-physics & coupled systems). Ferrumizer is a Tesseract Hackathon 2026
submission by riot' — built with the Tesseract framework from Pasteur Labs /
Institute for Simulation Intelligence.
