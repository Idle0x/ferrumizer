# Ferrumizer

**Gradients through the furnace.**

Ferrumizer is a differentiable gas-carburizing heat-treatment engine: it
composes thermal history, carbon diffusion, phase transformation, and
hardness into **one differentiable pipeline** that ends at effective case
depth (ISO 2639, 550 HV). End-to-end gradients make two things possible that
trial-and-error cannot:

1. **Calibrate** — recover your furnace's effective process parameters from a
   measured hardness traverse, with honest Bayesian uncertainty.
2. **Design** — invert the model to find the schedule that hits a target case
   depth, with an energy trade-off if you want it.

It is a Tesseract Hackathon 2026 submission (Track 04 — Differentiable
inference & UQ, cross-track 02 — Multi-physics), composed of three
independently buildable Tesseracts across a real finite-difference/autodiff
boundary.

## Start here

| Goal | Go to |
|---|---|
| What can it do? | [README](../README.md) |
| Run the app (Virtual Furnace) | `ferrumize app` |
| Run the verification suite | `make data && ferrumize verify` (~25 min) |
| Regenerate all figures | `ferrumize figures` |
| Full command surface | [CLI reference](cli.md) |

## Guides

- [Physics](physics.md) — the models and constants behind every curve
- [Architecture](architecture.md) — the three Tesseracts and how gradients cross them
- [Calibration](calibration.md) — Bayesian inference from traverses, gates, runtime
- [Design](design.md) — inverse schedule design and the Pareto front
- [Ingestion](ingestion.md) — parsing messy furnace PLC logs
- [Verification](verification.md) — the V1–V8 + Q1–Q3 gate table
- [Gallery](gallery.md) — the F1–F10 figures
- [API reference](api.md) — module-by-module map of the code
- [Roadmap](roadmap.md) — realistic future work
- [Decision records](adr/) — why the design choices were made

## The one-paragraph summary

Gas carburizing hardens the surface of low-carbon steel parts by holding them
in a carbon-rich furnace atmosphere and quenching. The specification that
matters is **effective case depth** — the depth at which hardness crosses
550 HV. Ferrumizer models the whole chain (schedule → temperature → carbon →
hardness → ECD) as one differentiable function, so you can both **learn your
furnace's real parameters from measurements** and **solve for the schedule
that produces the depth you need** — instead of guess, cut, measure, repeat.

## Honesty note

Every approximation (1-D geometry, synthetic literature reconstruction,
order-of-magnitude C-curve constants, energy proxy) is documented in the
[README limitations](../README.md#limitations) and in the physics
pages. Nothing here is presented as certified commercial FE/CFD results.
