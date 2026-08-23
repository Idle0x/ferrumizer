# ferrumizer

[![Python](https://img.shields.io/badge/python-3.12%2B-23262A)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-D6B57C)](LICENSE)
[![Verification](https://img.shields.io/badge/verification-V1--V8-5D3320)](docs/verification.md)

> **The differentiable heat-treatment engine.**
>
> **Gradients through the furnace.**

## Quickstart

```bash
uv sync
make data
ferrumize verify
ferrumize figures
```

Run a forward simulation:

```bash
ferrumize validate data/synthetic/calibration_data.yaml
ferrumize simulate data/synthetic/calibration_data.yaml --out results/simulate
```

The app models gas carburizing as one pipeline:

```text
furnace schedule -> thermal history -> carbon diffusion -> phase/hardness -> ECD @ 550 HV
```

## Why Differentiable Heat Treatment

- Effective case depth is a thresholded consequence of the complete furnace history, not an isolated curve fit. Ferrumizer propagates sensitivities from emissivity, diffusion parameters, carbon potential, and mass transfer through thermal, carburizing, and hardening stages.
- The carbon box deliberately preserves a legacy NumPy finite-difference forward path. Its parameter Jacobian is central finite difference by design, while a numerically identical JAX twin supplies composition derivatives. The boundary is explicit rather than hidden.
- Calibration reports uncertainty, convergence diagnostics, and identifiability. A single schedule exposes D0-Q collinearity; two temperature schedules are the prescribed fix.

## Architecture

The three stage directories are independently buildable Tesseracts:

1. `components/thermal-stage`: JAX conduction with convective and radiative Robin boundaries.
2. `components/carburizing-stage`: NumPy FTCS carbon diffusion with Arrhenius diffusivity and Dirichlet or mass-transfer boundary conditions.
3. `components/hardening-stage`: JMAK/Scheil hook, Andrews Ms, Koistinen-Marburger martensite, smoothstep hardness, and ISO 2639 ECD.

The real composition path is `FerrumizerPipeline.forward_containers`; local verification uses `Tesseract.from_tesseract_api` and the same schemas used by container builds.

## Verification Summary

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

Run the complete table with `ferrumize verify`.

## Limitations

- Thermal transport is one-dimensional slab or axisymmetric radial; no CAD, meshing, stress, distortion, or 3-D geometry is included in v1.
- The default thermal discretization is an explicit FTCS grid with production guidance `n >= 161`; its stability bound is enforced and never silently clamped.
- The carburizing forward box is intentionally legacy-style NumPy FD. The JAX twin is an interoperability/gradient reference, not a claim that commercial software is internally autodiffable.
- Hardness is a carbon-proxy plus martensite rule of mixtures. JMAK/Scheil support is present as a documented approximation; validated thermodynamic databases are out of scope.
- The literature directory currently contains a transparent synthetic reconstruction anchored to published parameter ranges; it is not presented as a direct digitization of a specific figure.
- The D0 prior correction is recorded in `docs/adr/ADR-001-carbon-diffusion-prefactor.md` because the literal range in the source plan is inconsistent with realistic gamma-iron diffusivity at carburizing temperatures.

## CLI

```text
ferrumize validate CONFIG
ferrumize simulate CONFIG
ferrumize calibrate DATA.yaml --chains 4 --draws 1000
ferrumize design TARGET_ECD_MM --alloy 8620 --penalty energy
ferrumize identifiability CONFIG
ferrumize verify
ferrumize figures
ferrumize app
```

## Roadmap

Enzyme/Fortran legacy-port experiments, 3-D geometry, quench Grossmann inversion, Hall-Petch grain-size coupling, uncertainty-aware design, and a digital-twin integration layer are intentionally deferred.

## Citation

See `CITATION.cff` and the architecture/verification pages in `docs/`.

## Acknowledgment

Ferrumizer is a Tesseract Hackathon 2026 submission focused on differentiable inference, uncertainty quantification, and real solver composition.

**Track: 04 — Differentiable inference & UQ** (cross-track with 02 — Multi-physics & coupled systems): Bayesian calibration of a coupled thermal–carbon–hardening heat-treatment pipeline composed of multiple Tesseracts, with end-to-end gradients driving schedule design.
