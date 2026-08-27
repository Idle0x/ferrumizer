# Design Rationale: Architecture, Differentiability, and System Boundaries

This document details the architectural principles, physical modeling scope,
and differentiable composition strategy of Ferrumizer. It explains the
rationale behind the three-component Tesseract design, the deliberate
finite-difference/automatic-differentiation (FD/AD) boundary, and the
verification mechanisms that ensure end-to-end gradient integrity.

## 1. System Overview

Ferrumizer is a differentiable process emulator for gas carburizing, a
heat-treatment process used to harden the surface of low-carbon steel
components while retaining a ductile core. The physical process consists of
three phases:

1. **Carburization**: The component is held in a carbon-rich atmosphere at
   900–1050 °C. Carbon diffuses into the surface according to Fick's laws,
   creating a concentration gradient that decreases with depth.
2. **Quenching**: The component is rapidly cooled in a medium (e.g., oil,
   water, polymer). The cooling rate dictates the phase transformation: rapid
   cooling transforms the carbon-enriched austenite into hard martensite,
   while slower cooling permits the formation of softer diffusional phases
   (bainite, pearlite).
3. **Metrology**: The hardened component is sectioned and measured to produce
   a hardness traverse (hardness versus depth).

The primary engineering output is the **Effective Case Depth (ECD)**: the
depth at which hardness falls below 550 HV, per ISO 2639.

Traditional cycle development is empirical and resource-intensive, requiring
iterative furnace runs, metallurgical sectioning, and measurement. Ferrumizer
addresses two intractable inverse problems inherent to this workflow:

- **Calibration**: Inferring effective furnace operating parameters from a
  measured hardness traverse, complete with Bayesian posterior uncertainty.
- **Inverse Design**: Optimizing a furnace schedule to achieve a target ECD,
  optionally subject to an energy consumption penalty.

By modeling the entire pipeline as a fully differentiable function, Ferrumizer
transforms these empirical challenges into well-posed numerical optimization
and inference problems.

## 2. Glossary of Terms

| Term | Definition |
| :--- | :--- |
| **AISI 8620 / 9310 / 5120** | Standard steel grades for carburized components, differing in alloy content (e.g., Ni, Cr, Mo) and resulting hardenability. |
| **Effective Case Depth (ECD)** | The depth below the surface at which hardness drops below 550 HV (ISO 2639). |
| **Hardness Traverse** | A profile of hardness measurements taken at incremental depths from the treated surface to the core. |
| **Carbon Potential ($C_{pot}$)** | The equilibrium surface carbon concentration imposed by the furnace atmosphere. |
| **$D_0$, $Q$** | Arrhenius parameters governing the temperature dependence of carbon diffusion ($D(T) = D_0 \exp(-Q/RT)$). |
| **Mass-transfer coefficient ($h_m$)** | Governs the rate of carbon transfer from the furnace atmosphere to the steel surface. |
| **Martensite / Bainite / Pearlite** | Steel microstructural phases formed during cooling, ordered here from hardest/fastest-forming to softest/slowest-forming. |
| **JMAK (Scheil Additivity)** | The Johnson-Mehl-Avrami-Kolmogorov model, adapted via Scheil's rule, to estimate diffusional phase fractions under continuous, non-isothermal cooling. |
| **Tesseract** | A software component defined by a machine-readable I/O schema and an `apply` contract, executable either in-process or as an isolated container. |
| **Autodiff (AD)** | Automatic differentiation: the algorithmic evaluation of exact derivatives via reverse-mode accumulation (e.g., JAX). |
| **Finite Differences (FD)** | Numerical approximation of derivatives via input perturbation and re-evaluation. |
| **NUTS** | No-U-Turn Sampler, a Markov Chain Monte Carlo (MCMC) algorithm used here via NumPyro for Bayesian inference. |
| **SBC (Simulation-Based Calibration)** | A statistical validation method ensuring that Bayesian credible intervals achieve their nominal coverage frequency. |

## 3. The Tesseract Paradigm and System Boundaries

A Tesseract component enforces a strict contract: it declares its required
inputs and produced outputs via a machine-readable schema and exposes an
`apply` function. Crucially, it can be executed either as an in-process
library or as an isolated container.

The primary value proposition of the Tesseract framework is **gradient
survival across heterogeneous boundaries**. In conventional engineering
software, pipelines are fragmented across black-box solvers (e.g., legacy
Fortran, commercial FEA), breaking the chain of computation and preventing
end-to-end differentiation. Tesseract enables a composition of such
heterogeneous components to behave as a single differentiable function,
allowing an optimizer at the output to propagate gradients through process,
language, and differentiation boundaries back to the input parameters.

## 4. Component Architecture and Heterogeneous Differentiation

The Ferrumizer pipeline is composed of three distinct Tesseract components,
each with its own schema, container configuration, and dependency
environment:

1. **`thermal-stage`**: Computes the 1D temperature field $T(x,t)$ via
   explicit FTCS conduction with convective/radiative Robin boundary
   conditions. Implemented in **JAX** (native AD).
2. **`carburizing-stage`**: Computes the carbon concentration profile
   $C(x,t)$ via Fickian diffusion with Arrhenius temperature dependence.
3. **`hardening-stage`**: Computes phase fractions, hardness $H(x)$, and ECD
   using Andrews $M_s$, Koistinen–Marburger kinetics, and JMAK/Scheil
   additivity. Implemented in **JAX** (native AD).

### 4.1 The Deliberate FD/AD Boundary

The `carburizing-stage` is intentionally designed to satisfy the heterogeneous
differentiation criterion. It provides two numerically equivalent
implementations:

1. **Legacy-style NumPy FD Box**: An explicit time-stepper in plain NumPy. Its
   parameter Jacobian (w.r.t. $D_0, Q, C_{pot}, h_m$) is computed via central
   finite differences. This accurately models the constraint of inheriting a
   legacy solver that cannot be restructured for AD.
2. **JAX Twin**: The identical physics expressed in JAX, providing exact
   reverse-mode AD gradients.

The composed pipeline utilizes the NumPy box for forward computation and the
JAX twin for gradient propagation. This ensures that exact AD gradients flow
through a component whose native differentiation strategy is finite
difference, while simultaneously crossing container boundaries via
`tesseract_jax.apply_tesseract`.

### 4.2 Verification of the Boundary

The integrity of this boundary is not asserted; it is rigorously tested:

- **V4 (Cross-AD Gate)**: Verifies that gradients from the NumPy FD box and
  the JAX twin agree to a relative tolerance of $< 10^{-3}$ (empirically
  $\sim 6 \times 10^{-10}$).
- **V4c (Container Gate)**: Verifies that the end-to-end gradient through the
  fully containerized three-stage composition remains finite, non-zero, and
  within 20% of the FD reference.
- **V5 (Runtime Checks)**: Validates AD stages against numerical perturbation
  at runtime with zero failures.

## 5. Architectural Necessity of Tesseract

The Tesseract framework is load-bearing, not merely decorative. Alternative
approaches were evaluated and rejected:

- **Monolithic JAX Rewrite**: Erases the heterogeneous boundary entirely,
  failing to demonstrate the framework's core value proposition of composing
  disparate solvers.
- **Commercial Solver Wrapper**: Introduces licensing restrictions, opacity,
  and hardware dependencies incompatible with a reproducible, open-source
  hackathon submission.
- **Informal Python Library**: Lacks schema enforcement, container isolation,
  and verifiable gradient propagation across process boundaries.

Tesseract enables the `carburizing-stage` to remain a black box to the rest of
the pipeline while still contributing to a globally differentiable system, all
while maintaining strict dependency isolation between the legacy-style solver
and the modern inference stack.

## 6. Applied Workflows

The differentiable pipeline enables two primary, reproducible workflows:

### 6.1 Calibration (Bayesian Inference)

Given a measured hardness traverse, the NUTS sampler infers the posterior
distribution over process parameters $\{ \log D_0, Q, C_{pot}, h_m,
\epsilon \}$.

- **Identifiability**: Parameters $D_0$, $Q$, and $C_{pot}$ are strongly
  identifiable (recovered to $< 5 \times 10^{-3}$ relative error using two
  distinct schedules, per V6). The mass-transfer coefficient $h_m$ is weakly
  identifiable from end-state hardness alone, resulting in a broader posterior,
  which is explicitly documented.
- **Rigor**: Convergence is enforced via hard gates on $\hat{R}$ (split-Rhat)
  and Effective Sample Size (ESS). Non-convergent chains are rejected,
  preventing silent reporting of invalid inferences.

### 6.2 Inverse Design (Gradient Optimization)

Given a target ECD, a gradient-based optimizer adjusts the furnace schedule
parameters (soak temperature, time, carbon potential) to minimize the error
against the target. When invoked with `--penalty energy`, the optimizer
generates a Pareto front (Figure F9), quantifying the trade-off between
achieved case depth and relative energy consumption.

## 7. Verification and Reproducibility Matrix

Every quantitative claim in the repository is tied to an executable
verification gate.

| Claim | Verification Command | Acceptance Criterion |
| :--- | :--- | :--- |
| **Numerical Correctness** | `ferrumize verify --fast` | V1: Surrogate error $\le 0.5\%$; V2: Diffusion $L_2$ error $< 10^{-3}$; V3: Convergence order $\ge 1.85$. |
| **FD/AD Boundary Integrity** | `ferrumize verify --fast` (V4, F5) | Relative $\infty$-norm between FD and AD gradients $< 10^{-3}$. |
| **Container Gradient Propagation** | `ferrumize verify --fast` (V4c) | End-to-end gradient through containers is finite, non-zero, and within 20% of FD reference. |
| **Parameter Recovery** | `ferrumize verify` (V6) | Strong parameters recovered to $< 5 \times 10^{-3}$; $h_m$ within factor of 2. |
| **Inference Calibration** | `ferrumize verify` (V7) | SBC rank-uniformity $p > 0.05$; 90% coverage within binomial bounds. |
| **Metallurgical Validity** | `ferrumize verify` (V8, Q1-Q3) | Jominy reconstruction MAE $\le 2.6$ HRC; quench model correctly ranks medium severity and exhibits differentiability. |

## 8. Scope and Limitations

Ferrumizer is intentionally scoped as a process emulator, not a commercial
finite-element or CFD package. Limitations are explicitly documented:

1. **Geometry**: The thermal and diffusion models are strictly 1D (slab or
   axisymmetric radial). Part size is represented by a characteristic length;
   3D geometry, meshing, and stress/distortion are out of scope.
2. **Validation Data**: The reference hardness traverse is a synthetic
   reconstruction anchored to published parameter ranges for AISI 8620H. While
   the ingestion and calibration machinery supports real industrial data,
   current validation is against synthetic ground truth.
3. **Phase Kinetics**: JMAK C-curve parameters are representative,
   order-of-magnitude values for low-alloy steels, not certified TTT data for
   specific heat numbers. The quench model is qualitative to semi-quantitative,
   suitable for trend analysis and failure-mode prediction, not certified phase
   fraction reporting.
4. **Inference Coverage (V7)**: While the SBC rank-uniformity test passes
   ($\chi^2 p = 0.074$) and sampler health is confirmed ($\hat{R} \le 1.12$),
   the measured 90% credible-interval coverage is 0.83. This indicates mild
   under-coverage (slightly overconfident intervals), though point estimates
   remain robustly validated by independent gates (V8).
5. **Energy Proxy**: The energy penalty is a relative metric based on the time
   integral of setpoint temperature above ambient, not an absolute metered
   kWh value.

## 9. Architectural Decisions and Rationale

Key non-obvious design decisions, aligned with the project's Architecture
Decision Records (ADRs):

- **Track Alignment**: Targeted at Track 04 (Differentiable Inference & UQ)
  with a cross-track contribution to Track 02 (Multi-Physics), as the
  pipeline inherently couples heat transfer, mass diffusion, and phase
  kinetics.
- **Three-Stage Separation**: Splitting the pipeline at thermal, carbon, and
  hardening boundaries prevents burying distinct PDEs in a single component
  (which would obscure the FD/AD boundary) while maximizing meaningful
  container isolation.
- **Carburizing as the FD Box**: Carbon diffusion is the most plausible
  candidate for a "legacy" solver in real-world contexts, and its parameters
  ($D_0, Q, C_{pot}, h_m$) are the primary targets of calibration, placing the
  differentiation boundary exactly where gradients are most valuable.
- **Lumped-Capacitance Surrogate**: For calibration, a validated
  lumped-capacitance model (V1 error $\le 0.5\%$) replaces the full thermal
  solve to accommodate the thousands of evaluations required by MCMC, without
  sacrificing accuracy.
- **Finite-Rate Quench Model**: Replaces the standard instantaneous-quench
  assumption. By modeling the cooling curve and integrating Scheil-JMAK over
  time, the system can accurately simulate the classic failure mode of
  insufficient quench severity (e.g., air quench yielding 100% pearlite and 0
  mm ECD).
- **Determinism**: All stochastic commands accept a `--seed` argument.
  Deterministic figures regenerate byte-identically under pinned dependencies;
  stochastic workflows (NUTS) reproduce distributionally.

## 10. Related Documentation

- **Physics Model**: `docs/physics.md` (Derivations, constants, and
  provenance).
- **Component Contracts**: `components/*/tesseract_api.py` and
  `docs/architecture.md`.
- **Reproducibility Guide**: `docs/reproducing.md` (Commands,
  runtime/memory metrics, and artifact generation).
- **Verification Definitions**: `docs/verification.md` (Detailed gate
  criteria).
- **Bayesian Workflow**: `docs/calibration.md` (Inference setup and runtime
  expectations).
- **Design Rationale**: `docs/design-rationale.md` (This document).
