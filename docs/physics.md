# Physics

The physics is deliberately small, explicit, and documented — every constant
has provenance in the alloy YAMLs or an ADR. Full derivations live in the
module docstrings (`components/shared/ferrumizer_physics/`); this page gives
the model inventory and the honest assumptions.

## Stage 1 — Thermal (`thermal.py`)

- **Full path**: 1-D explicit FTCS conduction with a nonlinear
  convective/radiative Robin boundary condition, stability-enforced time step.
- **Surrogate path** (used inside calibration/design for tractability):
  lumped-capacitance surface temperature with schedule-knot interpolation —
  V1-validated to < 0.5% max relative error against the full path
  (see [ADR-002](adr/ADR-002-calibration-surrogate.md)).
- Assumption: 1-D slab or axisymmetric radial; no CAD, stress, or distortion.
  Part size enters as a characteristic half-thickness.

## Stage 2 — Carbon diffusion (`carbon.py`)

Fick's second law with Arrhenius diffusivity:

```text
D(T) = D0 · exp(−Q / (R T))
```

- Boundary: Dirichlet (fixed atmosphere carbon potential) or mass-transfer
  (finite h_m) — the config's `carbon_mode` selects.
- Assumption: 1-D; carbon in austenite; no grain-boundary or
  geometry-dependent effects. ADR-001 documents why D0 = 2.2e-5 m²/s and
  Q = 137 kJ/mol are the shipped defaults (the D0 range in the original plan
  was physically inconsistent with real gamma-iron diffusivity).

## Stage 3 — Hardening (`hardening.py`)

- **Andrews (1965) martensite-start**: `Ms(C) = A − b_C·C` (K, C in mass-%).
- **Koistinen–Marburger martensite**: `f_M = 1 − exp(−α_KM·max(Ms − Tq, 0))`.
- **Smoothstep hardness mixing**: `H = Hcore + (Hmax − Hcore)·smoothstep((C − Cmin)/(Cideal − Cmin))`
  — never a hard clamp, so the model stays C¹-differentiable.
- **ECD**: first inward depth where H crosses 550 HV (ISO 2639 practice),
  computed via clamped linear interpolation per segment — fully
  differentiable.

## Stage 3b — Finite-rate quench model (new in 0.1.1)

The legacy path assumed an **instantaneous** quench to room temperature —
which physically cannot form bainite/pearlite and therefore over-predicts
case depth. The new model:

1. **Newton cooling curve**:
   `T(t) = Tq + (T0 − Tq)·exp(−t/τ)`, with
   `τ = ρ·cp·L / (h·(1 + agitation))` where `L` is the surface-to-center
   distance and `h` is the film coefficient for the quench medium:

   | Medium | Film coefficient h (W/m²·K) |
   |---|---|
   | air | ~50 |
   | oil | ~900 |
   | polymer | ~1800 |
   | water | ~3500 |

   Bath temperature, agitation, and part size all enter explicitly.

2. **Scheil-additivity JMAK** over the cooling curve for the two diffusional
   phases (pearlite ~600 °C nose, bainite ~450 °C nose), consuming austenite:
   `X_diff = X_pearlite + (1 − X_pearlite)·X_bainite`.

3. **Martensite on the survivor**: `f_M = (1 − X_diff)·KM(Ms, T_quench)`,
   then hardness is the martensite-weighted mixing rule.

Consequences that are now visible (and verified by gates Q1–Q3):

- Air quench → ~100% pearlite, surface hardness collapses to core level,
  ECD → 0. This is the real production failure mode the instant-quench model
  could never predict.
- Oil (low agitation) → partially pearlitic, surface hardness lands just
  under spec for typical geometries.
- Water (agitated) → ~96% martensite, full case.

**Honest limits**: the JMAK rate constants are order-of-magnitude C-curve
magnitudes representative of low-alloy carburizing steels — qualitative to
semiquantitative (failure modes and rankings), not certified TTT data. The
cooling curve is a single characteristic curve per part, not a depth-resolved
FEM quench.

## Dynamic alloy chemistry (`alloys.py`)

`composition_to_preset()` builds a full preset from bare wt-% composition:

- **Ms** via Andrews' multi-element line (C, Mn, Cr, Ni, Mo, Si terms).
- **Hardness plateau** anchored at the carburized-case composition (~0.9% C)
  with a capped hardenability bump for alloying.
- **Diffusion defaults** to the gamma-iron pair (D0, Q) — carbon diffusion in
  austenite is weakly alloy-dependent at carburizing temperatures.
- **Thermal defaults** to generic low-alloy steel.

These are documented estimates, not certified constants — see the README
limitations.

## Units and conventions

- Temperatures: K internally; °C at user boundaries (CLI/app).
- Depth: mm (case depth and traverse depths), m internally for transport.
- Hardness: Vickers HV.
- ECD threshold: 550 HV (ISO 2639 practice).
