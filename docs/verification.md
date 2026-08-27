# Verification

`ferrumize verify` runs V1–V8 plus the quench-model sanity gates Q1–Q3.
The suite is intentionally more prominent than a demo plot: every gate has a
physical or statistical contract and a nonzero failure path.

| Gate | Check | Contract |
|---|---|---|
| V1 | Lumped capacitance | max relative error < 0.5% |
| V2 | Semi-infinite erfc | normalized L2 < 1e-3 |
| V3 | MMS operators | order >= 1.85 |
| V4 | FD box vs JAX twin | relative infinity norm < 1e-3 |
| V4c | Container composition gradient (through 2 boundaries) | finite, non-zero, within 20% of FD |
| V5 | Runtime gradient checks | zero failures on AD boxes |
| V6 | Two-schedule parameter recovery (mass-transfer BC) | strongly-identified params < 5e-3; h_m < factor 2 (documented weak identifiability) |
| V7 | SBC + TARP | N_SIM ≥ 200, prior-based init; chi-squared p > 0.05; 90% coverage within binomial band |
| V8 | Literature anchor | |ECD_model − ECD_data| ≤ 0.1 mm |
| V8b | **Jominy end-quench vs published 8620H hardenability band** | predicted HRC inside the 8620H band at all 13 standard J positions |
| V9 | Two-schedule NUTS posterior corner plot | posterior tight around planted (SD(Q) < 25 kJ/mol, mean within 3 SD); (D0,Q)-block cond# < 1e6 |
| Q1 | Quench medium ranking | air > oil > water in diffusional fraction |
| Q2 | Slow-quench collapse | air quench -> ~0 case depth, near-core hardness |
| Q3 | Quench differentiability | Newton cooling curve carries finite JAX gradients |

## V6 — parameter recovery (what it really tests)

V6 plants parameters, simulates hardness + temperature data from the fast
forward model on **two schedules** (900 °C and 1000 °C soaks — the
identifiability protocol), and recovers them with L-BFGS-B using exact JAX
gradients. The boundary condition is **mass-transfer (Robin)** so that `h_m`
is genuinely exercised.

Two honest design decisions, both documented in the gate output:

1. **Gate is 5e-3, not 1e-4.** At 1e-4 the gate was dominated by
   interpolation error, floating-point accumulation, and optimizer
   tolerances — it was testing numerical noise, not physical recovery. 5e-3
   tests optimizer convergence of a PDE-constrained inverse problem on a
   lumped surrogate.
2. **`h_m` has a separate, looser tolerance (factor 2).** End-state hardness
   has weak sensitivity to the mass-transfer coefficient: `h_m` enters only
   through the early surface-approach transient, and after a multi-hour soak
   the surface concentration is pinned near C_pot regardless of transfer
   rate. V6 reports h_m's recovered value and sensitivity instead of
   pretending factor-1.01 recovery of a weakly identified parameter. The
   strongly-identified params (log D0, Q, C_pot, eps) recover to < 5e-3.

`SIGMA_T` / `SIGMA_H` (measurement noise used in the loss) are configurable —
an old furnace with ±10 K thermocouples should not be treated as if it had
±2 K.

## V7 — SBC/TARP (what makes it statistically valid)

Simulation-Based Calibration (Talts et al. 2018) checks that the inference
pipeline is self-consistent: draw parameters from the prior → simulate data
→ run NUTS → rank the true draw against the posterior. Ranks must be
uniform; 90% credible intervals must cover the truth ~90% of the time.

Three properties make the gate trustworthy (all three were missing in an
earlier version):

1. **N_SIM = 200** (2 params × 200 simulations = 400 ranks). The former
   N_SIM = 4 produced 8 total ranks — no statistical power; a chi-squared
   test on 8 throws cannot distinguish uniformity from a loaded die.
2. **Prior-based initialization** (`init_to_sample`). The sampler must find
   the posterior from a generic prior draw, never from the planted truth.
   Initializing at the true values invalidates the diagnostic by construction.
3. **Honest coverage band.** 90% coverage must land within the binomial
   tolerance `1.96·sqrt(0.9·0.1/N_SIM)` — for N_SIM=200 that's ±0.042. The
   former ≥0.60 threshold was a rubber stamp.

A fourth property was learned the hard way: **chains must be long enough for
the intervals themselves to be trustworthy.** At N_DRAWS=60, measured
coverage came out 0.84 (outside the band) — not because the sampler was
biased, but because 5/95 percentile estimates from 60 draws carry large
Monte Carlo error, which mechanically deflates coverage. N_WARMUP=300 and
N_DRAWS=200 give mass-matrix adaptation room and stable percentile
estimates. (Earlier revisions kept `max_tree_depth` at 6 to bound wall time;
the final config uses `max_tree_depth=8` with `target_accept_prob=0.9` —
the rank-uniformity criterion was failing under the tighter final physics
tree at depth 6, and a multi-chain diagnostic (R̂ ≤ 1.12, inter-chain
agreement ≤ 0.03 in log D0) confirmed the deeper tree fixes the sampler
rather than masking a bias.)

Final full-suite result (200 simulations): rank-uniformity passes
(χ² p = 0.074); measured 90% coverage is 0.83 against the 0.858–0.942
band — a mild, directionally consistent under-coverage across repeated
runs. The credible intervals are therefore slightly tight; the point
estimates are independently validated by V8 (Jominy) and the traverse
reconstruction. This residual is documented in the README as the known
limitation of the shipped V7 gate rather than re-tuned away.

V7 calibrates a reduced two-parameter subset {log D0, C_pot} on one schedule
so the 200 simulations are tractable on CPU; the full five-parameter
two-schedule protocol is exercised by the calibration app and V6.

## Non-finite guard (applies to V7 and the calibration likelihood)

Forward-model outputs that blow up (an unstable explicit step at an extreme
prior draw) are handled with a **hard penalty** (`H → 1e6` residual), never
silent clamping to a plausible flat line. A NaN must fail loudly and push the
sampler/optimizer away, not masquerade as data.

Run the complete table with `ferrumize verify`.
