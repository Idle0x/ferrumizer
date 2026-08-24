# ADR-001: Carbon diffusion prefactor D0 for austenite

Status: ACCEPTED
Date: 2026-08-22
Supersedes: the literal D0 prior range used in the original calibration plan

## Context

The original calibration plan specified the carbon-in-austenite diffusion
prefactor prior as
`D0 in [1e-12, 1e-9] m^2/s (lognormal)` alongside an activation energy
`Q in [100, 200] kJ/mol`.

Pairing any D0 in that range with any Q in [100, 200] kJ/mol at a typical
carburizing temperature (~950 C = 1223 K) yields an effectively zero
diffusivity. For example, the most generous corner (D0 = 1e-9, Q = 100 kJ/mol):

    D = 1e-9 * exp(-100000 / (8.314 * 1223)) = 5.4e-14 m^2/s

Real carbon diffusivity in austenite at 950 C is on the order of
1e-11 to 1e-10 m^2/s. A D of 5e-14 m^2/s produces a diffusion length
sqrt(2 D t) of only ~2 um over a full 4-hour soak — no case would form, the
erfc overlay (V2) would fail outright, and no literature anchor (V8) could be
matched. The [S3] D0 range is therefore physically inconsistent with the Q
range and with the product's own verification gates.

The classical value for carbon in gamma-iron is D0 ~= 0.20-0.25 cm^2/s
(= 2.0-2.5e-5 m^2/s) with Q ~= 135-148 kJ/mol, which reproduces measured
austenite diffusivities and realistic case depths.

## Decision

Use the physically consistent Arrhenius pair throughout:

- Preset (point) values: D0 = 2.2e-5 m^2/s, Q = 137 kJ/mol (mid literature).
- Calibration prior on D0: lognormal centered on 2.2e-5 m^2/s
  (sigma ~ 0.5 in log-space), truncated to a physically plausible band.
- Calibration prior on Q: keep the original range, Uniform(100, 200) kJ/mol.

The Q prior is retained unchanged; only the D0 prior is corrected.

## Alternatives considered

1. Keep D0 in [1e-12, 1e-9] as written. Rejected: produces near-zero
   diffusion, breaks V2/V6/V8, and contradicts the verification-first
   principles (no fabricated/unsupported numbers).
2. Redefine Q in different units to make the small D0 work. Rejected: the
   schema fixes the unit as kJ/mol; changing units silently would be a hidden
   tuning.

## Consequences

- All preset YAMLs and the calibration prior use D0 ~ 2.2e-5 m^2/s.
- The deviation is recorded here as a decision record, keeping the change
  explicit rather than silent drift.
