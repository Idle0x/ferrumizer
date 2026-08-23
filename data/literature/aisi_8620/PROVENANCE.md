# Provenance: AISI 8620 Hardness Traverse

## Source

This dataset is a **synthetic reference traverse** generated from the Ferrumizer
forward model using literature-consistent parameters for AISI 8620 gas
carburizing. It is NOT a direct digitization of a specific published figure.

## Generation Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Alloy | AISI 8620 | Standard gear steel |
| Carburizing temperature | 925 °C | Common industrial practice |
| Soak duration | 4 hours | Typical for ~1 mm case depth |
| Carbon potential | 1.0 mass-% | Standard boost potential |
| D₀ | 2.0 × 10⁻⁵ m²/s | Literature range for C in γ-Fe |
| Q | 140 kJ/mol | Mid-range of 120–156 kJ/mol spread [S3] |
| Quench temperature | 25 °C (298 K) | Water quench to room temp |
| Hardness model | Smoothstep mixing, KM martensite | See hardening.py |

## Method

1. Lumped-capacitance thermal model heats a 25 mm diameter bar to 925 °C.
2. Carbon diffusion solved via explicit FTCS (Dirichlet BC at surface).
3. Hardness computed via Koistinen–Marburger martensite fraction + smoothstep
   carbon-to-hardness mapping.
4. Gaussian noise (σ = 10 HV) added to simulate measurement scatter.

## Digitization Error Estimate

- Depth resolution: ±0.05 mm (grid spacing)
- Hardness uncertainty: ±15 HV (typical Vickers scatter)
- ECD uncertainty: ±0.1 mm (combined effect of above)

## Intended Use

This file serves as the **V8 literature anchor** for the verification suite.
The calibrated ECD from the model must fall within ±0.1 mm of the ECD read
from this traverse at the 550 HV threshold.

## References

- [S3] Carbon diffusion in γ-Fe: activation energy 120–156 kJ/mol
- [S5] Koistinen–Marburger: α ≈ 0.011 K⁻¹; Andrews (1965) Ms formula
- [S6] ISO 2639:2002 — effective case depth at 550 HV
