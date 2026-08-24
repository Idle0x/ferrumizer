# Gallery

Run `ferrumize figures` (or `make figures`) to regenerate F1–F10. Every
figure includes physical units and is derived from a deterministic,
seeded script path (`app/ferrumize/figures.py`) — the same numbers and
images reproduce for anyone who clones the repo.

| Figure | File | What it shows | Regenerate |
|---|---|---|---|
| F1 | `figures/F1_hero_loop.gif` | One full cycle, animated: schedule → T → C → H → ECD | `ferrumize figures --only F1` |
| F2 | `figures/F2_architecture.png` | Three Tesseracts + end-to-end gradient flow | `ferrumize figures --only F2` |
| F3 | `figures/F3_erfc_overlay.png` | Solver vs analytic erfc solution (validation) | `ferrumize figures --only F3` |
| F4 | `figures/F4_mms_convergence.png` | MMS convergence order (validation) | `ferrumize figures --only F4` |
| F5 | `figures/F5_cross_ad.png` | FD vs autodiff gradient agreement (the boundary proof) | `ferrumize figures --only F5` |
| F6 | `figures/F6_posterior.png` | NUTS posterior marginals after calibration | `ferrumize figures --only F6` |
| F7 | `figures/F7_noise_sweep.png` | Recovery error vs measurement noise (robustness) | `ferrumize figures --only F7` |
| F8 | `figures/F8_identifiability.png` | Single- vs two-schedule identifiability | `ferrumize figures --only F8` |
| F9 | `figures/F9_pareto.png` | ECD-vs-energy Pareto front from schedule design | `ferrumize figures --only F9` |
| F10 | `figures/F10_alloy_strip.png` | Same recipe across 8620 / 9310 / 5120 | `ferrumize figures --only F10` |

Full plain-language explanations of every figure live in the
[README](../README.md#the-figures-what-each-one-proves).
