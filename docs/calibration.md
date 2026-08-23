# Calibration

NUTS calibrates `{log D0, Q, C_pot, h_m, eps}` and reports R-hat and bulk ESS. Use two schedules to break D0-Q collinearity. Calibration uses the V1-validated lumped thermal surrogate for tractability; resolved thermal runs remain available for forward simulation and verification.

## Runtime expectations (CPU)

Each likelihood evaluation runs the carbon FD + hardening chain. At the shipped
`data/synthetic/calibration_data.yaml` grid (n=81, dt=2 s, 2 h soak) a single
evaluation is ~1.5 s. NUTS with adaptive leapfrog costs several evaluations per
draw, so plan for:

- Smoke test (1 chain, 100 warmup, 100 draws): ~5–10 min
- Full default (4 chains, 1000/1000): hours on CPU; use `--chains 2 --draws 300`
  for a credible posterior in ~30–60 min, or reduce the carbon grid in the
  config (`carbon.n`, `carbon.dt`) — the prior corners are kept stable by
  adaptive sub-stepping in the JAX twin.

The convergence gates (R-hat < 1.01, bulk ESS > 400) are enforced before any
result is released; a run that fails them exits non-zero and writes no posterior.
