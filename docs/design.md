# Design

The design loop is the inverse of simulation: instead of "given a schedule,
what ECD results?", it answers **"given a target ECD, what schedule produces
it?"** — by gradient descent on the fully differentiable forward model.

## How it works

The optimizer treats the schedule's interior temperature knots as free
variables (the first and last knots stay anchored so the schedule remains
valid) and minimizes

```
loss = (ECD(schedule) − target)²   [+ w · energy_proxy if --penalty energy]
```

with `jax.grad` driving projected gradient descent. The gradient of ECD with
respect to a temperature knot flows back through all three stages — thermal
history → carbon diffusion → hardening → ISO 2639 crossing — which is exactly
the end-to-end differentiability the project exists to demonstrate.

Two implementation notes, both documented in
[`app/design/optimize.py`](../app/design/optimize.py):

1. **Feasibility pre-check.** Before grinding 120 gradient steps, the loop
   computes the ECD range actually reachable within the temperature bounds.
   If the target is outside `[ECD(T_min), ECD(T_max)]` it says so immediately
   and reports the shortfall instead of silently converging to the nearest
   bound.
2. **Normalized gradient steps.** Raw ∂ECD/∂T is physically tiny
   (~1e-3 mm/K), so plain gradient descent crawls. Normalizing the step turns
   the learning rate into a temperature step per iteration (units of K) —
   the direction still comes from the gradient, and convergence happens in
   tens of steps. Verified: `ferrumize design 0.15 --alloy 8620` converges to
   ECD = 0.1500 mm (loss 3.5e-10) at step 108 and writes a valid
   `design.json`.

## Usage

```bash
ferrumize design 0.15 --alloy 8620                      # plain target
ferrumize design 0.15 --alloy 8620 --penalty energy     # Pareto front
ferrumize design 0.15 --config my_furnace.yaml          # from a config
```

- The target is in **mm of effective case depth at 550 HV**.
- `--penalty energy` sweeps penalty weights and writes the **Pareto front**
  (figure F9): how much case depth must be given up to save gas. The energy
  proxy is a relative penalty axis (time-integral of setpoint above ambient),
  not an absolute energy figure — see Limitations in the
  [README](../README.md#limitations).
- Unreachable targets produce an **INFEASIBLE** verdict with the reachable
  range and a hint (increase soak time, part size, or carbon potential).

## Output

`results/design/design.json` contains the optimized schedule knots, achieved
ECD, reachable range, feasibility, loss/ECD traces (JSON-safe lists), and the
energy proxy.

## Relationship to calibration

Design trusts the parameters it is given. If those parameters are textbook
defaults, the designed schedule is a textbook answer; if they come from
[calibration](calibration.md) against your furnace's traverses, the designed
schedule is yours. The natural production loop is:

```
measure → calibrate → design → run → measure (tighten posterior) → …
```
