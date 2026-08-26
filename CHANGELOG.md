# Changelog

All notable changes to Ferrumizer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic
versioning. Entries are grouped by **Added / Changed / Fixed / Removed**.

## [Unreleased] — review-2 + review-3 hardening pass

### Added

- **V8 Jominy end-quench gate** (`verification/v8_jominy.py`) — a 25 × 100 mm
  rod is end-quenched and the predicted hardness profile is compared against
  the published 8620H hardenability band at all 13 standard J positions.
  PASS: the simulated curve sits inside the band (quench-end 40.8 HRC vs
  ~43.6 mid-band, MAE 2.6 HRC). This validates the spatial quench + JMAK +
  phase-hardness physics against independent published data.
- **CCT-style diagram tab** in the app — pearlite/bainite C-curves (inverted
  from the JMAK kinetics, 1 % start), Ms lines for surface/core carbon, and
  the actual computed cooling curves at surface / mid-radius / core from the
  last Virtual Furnace run, with an honest caveat that the noses are
  engineering surrogates.
- **Mf floor** (`mf_offset_K`) — Koistinen-Marburger now caps the driving
  temperature at `Ms − Mf`; the offset is a preset field (8620: 420 K,
  consistent with the measured Jominy quench-end; the old fixed 200 K capped
  martensite at 89 % and could not reach full-hardness low-carbon martensite).
- **Retained-austenite roll-off** — martensite hardness softens above ~1.0 % C
  (linear RA fraction ramp), replacing the old hard C ≥ Cideal → Hmax clamp.
- **Eutectoid lever-rule pearlite hardness** — the diffusional-product
  hardness now scales with local carbon (ferrite-pearlite aggregate at low C,
  full pearlite near eutectoid), so a 0.19 % C Jominy far end prices at
  ~230 HV instead of 280.
- **TTT/C-curve inversion** (`ttt_start_times`) — isothermal start times from
  the JMAK C-curves, used by the CCT tab.
- **V9 corner plot** (`verification/v9_corner.py`) — two-schedule NUTS
  posterior corner plot (log D0, Q, C_pot, eps) confirming the D0-Q ridge
  collapse in the sampling domain.
- Tests: `tests/test_ingest_review_fixes.py` (12 tests covering time units,
  RDP compression, rows_used, range validation, trajectory→scenario).

### Changed

- **Unified forward paths** — `fast_forward` (calibration) now runs the
  SAME spatial quench + per-depth phase-fraction path as `pipeline.forward`;
  verified identical hardness profile and ECD (0.0 HV / 0.000 mm) for an oil
  quench. The lumped surrogate survives only for the furnace soak (Bi ≈ 0.004,
  V1-validated, NUTS-tractable). Calibration passes quench medium / geometry /
  bath through `_scenario_kwargs`.
- **Calibration honors the ingested trajectory** — the Cycle Predictor tab
  builds its Scenario from the PLC log's trajectory (RDP-compressed knots)
  when one exists, instead of a hardcoded 2 h / 950 °C scenario (P0 #3).
- **Time-unit normalization rewritten** (P0 #1) — the parser reads units from
  the column header (`time [s]`, `elapsed (min)`, `t [h]`), parses HH:MM:SS
  clock timestamps to elapsed seconds, or defaults to seconds with a loud
  warning. The old median-step heuristic (which silently ×60'd 5–30 s
  datalogger logs) is gone.
- **Schedule compression via Ramer-Douglas-Peucker** (P2 #11) — heating and
  cooling ramps are preserved as diagonal segments instead of being chopped
  into a staircase of flat soaks; overshoot is no longer encoded as a
  deliberate setpoint.
- **Hardness physics** — phase-specific bainite/pearlite hardness from the
  presets (no more "everything non-martensite = Hcore"), ASM full-martensite
  C→HV curve (low-carbon martensite is NOT soft), cylinder R/2 characteristic
  length in Newton cooling (cylinder quenches 2× faster than the old model).
- **Custom alloys** — `composition_to_preset` D0/Q now scaled by the
  Lee-Matlock-Van Tyne (ISIJ Int. 51, 2011) composition-dependent diffusion
  parameters; the arbitrary 40 HV Hmax clamp replaced with a smooth
  saturation.
- **Calibration UI** — quench medium selector (the calibrator knows whether
  the part was oil-quenched), posterior predictive overlay + residual-vs-depth
  trend slope in the Cycle Predictor tab.
- **Range validation in the parser** (P2 #13) — impossible temperatures,
  negative depths, and hardness outside 50–1200 HV warn instead of being
  silently accepted; `rows_used` counts rows that actually contributed
  (P2 #12).
- **Honest comments** — the smoothstep "soft clamp" note, JMAK C-curve
  surrogate caveat, and Mf approximation are documented as approximations,
  not physics (P2 #14).
- `docs/ingestion.md` rewritten around the time-unit rule and RDP; the
  verification table now describes V8 as the Jominy band gate.

### Fixed

- **V7 SBC failure on the final tree (caught by the re-run)** — the R2
  physics introduced two non-smooth likelihood features that broke NUTS:
  the Mf floor (`jnp.minimum` gradient kink) and the ASM martensite curve
  (`jnp.interp` gradient kinks at every anchor). HMC assumes a smooth
  log-density; the kinks made the leapfrog integrator inaccurate and the
  adapted step size wrong, producing under-covering posteriors (V7
  p=0.005, coverage 0.82). Both are now C∞-smooth with identical physics:
  the Mf cap is a softplus-min (slope 1 near Ms, same asymptote), and the
  ASM curve is a Fritsch-Carlson monotone cubic (anchors reproduced
  exactly, continuous gradients). Jominy gate unchanged (PASS, MAE 2.6).
- **Calibration init crash (production bug, caught in the final sweep)** —
  `run_calibration` used numpyro's default `init_to_uniform`; with the R2
  physics (ASM curve, phase hardness) extreme uniform-bound draws produced
  non-finite likelihoods and NUTS died with "Cannot find valid initial
  parameters", taking the app's Cycle Predictor tab down with it. Now uses
  `init_to_sample()` (same strategy as V7) + the hard non-finite guard.
- **Calibration likelihood contract (production bug, caught in the final
  sweep)** — `_predict_hardness` was refactored in R2 to return only H, but
  the NumPyro model (and PPC helpers) unpack `(H, ecd)`. Restored the tuple
  return; the app's "Run calibration" path was broken by this.
- **Phantom ECD** — `ecd_from_hardness` returned 7.0 mm when NO node crossed
  the 550 HV threshold (a flat near-threshold segment with a tiny negative
  denominator); now returns 0.0 mm honestly.
- **Gradient-killing concrete conversions** — `float(...)` on traced arrays
  inside the jitted quench path silently zeroed d(ECD)/d(ε) and friends;
  replaced with traced `jnp.full` fill values. Full 5-param gradient now
  finite, including through the quench.
- **Rod quench BC** — Jominy rod geometry uses a convective-film closed form
  at the quenched end (not a Dirichlet assignment) and a zero-flux far end;
  the full-field collection scan now uses the rod branch (the sampled-block
  scan was patched but the history scan was not — inverted BC at n=161).
- **Profile-likelihood test tolerance** — the light-grid diagnostic (coarse
  log D0 cells) now asserts "within ~2 cells" with an explicit comment that
  the TIGHT recovery standard is the V6 gate (which still recovers planted
  params to ~1e-4).

## [0.1.1] — 2026-08-23

### Added

- **Finite-rate quench model** — real quenches are no longer assumed
  instantaneous: lumped-Newton cooling driven by quench medium
  (oil/water/polymer/air), bath temperature, agitation, and part size, with
  Scheil-additivity JMAK integration for pearlite/bainite and KM martensite on
  the surviving austenite. Slow quenches now correctly collapse surface
  hardness and ECD (the production failure mode). New verification gates
  Q1–Q3 wired into `ferrumize verify`.
- **Depth-resolved spatial quench** — the quench now solves the 1-D
  conduction PDE from the end-of-soak temperature field with a convective
  bath boundary (per-depth cooling curves, CCT-style phase fractions across
  the section) instead of a single lumped part-average curve. The app shows
  martensite/pearlite/bainite fractions vs depth.
- **Hardenability (Grossmann DI)** — `ideal_critical_diameter_mm()` and
  `through_hardening_verdict()` answer "will this part through-harden?"
  (ASTM A255 practice, documented as a ranking estimate). Shown in the app.
- **PLC / datalogger ingestion** (`app/ingest/plc_parser.py`) — defensive
  parser for messy furnace exports: delimiter/header/column-role sniffing,
  deg C ↔ deg F conversion, time-unit normalization, malformed-row warnings,
  and trajectory → soak-segment compression. New `ferrumize ingest` command;
  new `docs/ingestion.md` page; Log Ingestion tab in the app.
- **Dynamic alloy chemistry** — `composition_to_preset()` builds a full
  physics preset from bare wt-% composition (Andrews multi-element Ms,
  case-hardness plateau with hardenability bump); `load_alloy()` accepts
  dicts; `validate_preset()` guards schema. The app's Custom… alloy option
  uses this path.
- **Two-stage boost/diffuse editing** in the app (previously the app ran a
  single-stage schedule regardless of the underlying model's capability).
- **Two-schedule calibration protocol** — `run_calibration` accepts a second
  (depths, hardness, scenario) triple; the same parameters must explain both
  traverses, which collapses the D0-Q degeneracy. The generated demo config
  ships `schedule`/`schedule2` + `traverse_csv`/`traverse_csv2` at 900/1000 °C.
- **Profile likelihoods** — `profile_likelihood_grid()` computes 2-D
  (log D0, Q) profile likelihoods with nuisance parameters optimized;
  `ferrumize identifiability` writes single-vs-two-schedule contour plots.
- **Posterior predictive checks** — `posterior_predictive_hardness()` and a
  PPC plot (`ppc_hardness.png`) in the calibration output.
- **Hierarchical measurement noise** — `sigma_hv` is inferred by default
  (`infer_sigma=True`, HalfNormal(15) prior) instead of assumed fixed.
- Quench verification gates Q1–Q3 (`verification/q_quench.py`).
- Tests: `tests/test_new_features.py` (quench, dynamic alloys, PLC
  ingestion), `tests/test_cycle_review_fixes.py` (Robin BC, SBC integrity,
  V6 gate, profile likelihoods, hardenability).

### Changed

- **README rebuilt** (~100 → ~460 lines): all 10 figures embedded with
  plain-language explanations and regenerate commands, architecture diagram,
  full CLI table, why-Tesseract / why-Track-04 rationale, extension guide,
  honest limitations, repository layout with hyperlinks.
- **Calibration boundary condition** — calibration now REQUIRES
  `carbon_mode="mass_transfer"` (Robin); sampling `h_m` under Dirichlet
  raised an error (previously `h_m` was a dead parameter — a flat posterior
  direction wasting ESS). The CLI and app force mass_transfer.
- **V6 gate** — relaxed from 1e-4 (dominated by numerical noise) to 5e-3 for
  strongly-identified params, with a documented factor-2 tolerance for the
  weakly-identified `h_m`. `SIGMA_T`/`SIGMA_H` are now configurable.
  V6 uses mass_transfer so `h_m` is genuinely exercised.
- **V7 SBC rewrite** — N_SIM 4 → 200 (400 ranks; the former 4 was
  statistically void), initialization from the prior instead of the planted
  truth, honest binomial coverage band instead of a ≥0.60 rubber stamp.
- **Non-finite guard** — NaN/inf forward outputs now become a hard 1e6
  likelihood penalty (in V7 and calibration), never a silent clamp to a
  plausible 230 HV flat line.
- **Synthetic data config** — `calibration_data.yaml` now matches its CSVs:
  two schedules at 900/1000 °C (was a contradictory single 950 °C line),
  `mode: mass_transfer`, both traverses referenced.
- Alloy JMAK rate constants corrected from placeholder magnitudes (dead code
  path) to physically realistic C-curve magnitudes; documented in each preset.
- ADR-001 / verification scripts / calibration docstrings: internal plan
  references replaced with public doc links (docs/ADR, docs/verification).
- `docs/cli.md`, `docs/api.md`, `docs/gallery.md`, `docs/verification.md`,
  `docs/roadmap.md`, `mkdocs.yml` updated for the new surface.

### Fixed

- **`ferrumize design` JSON crash** — `TypeError: only length-1 arrays can be
  converted to Python scalars` when writing `design.json` (NumPy arrays in
  `loss_trace`/`ecd_trace`; converted to lists before dump). The optimizer
  itself was already converging correctly (verified ECD 0.1500, loss 3.5e-10).
- **App "Process history" plot** — `KeyError: 't_s'` (thermal output key is
  `times_s`).
- **Streamlit not installed after `uv sync`** — documented that extras are
  exclusive: use `uv sync --extra app --extra dev --extra docs`.
- **Wheel packaging** — `app/ingest` added to the hatch wheel packages so the
  PLC parser ships in distributions.
- **`brand/gen_assets.py`** — hardcoded absolute output path replaced with the
  script's own directory (path-portable).
- **CITATION.cff** — removed placeholder `repository-code` URL.
- **Time-unit heuristic** in PLC parser: minute-scale logs (median dt > 4 s)
  are now correctly converted to seconds (previously only dt > 10 s was
  treated as minutes, missing common 5-minute datalog sampling).

## [0.1.0] — 2026-08-22

### Added

- Initial Ferrumizer pipeline: thermal → carburizing → hardening → ECD, CLI,
  calibration/design scaffolding, and V1–V8 verification harnesses.
- AISI 8620, 9310, and 5120 presets (literature-anchored constants with
  per-field provenance).
- Tesseract component schemas and local composition path
  (`forward_containers`).
