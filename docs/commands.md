# Commands reference

This is the complete, detailed reference for every command used in the
project: the `ferrumize` CLI, the Makefile targets, and the dev tooling.
For each command you get the exact syntax, a real usage example, what it
writes, what to expect on success, and what failure looks like.

Timings are **measured** wall time on the 4-vCPU AMD EPYC machine with the
pinned dependency versions (see [reproducing.md](reproducing.md)); a faster
machine runs them faster. Memory numbers are peak RSS of the actual process.

- [Setup](#setup)
- [Global options](#global-options)
- [Exit code convention](#exit-code-convention)
- [Run-config YAML (shared by simulate / identifiability / figures)](#run-config-yaml)
- [ferrumize commands](#ferrumize-commands)
  - [`validate`](#validate)
  - [`simulate`](#simulate)
  - [`calibrate`](#calibrate)
  - [`design`](#design)
  - [`identifiability`](#identifiability)
  - [`ingest`](#ingest)
  - [`verify`](#verify)
  - [`figures`](#figures)
  - [`app`](#app)
- [Makefile targets](#makefile-targets)
- [Dev tooling](#dev-tooling)

---

## Setup

One-time, from a clean clone:

```bash
git clone https://github.com/Idle0x/ferrumizer.git
cd ferrumizer

uv tool install -e . --with streamlit    # puts `ferrumize` on PATH, no activation
uv sync --extra app --extra dev --extra docs   # full dev environment
```

- `uv tool install -e . --with streamlit` makes `ferrumize` available in
  every shell. `-e` keeps the install editable (code edits are picked up
  without reinstalling, and the verification gates can resolve repo data via
  their own file location); `--with streamlit` pulls the app dependency,
  which is an optional extra rather than a core requirement.
- `uv sync --extra app --extra dev --extra docs` builds the `.venv` used by
  `make test`, pre-commit, and the `source .venv/bin/activate` fallback.
- The synthetic datasets under `data/` are committed — nothing else is
  needed before the first `ferrumize` command.

**`ferrumize: command not found`?** The tool install did not land, or
`~/.local/bin` is not on your `PATH`. Re-run
`uv tool install -e . --with streamlit` from the repo root, check that
`~/.local/bin` is on `PATH`, or fall back to
`source .venv/bin/activate` (requires `uv sync`).

Quick check that the install works:

```bash
ferrumize verify --fast
```

All 11 fast gates should print PASS and the exit code should be 0.

## Global options

`ferrumize` is a [Typer](https://typer.tiangolo.com/) app. Two global
options are accepted **before** the subcommand:

| Flag | Meaning |
|---|---|
| `--seed N` | Global RNG seed for deterministic runs. Default `0`. |
| `--verbose` | Debug-level logging. |

`ferrumize --help` lists the commands; `ferrumize CMD --help` lists every
option of one command.

## Exit code convention

Every command exits **0 on success** and **1 on failure or a blocked gate**.
A verification gate that FAILs and a calibration whose convergence gates do
not pass are both hard failures (exit 1), not warnings — so the commands are
safe to chain in scripts and CI.

## Run-config YAML

Several commands take a `CONFIG` argument. **`CONFIG` is the path to a
single YAML file that you write yourself** — not a folder, and there is no
special directory it must live in. Save it anywhere (next to the repo, in
`~/cases/`, in `/tmp` — any path works), then pass that path wherever a
command asks for `CONFIG`:

```bash
ferrumize simulate my_case.yaml          # my_case.yaml is the file you wrote
ferrumize identifiability my_case.yaml
```

`simulate`, `identifiability`, and (for the process figures) `figures` all
accept the same file. The shipped
[`data/synthetic/calibration_data.yaml`](../data/synthetic/calibration_data.yaml)
is a ready-made template — copy it and edit the values. A minimal one:

```yaml
alloy: 8620            # 8620 | 9310 | 5120 (shipped presets)
geometry: slab         # slab | cylinder
size_mm: 16.0
t_total: 7200.0        # total run time, seconds

schedule:              # piecewise-constant furnace schedule
  times:   [0.0, 1200.0, 7200.0]
  temps_C: [900.0, 950.0, 950.0]

params:
  D0: 2.2e-5           # carbon diffusion prefactor (m^2/s)
  Q_kJ: 137.0          # activation energy (kJ/mol)
  C_pot: 1.0           # carbon potential (wt % C)
  h_m: 1.0e-4          # mass-transfer coefficient (m/s)
  eps: 0.8             # surface emissivity

# optional discretization (defaults are stable)
thermal: {n: 21, sample_every: 20}
carbon:  {n: 41, dt: 1.0, sample_every: 300, mode: mass_transfer}
```

Optional extras:

- `schedule2:` — a second temperature schedule with the same schema. When
  present, `calibrate`, `identifiability`, and the F6/F8 figures use the
  two-schedule protocol that breaks the D0-Q degeneracy.
- `T_init_K:` — initial part temperature (default 298.15 K).
- `observations:` — only used by `calibrate`; see below.

Validate any config before spending compute:

```bash
ferrumize validate my_case.yaml
```

On success it prints `Config my_case.yaml is valid.` plus the thermal
explicit-step stability limit. On failure it prints every problem (missing
field, unstable grid, unknown alloy preset) and exits 1 — `simulate`
refuses to run an invalid config and tells you to validate first.

## ferrumize commands

### validate

```bash
ferrumize validate CONFIG
```

Checks a run-config YAML against the schema: required fields, alloy preset
exists, schedule knots are monotonic, the thermal explicit time step is
inside the FTCS stability limit for the requested grid.

- **Expect:** one green `Config ... is valid.` line and the `thermal dt
  stability limit` value (your carbon `dt` must stay below it).
- **Errors:** each violation listed with `✗`, exit 1. Common ones:
  unknown alloy name (typo — presets are `8620`, `9310`, `5120`),
  `carbon.dt` above the stability limit, non-increasing `schedule.times`.
- **Runtime:** instant.

### simulate

```bash
ferrumize simulate CONFIG --out results/simulate
```

Runs the full differentiable forward pipeline
(schedule → thermal history → carbon profile → hardness → ECD) for one
scenario.

- **Example:**
  ```bash
  ferrumize validate my_case.yaml
  ferrumize simulate my_case.yaml --out results/8620_16mm
  ```
- **Writes to `--out`:** `simulation.npz` (times, T_surface, T_core,
  C_final, x_mm, H, ecd_mm), `profile.parquet` (depth / carbon_wt_pct /
  hardness_HV table), plus standard plots.
- **Expect:** `ECD = X.XXX mm` printed in cyan and the output directory.
  Typical single run: ~350 MB peak RSS, well under a minute on CPU.
- **Errors:** invalid config → exit 1 with the validation list; a
  non-convergent or non-finite forward pass raises and exits 1 (this would
  indicate a real model bug — the V1–V5 gates exist to keep it from
  happening silently).
- **Use case:** "what ECD / hardness profile does *this* schedule give?" —
  the workhorse for forward questions and for building test traverses.

### calibrate

```bash
ferrumize calibrate DATA.yaml --chains 4 --draws 1000
```

Bayesian (NumPyro NUTS) inference of the five process parameters
`{log D0, Q, C_pot, h_m, eps}` from one or two measured hardness traverses.

The data YAML is the same run-config file plus an `observations` block that
names your measurement CSV files. **Put the CSVs in the same folder as the
YAML** and refer to them by filename only — the command looks for them
relative to the YAML's own folder:

```yaml
# ... same header as a run-config ...
observations:
  traverse_csv: my_measurements.csv        # columns: depth_mm,hardness_HV
  traverse_csv2: my_measurements_2.csv     # optional second schedule's traverse
```

```csv
depth_mm,hardness_HV
0.0,980
0.5,940
1.0,880
2.0,700
3.0,450
5.0,310
```

- **Example:**
  ```bash
  ferrumize calibrate data/synthetic/calibration_data.yaml
  ferrumize calibrate my_traverse.yaml --chains 4 --draws 500 --warmup 500
  ```
- **Writes to `--out` (default `results/calibration/`):** `posterior.nc`
  (arviz-compatible netCDF), `ppc_hardness.png` (observed vs posterior mean
  with the 90% credible band).
- **Expect:** a per-parameter gate report:
  `mean / r_hat / ess / [OK|BLOCKED]`. Release gates: R-hat < 1.01 and bulk
  ESS > 400 per parameter; a final PPC line with the max |residual| in HV.
- **Cost and how to lighten it:** this is the slowest day-to-day command —
  each likelihood evaluation runs the forward model, so full 4-chain /
  1000-draw runs are CPU-heavy. Options, in order of preference:
  - `--carbon-n 41 --carbon-dt 2.0` — lighter carbon grid in the likelihood
    (still inside the stability limit); big wall-time reduction.
  - fewer `--chains` / `--draws` / `--warmup` — direct MCMC cost reduction;
    the gates will tell you if it went below release standard.
- **Errors:**
  - missing `observations.traverse_csv` → clear failure, exit 1.
  - traverse CSV without `depth_mm,hardness_HV` header columns → parse
    error, exit 1.
  - **convergence gates not met → the command exits 1 and prints
    "Calibration BLOCKED from release".** This is intentional: a
    non-converged posterior is a refusal, not a result. `h_m` is the usual
    suspect — it is weakly identifiable from end-state hardness (see V6 and
    the Limitations section of the README), so its gate may legitimately
    stay wide.
- **Use case:** "what were the effective furnace parameters for the part I
  measured?" — with honest posterior uncertainty.

### design

```bash
ferrumize design TARGET_ECD_MM --alloy 8620 [--penalty gas|energy|none] [--config CONFIG]
```

Gradient-based inverse design: find a furnace schedule that hits a target
ECD.

- **Examples:**
  ```bash
  ferrumize design 0.8                                    # 8620, no penalty
  ferrumize design 1.0 --alloy 9310 --penalty energy      # ECD-vs-energy Pareto front
  ferrumize design 0.6 --config my_case.yaml              # base scenario from my config
  ```
- **Writes to `--out` (default `results/design/`):** `design.json` with the
  optimized schedule(s), achieved ECD, and energy proxy.
- **Expect:** one line per candidate —
  `[OK] ECD=0.801 mm (target 0.800) energy=1.23e+06`. With a penalty you get
  the whole Pareto front instead of a single point.
- **Errors / honest failure:** an infeasible target prints
  `[INFEASIBLE]` **plus the physically reachable range** and what to change
  (longer soak, larger part, higher carbon potential) instead of grinding
  toward an impossible value.
- **Note:** the energy term is a relative proxy (time-integral of setpoint
  above ambient), not a metered energy figure — see README Limitations.

### identifiability

```bash
ferrumize identifiability CONFIG
```

Fisher-information / correlation analysis: which parameters can the data
actually pin down, and does the two-schedule protocol break the D0-Q
degeneracy?

- **Example:**
  ```bash
  ferrumize identifiability data/synthetic/calibration_data.yaml
  ```
  (that config has a `schedule2` block, so you get the full comparison).
- **Writes to `--out` (default `results/identifiability/`):** Fisher and
  correlation matrices (`.npy`), and 2-D profile-likelihood contours over
  (log D0, Q) for the single- and combined-schedule cases.
- **Expect:** condition numbers (full matrix and the (D0,Q)-block with
  nuisance parameters profiled out), a printed correlation matrix, and a
  one-paragraph reading: the (D0,Q)-block's flat eigenvalue grows from the
  single-schedule value to the combined value — that growth is the
  quantitative evidence the second schedule adds real curvature. The full
  matrix stays ill-conditioned because `h_m` is weakly identifiable; the
  output says this explicitly rather than hiding it.
- **Use case:** deciding whether one hardness traverse is enough data, or
  whether a second schedule is needed, before spending a calibration run.

### ingest

```bash
ferrumize ingest PLC_LOG --out results/ingested
```

`PLC_LOG` is the path to the export file you received from your furnace's
PLC or datalogger (CSV, TSV, or TXT) — any path works, no setup needed. You
do **not** have to clean it up first: a typical raw export looks like this
and is handled as-is:

```csv
timestamp,furnace_temp (degF),part_temp (degF)
08:00:00,1742,1701
08:05:00,1742,1712
08:10:00,1740,1725
...
```

The command parses this messy log into normalized trajectory and/or
traverse data.

- **Example:**
  ```bash
  ferrumize ingest /path/to/furnace_export.csv
  ```
- **Handles:** delimiter and header-row auto-detection; column-role mapping
  (time / temperature / depth / hardness) via a synonym table; °C and °F;
  time units in s, min, h, and HH:MM:SS clock timestamps; malformed rows
  (skipped and counted). Noisy temperature trajectories are
  Ramer-Douglas-Peucker-compressed into a few piecewise-constant schedule
  knots.
- **Writes to `--out`:** `ingested.json` — the full normalized payload,
  every assumption made, and, when a trajectory is present, a `schedule`
  block ready to paste into a `simulate` / `calibrate` config.
- **Expect:**
  `Ingested furnace_export.csv: 11540/11545 rows used (temp unit: C)`
  plus one `!` warning line per assumption (unit conversion, skipped rows,
  range violations). Nothing is silently dropped.
- **Errors:** a file with no recognizable time + temperature columns fails
  loudly with the reasons. Traverse-only files (depth vs hardness) parse
  fine but carry no trajectory — the app's Log Ingestion tab then points
  you to the Cycle Predictor for calibration instead of offering a dead
  end.
- **Use case:** the bridge from "CSV export I got from the furnace vendor"
  to calibration and emulation. See [ingestion.md](ingestion.md) for the
  column synonyms and unit rules.

### verify

```bash
ferrumize verify            # full suite: ~4 h
ferrumize verify --fast     # CI: minutes
```

Runs the verification gates, each an independent script under
`verification/`. Full suite: V1, V2, V3, V4, V4c, V5, V6, V7, V8, V8b,
Q1, Q2, Q3.

- **What each gate proves** (one line):
  - **V1** lumped-thermal surrogate vs full solver (max rel err < 0.5%)
  - **V2** carbon diffusion vs analytical erfc (normalized L2 < 1e-3)
  - **V3** manufactured-solution convergence (order ≥ 1.85)
  - **V4** FD Jacobian vs JAX autodiff (rel inf-norm < 1e-3)
  - **V4c** end-to-end gradient through the containerized Tesseract
    composition (finite, non-zero, within 20% of the FD reference)
  - **V5** runtime gradient checks, zero failures on all AD boxes
  - **V6** two-schedule parameter recovery (~20 min) — strongly identified
    parameters recover to < 5e-3; `h_m` documented as weakly identifiable
  - **V7** 200-simulation SBC/TARP posterior-calibration check (~4 h)
  - **V8** literature-traverse reconstruction against synthetic ground
    truth
  - **V8b** Jominy end-quench vs the published 8620H hardenability band
    (current MAE 2.6 HRC at all 13 J positions)
  - **Q1–Q3** quench-model sanity: medium ranking, slow-quench collapse,
    differentiability
- **Expect:** a `=== Verification Suite ===` table with one line per gate
  and its key metric; exit 0 only if every gate printed PASS.
- **`--fast`:** skips exactly V6 and V7 (the two long gates), everything
  else runs. No thresholds are loosened.
- **Errors:** any FAIL exits 1 with the metric that failed, e.g.
  `V4 [FAIL] rel_inf_norm=3.2e-3`. A gate crashing mid-run is reported the
  same way — an exception is a FAIL, not a silent skip.

### figures

```bash
ferrumize figures
ferrumize figures --only F3,F8
ferrumize figures --only F1,F6,F10 --config my_case.yaml --seed 1
```

Regenerates the F1–F10 figures, seeded and deterministic.

- **The three figure classes:**
  - **Solver validation** (F3, F4, F5, F7) — fixed by design; they test
    numerical properties, not a part.
  - **Method demonstrations** (F2, F8, F9) — fixed protocols
    (architecture, identifiability, Pareto).
  - **Process figures** (F1, F6, F10) — accept `--config` and render your
    alloy/schedule.
- **Writes to `--out` (default `figures/`):** the PNG/GIF files with the
  names used by the README.
- **Expect:** slow figures (F6 NUTS posterior, F7 noise sweep, F9 Pareto)
  run first and print their elapsed seconds; the canonical no-config run is
  byte-identical to the committed `figures/` under the same dependency
  versions (seed 0).
- **Errors:** `--only` with an unknown name (e.g. `F11`) fails immediately
  listing the valid names; a bad `--config` fails at the same validation as
  `simulate`.

### app

```bash
ferrumize app               # port 8501
ferrumize app --port 8599   # e.g. alongside an already-running instance
```

Launches the Streamlit Virtual Furnace: three tabs (Virtual Furnace —
interactive schedule/quench/alloy/geometry exploration; Cycle Predictor —
Bayesian calibration from a traverse or PLC log; Log Ingestion — inspect
and normalize furnace exports, then run the forward model on the ingested
schedule).

- **Expect:** `Launching Streamlit app on port 8501 ...` then a browser at
  `http://localhost:8501`. The running app is a ~230 MB process (two
  Streamlit processes).
- **Errors:** port in use → Streamlit's standard "port already in use"
  message; use `--port` for a second instance. App errors are logged to
  stderr and to the Streamlit UI.
- **Note:** the app uses the same physics engine as the CLI and Python API —
  a result in the app is reproducible with `ferrumize simulate` on the
  equivalent config.

## Makefile targets

The Makefile is for the in-place (not installed) workflow; it runs through
the venv's Python with `PYTHONPATH=components/shared:app`.

| Target | Does | Notes |
|---|---|---|
| `make data` | Regenerates the synthetic calibration data and the literature reference traverse | The committed datasets are canonical; only run this when the generator itself changed. |
| `make test` | Full pytest suite with coverage | 49 tests; exit 0 = all pass. |
| `make verify` | `ferrumize verify` via the module path | Same gates, no install needed. |
| `make figures` | `ferrumize figures` via the module path | Same figures. |
| `make run` | Launches the app via the module path | Same app as `ferrumize app`. |
| `make clean` | Removes build/test artifacts | `build/ dist/ .pytest_cache/ htmlcov/ results/ *.egg-info` |
| `make build` | Builds the sdist/wheel via `uv build` | For packaging, not day-to-day use. |

## Dev tooling

```bash
pytest tests/          # or: make test
ruff check .           # lint
ruff format .          # format (also enforced by pre-commit)
mypy                   # clean on all source files
pre-commit run --all-files
```

Pre-commit is wired to the repo (`.pre-commit-config.yaml`) and enforces
trailing-whitespace trimming, EOF newlines, YAML/JSON checks, ruff, and
ruff-format. If your editor writes markdown hard-line-breaks (lines ending
in two spaces), pre-commit will strip those — commit after running
`pre-commit run --all-files` rather than committing first.

---

Implementation: [`app/ferrumize/cli.py`](../app/ferrumize/cli.py) (CLI),
[`Makefile`](../Makefile) (targets). Timings and machine measurements:
[reproducing.md](reproducing.md). Command table summary: [cli.md](cli.md).
