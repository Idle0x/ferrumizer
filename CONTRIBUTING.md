# Contributing

Ferrumizer is an Apache-2.0 research codebase. The bar for a contribution is:
**a numerical or behavioral change must carry its own proof.** Keep changes
narrow, reproducible, and domain-explicit.

## Development setup

```bash
uv sync --extra app --extra dev --extra docs   # extras are exclusive — include them all
pre-commit install
make data
make test
make verify    # full V1–V8 + Q1–Q3, ~25 min
```

Requirements: Python 3.12+, `uv`. JAX runs on CPU; no GPU needed for any gate
or command.

## What to touch for what

| You want to… | Touch this |
|---|---|
| Change physics (thermal/carbon/hardening/quench) | `components/shared/ferrumizer_physics/` + the affected verification gate |
| Add an alloy preset | `components/shared/ferrumizer_physics/alloys/aisi_*.yaml` (follow the 8620 schema, provenance per field) or `composition_to_preset()` for runtime chemistry |
| Add a CLI command | `app/ferrumize/cli.py` + `docs/cli.md` |
| Touch the app | `app/streamlit_app.py` — every control gets a tooltip, every chart gets an explainer paragraph |
| Add a parsing feature | `app/ingest/plc_parser.py` — the parser must warn on every assumption it makes |
| Add a figure | `app/ferrumize/figures.py` — must be deterministic (seeded) and documented in README + gallery |

## Rules

1. **Tests first.** Every numerical change adds or updates a
   failing-test-first artifact, a verification gate, or an ADR — not a
   comment.
2. **No hidden knobs.** Do not hide solver tolerances, grid sizes, priors, or
   finite-difference steps in code without an ADR explaining why they exist.
3. **Honesty over aesthetics.** If a result is a synthetic reconstruction or
   an order-of-magnitude estimate, say so in the same file (see the alloy
   YAML comments and README "Honest limitations").
4. **Conventional commits.** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.
5. **Gates are gates.** `make verify` must pass for anything touching physics;
   `ferrumize calibrate` refuses to release results that fail convergence
   gates — do not "fix" that by weakening the gates.
6. **Determinism.** Anything that produces figures/data uses a fixed seed.
   Never introduce `np.random` without a seed.
7. **No internal references.** Public files never reference local absolute
   paths, internal planning docs, or private infra. Links are relative.

## Pull request checklist

- [ ] `make test` green (pytest)
- [ ] `ruff check app components tests verification` clean
- [ ] `mypy app` clean
- [ ] `mkdocs build` succeeds
- [ ] `make verify` green if physics touched
- [ ] README/gallery updated if the user-facing surface changed
- [ ] CHANGELOG entry added

## Reporting bugs

File an issue with: the command run, the full config/data, expected vs actual
output, and the log. If a gate fails, include the gate's numbers — the
contract is the source of truth.
