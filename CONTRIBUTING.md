# Contributing

Ferrumizer is an Apache-2.0 research codebase. Keep changes narrow, reproducible, and domain-explicit.

## Development

```bash
uv sync --extra dev
pre-commit install
make test
make verify
```

Every numerical change should add or update a failing-test-first artifact, a verification gate, or an ADR. Do not hide solver tolerances, grid sizes, priors, or finite-difference steps in code.

Use conventional commits. Pull requests must pass Ruff, tests, verification, and the documentation build.
