"""Ferrumizer command-line interface.

Implements the verbs documented in README "What you can do with it":

    ferrumize validate CONFIG
    ferrumize simulate CONFIG
    ferrumize calibrate DATA.yaml --chains 4 --draws 1000
    ferrumize design TARGET_ECD_MM --alloy 8620 [--penalty gas|energy|none]
    ferrumize identifiability CONFIG
    ferrumize ingest PLC_LOG
    ferrumize verify
    ferrumize figures
    ferrumize app

Global flags --seed and --verbose are respected everywhere. Structured logging
is enabled; exit codes are meaningful (0 = success, 1 = failure/gate blocked).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

# Make repo root and app/ importable when running in-place (uninstalled).
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ferrumize.config import (  # noqa: E402
    load_config,
    params_from_config,
    scenario_from_config,
    validate_config,
)
from ferrumize.pipeline import FerrumizerPipeline  # noqa: E402

app = typer.Typer(
    name="ferrumize",
    help="Ferrumizer — the differentiable heat-treatment engine.",
    add_completion=False,
    no_args_is_help=True,
)

# Global options stored on the typer context.
_state: dict = {"seed": 0, "verbose": False}


@app.callback()
def main(
    seed: int = typer.Option(0, "--seed", help="Global RNG seed for determinism."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging."),
):
    """Ferrumize: model, calibrate and design gas-carburizing heat treatment."""
    _state["seed"] = seed
    _state["verbose"] = verbose
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def validate(config: Path = typer.Argument(..., exists=True, help="Run config YAML.")):
    """Validate a run configuration (schema, stability bounds, alloy preset)."""
    cfg = load_config(config)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            typer.secho(f"  ✗ {e}", fg=typer.colors.RED)
        _fail(f"Config {config} is INVALID ({len(errors)} error(s)).")
    info = cfg.get("_info", {})
    typer.secho(f"Config {config} is valid.", fg=typer.colors.GREEN)
    if "thermal_dt_max" in info:
        typer.echo(f"  thermal dt stability limit: {info['thermal_dt_max']:.4g} s")


@app.command()
def simulate(
    config: Path = typer.Argument(..., exists=True, help="Run config YAML."),
    out: Path = typer.Option(Path("results/simulate"), "--out", help="Output directory."),
):
    """Run the forward pipeline and write npz/parquet + standard plots."""
    import numpy as np

    cfg = load_config(config)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            typer.secho(f"  ✗ {e}", fg=typer.colors.RED)
        _fail("Cannot simulate an invalid config. Run `ferrumize validate` first.")

    scenario = scenario_from_config(cfg)
    params = params_from_config(cfg)
    pipe = FerrumizerPipeline(scenario, params)

    typer.echo(
        f"Simulating alloy={scenario.alloy} geometry={scenario.geometry} "
        f"t_total={scenario.t_total:.0f}s ..."
    )
    result = pipe.forward()

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "simulation.npz",
        times_s=np.asarray(result["thermal"]["times_s"]),
        T_surface=np.asarray(result["thermal"]["Ts"]),
        T_core=np.asarray(result["thermal"]["Tcore"]),
        C_final=np.asarray(result["carbon"]["C_final"]),
        x_mm=np.asarray(result["x_mm"]),
        H=np.asarray(result["H"]),
        ecd_mm=float(result["ecd_mm"]),
    )
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "depth_mm": np.asarray(result["x_mm"]),
                "carbon_wt_pct": np.asarray(result["carbon"]["C_final"]),
                "hardness_HV": np.asarray(result["H"]),
            }
        )
        df.to_parquet(out / "profile.parquet", index=False)
    except Exception as exc:  # parquet is optional convenience
        logging.debug("parquet export skipped: %s", exc)

    typer.secho(f"ECD = {float(result['ecd_mm']):.3f} mm", fg=typer.colors.CYAN)
    typer.secho(f"Wrote outputs to {out}/", fg=typer.colors.GREEN)


@app.command()
def calibrate(
    data: Path = typer.Argument(..., exists=True, help="Calibration data YAML."),
    chains: int = typer.Option(4, "--chains", help="Number of MCMC chains."),
    draws: int = typer.Option(1000, "--draws", help="Posterior draws per chain."),
    warmup: int = typer.Option(1000, "--warmup", help="Warmup steps per chain."),
    out: Path = typer.Option(Path("results/calibration"), "--out", help="Output directory."),
    carbon_n: int | None = typer.Option(
        None, "--carbon-n", help="Override carbon grid nodes for the likelihood (smaller = faster)."
    ),
    carbon_dt: float | None = typer.Option(
        None, "--carbon-dt", help="Override carbon time step (s) for the likelihood (larger = faster)."
    ),
):
    """Bayesian calibration (NumPyro NUTS) against a measured hardness traverse.

    The likelihood runs the lumped-thermal surrogate + carbon FD + hardening
    chain (ADR-002). At the shipped calibration_data.yaml grid (n=81, dt=2 s)
    each likelihood eval is ~1.5 s, so full 4-chain runs are CPU-heavy; use
    --carbon-n/--carbon-dt to lighten the grid (still well inside the
    explicit stability limit) or --chains/--draws to reduce MCMC cost.
    """
    import numpy as np

    from calibration.calibrate import run_calibration
    from ferrumize.config import scenario_from_config

    cfg = load_config(data)
    scenario = scenario_from_config(cfg)
    if carbon_n is not None or carbon_dt is not None:
        import dataclasses

        scenario = dataclasses.replace(
            scenario,
            carbon_n=carbon_n if carbon_n is not None else scenario.carbon_n,
            carbon_dt=carbon_dt if carbon_dt is not None else scenario.carbon_dt,
        )

    # Load observed traverse
    traverse = cfg.get("observations", {}).get("traverse_csv")
    if not traverse:
        _fail("Calibration data YAML must specify observations.traverse_csv")
    tpath = (data.parent / traverse).resolve()
    obs = np.genfromtxt(tpath, delimiter=",", names=True)
    obs_depths = np.asarray(obs["depth_mm"], dtype=np.float64)
    obs_H = np.asarray(obs["hardness_HV"], dtype=np.float64)

    typer.echo(f"Running NUTS calibration: chains={chains} draws={draws} warmup={warmup}")
    mcmc, summary = run_calibration(
        obs_depths,
        obs_H,
        scenario,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        seed=_state["seed"],
    )

    out.mkdir(parents=True, exist_ok=True)
    try:
        import arviz as az

        idata = az.from_numpyro(mcmc)
        idata.to_netcdf(str(out / "posterior.nc"))
    except Exception as exc:
        logging.warning("Could not write posterior netCDF: %s", exc)

    # Print gate report
    typer.secho(
        "\nParameter summary (release gates: R-hat < 1.01, bulk ESS > 400):",
        fg=typer.colors.BRIGHT_CYAN,
    )
    for name, row in summary["params"].items():
        flag = typer.colors.GREEN if row["gate_ok"] else typer.colors.RED
        typer.secho(
            f"  {name:>8s}  mean={row['mean']:.4g}  r_hat={row['r_hat']:.4f}  "
            f"ess={row['bulk_ess']:.0f}  [{'OK' if row['gate_ok'] else 'BLOCKED'}]",
            fg=flag,
        )
    if not summary["gates_ok"]:
        _fail("Calibration BLOCKED from release: convergence gates not met.")
    typer.secho(f"\nCalibration passed gates. Results in {out}/", fg=typer.colors.GREEN)


@app.command()
def design(
    target_ecd_mm: float = typer.Argument(..., help="Target effective case depth (mm)."),
    alloy: str = typer.Option("8620", "--alloy", help="Alloy preset."),
    penalty: str = typer.Option("none", "--penalty", help="Penalty: gas|energy|none."),
    config: Path | None = typer.Option(None, "--config", exists=True, help="Base scenario config."),
    out: Path = typer.Option(Path("results/design"), "--out", help="Output directory."),
):
    """Optimize a furnace schedule to hit a target ECD (optional energy penalty)."""
    from design.optimize import design_schedule, pareto_front
    from ferrumize.pipeline import ProcessParams, Scenario

    if config is not None:
        cfg = load_config(config)
        scenario = scenario_from_config(cfg)
        params = params_from_config(cfg).__dict__
    else:
        scenario = Scenario(alloy=alloy)
        params = ProcessParams().__dict__

    typer.echo(
        f"Designing schedule: target ECD={target_ecd_mm} mm, alloy={scenario.alloy}, penalty={penalty}"
    )
    if penalty == "none":
        res = design_schedule(target_ecd_mm, scenario, params, penalty="none", seed=_state["seed"])
        results = [res]
    else:
        results = pareto_front(target_ecd_mm, scenario, params, seed=_state["seed"])

    out.mkdir(parents=True, exist_ok=True)
    import json

    with open(out / "design.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    for r in results:
        feasible = r.get("feasible", True)
        tag = "OK" if feasible else "INFEASIBLE"
        typer.secho(
            f"  [{tag}] ECD={r['achieved_ecd_mm'] if 'achieved_ecd_mm' in r else r['ecd_mm']:.3f} mm  "
            f"(target {r['target_ecd_mm']:.3f})  energy={r['energy_proxy']:.3e}",
            fg=typer.colors.GREEN if feasible else typer.colors.YELLOW,
        )
        if not feasible:
            reach = r.get("reachable_range_mm")
            if reach:
                typer.secho(
                    f"         reachable range: [{reach[0]:.3f}, {reach[1]:.3f}] mm — "
                    "increase soak time, part size, or carbon potential to reach target",
                    fg=typer.colors.YELLOW,
                )
    typer.secho(f"Wrote design results to {out}/design.json", fg=typer.colors.GREEN)


@app.command()
def ingest(
    log: Path = typer.Argument(..., exists=True, help="PLC / datalogger export (CSV/TSV/TXT)."),
    out: Path = typer.Option(Path("results/ingested"), "--out", help="Output directory."),
):
    """Ingest a messy furnace PLC log into normalized trajectory/traverse data.

    Auto-detects delimiters, header rows, column roles (time/temperature/
    depth/hardness), and units (deg C vs deg F). Writes a JSON report and,
    when a time-temperature trajectory is present, a compressed
    piecewise-constant schedule usable by `ferrumize simulate`.
    """
    import json

    from ingest.plc_parser import parse_plc_log, schedule_from_trajectory

    report = parse_plc_log(log)
    typer.echo(f"Ingested {log.name}: {report.rows_used}/{report.rows_total} rows used "
               f"(temp unit: {report.temperature_unit})")
    for w in report.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)

    out.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()

    if report.has_trajectory:
        traj = report.trajectory
        assert traj is not None
        sched = schedule_from_trajectory(traj["t_s"], traj["T_C"])
        payload["schedule"] = sched
        typer.secho(
            f"  trajectory: {len(traj['t_s'])} points, "
            f"{len(sched['schedule_times'])} soak segment(s)",
            fg=typer.colors.GREEN,
        )
    if report.has_traverse:
        trav = report.traverse
        assert trav is not None
        typer.secho(
            f"  traverse: {len(trav['depth_mm'])} points (depth_mm vs hardness_HV)",
            fg=typer.colors.GREEN,
        )

    with open(out / "ingested.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    typer.secho(f"Wrote ingested data to {out}/ingested.json", fg=typer.colors.GREEN)


@app.command()
def identifiability(
    config: Path = typer.Argument(..., exists=True, help="Run config YAML."),
    out: Path = typer.Option(Path("results/identifiability"), "--out", help="Output directory."),
):
    """Fisher/correlation analysis: single- vs two-schedule identifiability."""
    import numpy as np

    from identifiability.analyze import identifiability_report

    cfg = load_config(config)
    scenario = scenario_from_config(cfg)
    params = params_from_config(cfg)

    # Build a synthetic observation set from the forward model for analysis.
    pipe = FerrumizerPipeline(scenario, params)
    result = pipe.forward()
    obs_depths = np.asarray(result["x_mm"])
    obs_H = np.asarray(result["H"])

    import numpy as np

    param_vec = np.array([np.log(params.D0), params.Q_kJ, params.C_pot, params.h_m, params.eps])
    report = identifiability_report(param_vec, obs_depths, obs_H, scenario)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "fisher.npy", report["fisher"])
    np.save(out / "correlation.npy", report["correlation"])
    typer.secho(f"Condition number: {report['condition_number']:.3e}", fg=typer.colors.CYAN)
    typer.echo("Correlation matrix:")
    names = report["param_names"]
    corr = report["correlation"]
    header = "        " + "  ".join(f"{n[:6]:>7s}" for n in names)
    typer.echo(header)
    for i, n in enumerate(names):
        row = f"{n[:8]:>8s}" + "  ".join(f"{corr[i, j]:7.3f}" for j in range(len(names)))
        typer.echo(row)
    typer.secho(f"Wrote matrices to {out}/", fg=typer.colors.GREEN)


@app.command()
def verify():
    """Run the full V1-V8 verification suite; nonzero exit on any FAIL."""
    from verification.cross_ad.v4_cross_ad import run_v4, run_v4_containers
    from verification.limits.v1_lumped import run_v1
    from verification.limits.v2_erfc import run_v2
    from verification.mms.v3_mms import run_v3
    from verification.v5_check_gradients import run_v5
    from verification.v6_recovery import run_v6
    from verification.v7_sbc_tarp import run_v7
    from verification.v8_literature import run_v8
    from verification.q_quench import run_q1, run_q2, run_q3

    results = []
    runners = [
        ("V1", run_v1),
        ("V2", run_v2),
        ("V3", run_v3),
        ("V4", run_v4),
        ("V4c", run_v4_containers),
        ("V5", run_v5),
        ("V6", run_v6),
        ("V7", run_v7),
        ("V8", run_v8),
        ("Q1", run_q1),
        ("Q2", run_q2),
        ("Q3", run_q3),
    ]
    for vid, fn in runners:
        typer.echo(f"Running {vid} ...")
        try:
            r = fn()
            passed = bool(r.get("passed", False))
            results.append((vid, passed, r))
        except Exception as exc:
            results.append((vid, False, {"error": str(exc)}))

    typer.echo("\n=== Verification Suite ===")
    all_ok = True
    for vid, passed, r in results:
        all_ok = all_ok and passed
        color = typer.colors.GREEN if passed else typer.colors.RED
        label = "PASS" if passed else "FAIL"
        detail = ""
        for key in ("max_rel_err", "norm_l2", "rel_inf_norm", "sbc_p_value", "delta_mm"):
            if key in r:
                detail = f" {key}={r[key]:.3g}"
                break
        typer.secho(f"  {vid} [{label}]{detail}", fg=color)
    if not all_ok:
        _fail("Verification suite FAILED.")
    typer.secho("All verification gates PASSED.", fg=typer.colors.GREEN)


@app.command()
def figures(
    out: Path = typer.Option(Path("figures"), "--out", help="Figures directory."),
    only: str = typer.Option(None, "--only", help="Comma-separated subset, e.g. F3,F6."),
):
    """Regenerate figures F1-F10 deterministically (seeded).

    Slow figures (F6 NUTS posterior, F7 noise sweep, F9 Pareto) run first and
    print their elapsed time; use --only to regenerate a subset.
    """
    from ferrumize.figures import generate_all

    out.mkdir(parents=True, exist_ok=True)
    if only:
        from ferrumize import figures as figmod

        wanted = {s.strip().upper() for s in only.split(",")}
        mapping = {
            "F1": figmod.fig_f1_hero,
            "F2": figmod.fig_f2_architecture,
            "F3": figmod.fig_f3_erfc,
            "F4": figmod.fig_f4_mms,
            "F5": figmod.fig_f5_cross_ad,
            "F6": figmod.fig_f6_posterior,
            "F7": figmod.fig_f7_noise_sweep,
            "F8": figmod.fig_f8_identifiability,
            "F9": figmod.fig_f9_pareto,
            "F10": figmod.fig_f10_alloys,
        }
        for name in sorted(wanted):
            if name not in mapping:
                _fail(f"Unknown figure {name} (valid: {','.join(sorted(mapping))})")
        import time as _time

        for name in sorted(wanted):
            t0 = _time.time()
            typer.echo(f"[figures] {name} ...")
            mapping[name](out)
            typer.echo(f"[figures] {name} done in {_time.time() - t0:.1f}s")
    else:
        generate_all(out, seed=_state["seed"])
    typer.secho(f"Figures written to {out}/", fg=typer.colors.GREEN)


def app_cmd(
    port: int = typer.Option(8501, "--port", help="Streamlit port."),
):
    """Launch the Streamlit demo app."""
    import subprocess

    app_path = REPO_ROOT / "app" / "streamlit_app.py"
    if not app_path.exists():
        _fail(f"Streamlit app not found at {app_path}")
    typer.echo(f"Launching Streamlit app on port {port} ...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=False,
    )


# Register under the spec's command name: `ferrumize app`.
app.command(name="app")(app_cmd)


if __name__ == "__main__":
    app()
