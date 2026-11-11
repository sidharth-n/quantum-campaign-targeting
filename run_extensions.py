"""Studies that go beyond the brief, built on the committed stage files:

    python run_extensions.py

It reads the six selected customers and their stage 1 probabilities from stage3_simulation.json
and the stage 2 problem from stage2_selection.json. It never writes to the three stage files,
so the submitted answer cannot change here. Results go to results/extensions/, one figure to
results/figures/, and the README is refreshed from the result files.

  correlated customers    an entangled circuit that relaxes the independence assumption of stage 3,
                          and what it does to the success probability and to the money at risk
  amplitude estimation    the stage 3 probability with an error that falls like 1/N instead of 1/sqrt(N)
  IBM device snapshots    the stage 2 and stage 3 circuits on the calibration data of two real IBM
                          processors (simulated; no account needed)

Run `python run_pipeline.py` first if the stage files are missing.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from campaign import amplitude, config, correlated, figures, ibm_backends, outputs  # noqa: E402


def inputs(root: Path) -> dict:
    """Ids, probabilities and money terms of the six selected customers, in stage 1 rank order."""
    s2 = json.loads((root / "stage2_selection.json").read_text())
    s3 = json.loads((root / "stage3_simulation.json").read_text())
    runs = json.loads((root / "results" / "stage2_qaoa_runs.json").read_text())["penalty_qaoa"]
    by_id = {row["customer_id"]: row for row in s2["candidates"]}
    ids = s3["selected_ids"]
    if [entry["customer_id"] for entry in s3["qubit_order"]] != ids or sorted(ids) != sorted(s2["selected_ids"]):
        raise RuntimeError("the stage 2 and stage 3 files do not describe the same six customers")
    probabilities = s3["probabilities"]
    for cid, p in zip(ids, probabilities):
        if by_id[cid]["predicted_probability"] != p:
            raise RuntimeError(f"probability of {cid} differs between the stage 2 and stage 3 files")
    return {
        "ids": ids,
        "probabilities": probabilities,
        "deposits": [float(by_id[cid]["expected_deposit_value"]) for cid in ids],
        "costs": [float(by_id[cid]["contact_cost"]) for cid in ids],
        "threshold": int(s3["success_condition"].split()[2]),
        "stage2": s2,
        "qaoa_runs": {
            int(depth): {"gamma": run["gamma"], "beta": run["beta"], "p_optimal_ideal": run["exact_distribution"]["p_optimal"]}
            for depth, run in runs.items()
        },
    }


def main(root: Path = ROOT, out: Path = None, make_figures: bool = True) -> dict:
    out = root if out is None else out
    clock = time.time()

    def log(message):
        print(f"[{time.time() - clock:6.1f}s] {message}", flush=True)

    given = inputs(root)
    ids, p, threshold = given["ids"], given["probabilities"], given["threshold"]
    log(f"six customers {','.join(ids)}, success = at least {threshold}")

    corr = correlated.run(ids, p, given["deposits"], given["costs"], threshold)
    log("correlated customers: " + ", ".join(
        f"rho {entry['rho']:.1f} -> {entry['exact']:.4f}" for entry in corr["scenarios"]
    ))

    ae = amplitude.run(p, threshold)
    best = ae["best_schedule"]
    log(f"amplitude estimation: error slope {ae['fitted_slope_log_rmse_vs_log_cost']:.2f} "
        f"(direct sampling {ae['fitted_slope_direct_sampling_measured']:.2f}), RMSE {best['rmse']:.5f} against "
        f"{best['direct_sampling_rmse_measured']:.5f} at {best['applications_of_A']} applications; "
        f"on the noise model {best['noisy_rmse']:.4f} against {best['noisy_direct_sampling_rmse_measured']:.4f}")

    ibm = ibm_backends.run(p, threshold, given["stage2"], given["qaoa_runs"])
    for backend in ibm["backends"]:
        s3 = backend["stage3"]
        log(f"{backend['snapshot']}: stage 3 unmitigated {s3['unmitigated']['estimate']:.4f}, "
            f"mitigated {s3['mitigated']['estimate']:.4f} (exact {s3['exact']:.4f})")

    target = out / "results" / "extensions"
    outputs.write_json(target / "correlated_customers.json", corr)
    outputs.write_json(target / "amplitude_estimation.json", ae)
    outputs.write_json(target / "ibm_snapshots.json", ibm)
    if make_figures:
        figures.extensions(corr, ae, ibm, out / "results" / "figures" / "beyond_the_brief.png")
    log(f"files written to {target}")
    if out.resolve() == ROOT:
        import build_report
        build_report.README.write_text(build_report.render())
        log("README.md refreshed from the result files")
    return {"correlated": corr, "amplitude": ae, "ibm": ibm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT, help="output directory (default: repository root)")
    parser.add_argument("--no-figures", action="store_true", help="skip the PNG figure")
    args = parser.parse_args()
    main(ROOT, args.out, make_figures=not args.no_figures)
