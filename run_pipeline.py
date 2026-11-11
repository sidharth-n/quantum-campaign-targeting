"""Rebuild every result from the organisers' workbook with one command:

    python run_pipeline.py

Stage 1 probabilities -> top 10 -> stage 2 selection (QAOA, checked by brute force) -> stage 3
probability that at least 3 of the 6 selected customers subscribe. The three stage files are
written to the repository root, everything else to results/.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

from campaign import config, figures, outputs, propagation, stage1, stage2, stage3  # noqa: E402
from campaign.data import load  # noqa: E402


def main(out: Path, make_figures: bool = True, n_jobs: int = -1) -> dict:
    clock = time.time()

    def log(message):
        print(f"[{time.time() - clock:6.1f}s] {message}", flush=True)

    data = load()
    constraints = data.constraints
    log(f"data: {len(data.train)} training rows ({int(data.train['y'].sum())} subscribers), {len(data.test)} test rows")

    s1 = stage1.run(data, n_jobs=n_jobs)
    submitted = s1.cv_table[s1.cv_table["submitted"]].iloc[0]
    log(f"stage 1: submitted model = {s1.selection['submitted_model']}, "
        f"CV PR-AUC {submitted['pr_auc']:.3f}, Brier {submitted['brier']:.4f}")
    kernel = stage1.kernel_circuit_report(s1.model.quantum_)
    log(f"stage 1: kernel entries from the overlap circuit, mean abs error {kernel['mean_abs_error_ideal']:.4f} (ideal shots)")

    s2 = stage2.run(s1.predictions, data.ops, constraints)
    selection = s2["selection"]
    log(f"stage 2: selected {','.join(selection['ids'])} | value {selection['value']:.4f} | "
        f"cost {selection['cost']:.0f}/{constraints.budget:.0f} | capacity {selection['capacity']:.0f}/{constraints.capacity:.0f} | "
        f"gap to brute force {selection['optimality_gap']:.2e}")

    # the same ids and the same stage 1 probabilities, in stage 1 rank order
    chosen = s2["candidates"][s2["candidates"]["customer_id"].isin(selection["ids"])].sort_values("rank")
    threshold = -(-constraints.n_select // 2)        # at least half of the selected customers
    s3 = stage3.run(list(chosen["customer_id"]), chosen["predicted_probability"].to_numpy(), threshold)
    log(f"stage 3: P(at least {threshold} of {constraints.n_select}) quantum {s3['headline']['estimate']:.6f} | "
        f"exact {s3['exact']:.6f} | Monte Carlo {s3['monte_carlo']['estimate']:.6f}")

    prop = propagation.run(data, s1.estimator, s1.predictions, s2["candidates"], selection["ids"], n_jobs=n_jobs)
    log(f"error propagation: {prop['rounds']} refits, P(at least {threshold}) 5th to 95th percentile "
        f"{prop['stage3']['fixed_six']['p05']:.3f} to {prop['stage3']['fixed_six']['p95']:.3f}")

    comparison = propagation.model_comparison(data, s1.all_model_predictions, s1.selection["submitted_model"])
    outputs.write_all(out, s1, kernel, s2, s3, prop, comparison, constraints)
    if make_figures:
        figures.stage1(s1, out / "results" / "figures" / "stage1_cross_validation.png")
        figures.stage2(s2, out / "results" / "figures" / "stage2_qaoa.png")
        figures.stage3(s3, out / "results" / "figures" / "stage3_simulation.png")
        figures.propagation(prop, out / "results" / "figures" / "error_propagation.png")
    log(f"files written to {out}")
    if out.resolve() == ROOT:
        import build_report
        build_report.README.write_text(build_report.render())
        log("README.md refreshed from the result files")
    return {"stage1": s1, "kernel": kernel, "stage2": s2, "stage3": s3, "propagation": prop}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT, help="output directory (default: repository root)")
    parser.add_argument("--no-figures", action="store_true", help="skip the PNG figures")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers for cross-validation and bootstrap")
    args = parser.parse_args()
    main(args.out, make_figures=not args.no_figures, n_jobs=args.jobs)
