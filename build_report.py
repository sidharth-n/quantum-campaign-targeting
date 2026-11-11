"""Fill README.md from report/README.template.md with the numbers in the result files.

    python build_report.py            # rewrite README.md
    python build_report.py --check    # exit 1 if README.md is out of date

Every number in the report is read from a result file, so the text cannot drift from the data.
`run_pipeline.py` calls this as its last step.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "report" / "README.template.md"
README = ROOT / "README.md"

MODEL_NAMES = {
    "logistic": "Logistic regression (classical baseline)",
    "boosting": "Gradient boosting (classical reference)",
    "rbf_kernel": "RBF kernel SVM (classical control)",
    "quantum_kernel": "Quantum kernel SVM (quantum)",
    "hybrid": "Hybrid: logistic + quantum kernel (submitted)",
}


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def load(root: Path) -> dict:
    def js(name):
        return json.loads((root / name).read_text())

    return {
        "s1": pd.read_csv(root / "stage1_predictions.csv"),
        "cv": pd.read_csv(root / "results" / "stage1_cv_metrics.csv"),
        "sel": js("results/stage1_selection.json"),
        "kernel": js("results/stage1_kernel_circuit_check.json"),
        "s2": js("stage2_selection.json"),
        "s2csv": pd.read_csv(root / "stage2_selection.csv"),
        "depth": pd.read_csv(root / "results" / "stage2_qaoa_depth_study.csv"),
        "stress": js("results/stage2_stress_instance.json"),
        "s3": js("stage3_simulation.json"),
        "s3csv": pd.read_csv(root / "stage3_simulation.csv"),
        "tp": pd.read_csv(root / "results" / "transpile_stats.csv"),
        "prop": js("results/error_propagation.json"),
        "models": js("results/stage_comparison_by_model.json"),
        "seeds": js("results/seeds.json"),
        "env": js("results/environment.json"),
        "corr": js("results/extensions/correlated_customers.json"),
        "ae": js("results/extensions/amplitude_estimation.json"),
        "ibm": js("results/extensions/ibm_snapshots.json"),
        "hw": js("results/hardware/ibm_hardware.json") if (root / "results" / "hardware" / "ibm_hardware.json").exists() else None,
    }


def tables(r: dict) -> dict:
    out = {}
    cv = r["cv"][r["cv"]["finalist"]].set_index("model")
    out["cv"] = table(
        ["Model", "Setting", "ROC-AUC", "PR-AUC", "P@10", "R@10", "Brier", "Log loss", "ECE", "Cal. slope"],
        [[MODEL_NAMES[m], {"boosting": "fixed", "hybrid": "50/50 of the tuned logistic regression and quantum kernel SVM"}.get(
            m, str(cv.loc[m, "params"]).replace("'", "")),
          f"{cv.loc[m, 'roc_auc']:.3f} ± {cv.loc[m, 'roc_auc_se']:.3f}",
          f"{cv.loc[m, 'pr_auc']:.3f} ± {cv.loc[m, 'pr_auc_se']:.3f}",
          f"{cv.loc[m, 'precision_at_10']:.3f}", f"{cv.loc[m, 'recall_at_10']:.3f}",
          f"{cv.loc[m, 'brier']:.4f} ± {cv.loc[m, 'brier_se']:.4f}",
          f"{cv.loc[m, 'log_loss']:.4f} ± {cv.loc[m, 'log_loss_se']:.4f}",
          f"{cv.loc[m, 'ece']:.3f}", f"{cv.loc[m, 'calibration_slope']:.2f}"]
         for m in ("logistic", "boosting", "rbf_kernel", "quantum_kernel", "hybrid")],
    )
    comparison = r["sel"]["selection"]["submitted_minus_classical"]
    out["paired"] = table(
        ["Metric", "Hybrid minus logistic regression", "Standard error", "Better for the hybrid?"],
        [[name, f"{comparison[key]['difference']:+.4f}", f"{comparison[key]['standard_error']:.4f}",
          "yes" if (comparison[key]["difference"] > 0) == higher else "no"]
         for key, name, higher in (("roc_auc", "ROC-AUC", True), ("pr_auc", "PR-AUC", True),
                                   ("precision_at_10", "Precision@10", True), ("recall_at_10", "Recall@10", True),
                                   ("brier", "Brier score", False), ("log_loss", "Log loss", False))],
    )
    c = r["s2csv"]
    out["candidates"] = table(
        ["Qubit", "Customer", "Rank", "Probability", "Deposit value", "Cost", "Capacity", "Expected net value", "Selected", "Brute force"],
        [[row.qubit, row.customer_id, row["rank"], f"{row.predicted_probability:.4f}", row.expected_deposit_value,
          row.contact_cost, row.capacity_units, f"{row.expected_net_value:.3f}",
          "yes" if row.selected else "", "yes" if row.brute_force_selected else ""] for _, row in c.iterrows()],
    )
    d = r["depth"]
    out["depth"] = table(
        ["Optimiser", "p", "P(optimum) ideal", "P(optimum) noisy", "Feasible ideal", "Feasible noisy",
         "P(optimum given feasible) ideal", "P(optimum given feasible) noisy", "ECR gates", "SWAPs", "Optimum found (ideal / noisy)"],
        [[row.optimiser, row.layers, f"{row.p_optimal_exact:.4f}", f"{row.p_optimal_noisy_sample:.4f}",
          f"{row.feasible_exact:.3f}", f"{row.feasible_noisy_sample:.3f}",
          f"{row.p_optimal_given_feasible_exact:.4f}", f"{row.p_optimal_given_feasible_noisy_sample:.4f}",
          row.two_qubit_gates_transpiled, row.swap_count,
          f"{'yes' if row.noiseless_best_feasible_is_optimum else 'no'} / {'yes' if row.noisy_best_feasible_is_optimum else 'no'}"]
         for _, row in d.iterrows()],
    )
    out["bitstrings"] = table(
        ["Qiskit key (qubit 0 on the right)", "Counts", "Customers picked", "Cost", "Capacity", "Feasible", "Net value", "Optimum"],
        [[f"`{b['bitstring_qiskit']}`", b["count"], b["n_selected"], f"{b['cost']:.0f}", f"{b['capacity']:.0f}",
          "yes" if b["feasible"] else "no", f"{b['value']:.3f}", "yes" if b["is_exact_optimum"] else ""]
         for b in r["s2"]["qaoa"]["best_bitstrings"][:8]],
    )
    t = r["tp"]
    out["transpile"] = table(
        ["Circuit", "Logical qubits", "Physical qubits used", "Depth original", "Depth transpiled",
         "1q gates original", "2q gates original", "1q gates transpiled", "2q gates (ECR) transpiled", "SWAPs", "Seed"],
        [[row.circuit, row.logical_qubits, row.physical_qubits_used, row.depth_original, row.depth_transpiled,
          row.one_qubit_gates_original,
          row.two_qubit_gates_original if not row.multi_qubit_blocks_original
          else f"{row.two_qubit_gates_original} + {row.multi_qubit_blocks_original} larger blocks",
          row.one_qubit_gates_transpiled, row.two_qubit_gates_transpiled, row.swap_count, row.transpiler_seed]
         for _, row in t.iterrows()],
    )
    e = r["s3"]["estimates"]
    exact = r["s3"]["exact_probability"]

    def line(name, kind, entry, shots="", se=None):
        se = entry.get("standard_error") if se is None else se
        return [name, kind, shots, f"{entry['estimate']:.6f}", f"{entry['estimate'] - exact:+.6f}",
                "" if se is None else f"{se:.6f}"]

    out["stage3"] = table(
        ["Estimate", "Kind", "Shots", "P(at least 3)", "Error against exact", "Standard error"],
        [
            ["Exact Poisson-binomial (dynamic programme)", "classical, exact", "", f"{exact:.6f}", "0", ""],
            line("Statevector of the circuit", "quantum, ideal", e["statevector"]),
            line("**Circuit, noiseless shots (submitted quantum estimate)**", "quantum, finite shots", e["noiseless_shots_submitted"], f"{e['noiseless_shots_submitted']['shots']:,}"),
            line("Circuit, noiseless shots", "quantum, finite shots", e["noiseless_shots"], f"{e['noiseless_shots']['shots']:,}"),
            line("Circuit on the device noise model, unmitigated", "quantum, noisy", e["noisy_unmitigated"], f"{e['noisy_unmitigated']['shots']:,}"),
            line("Same counts, readout mitigated", "quantum, noisy, mitigated", e["noisy_readout_mitigated"], f"{e['noisy_readout_mitigated']['shots']:,}"),
            line("Counter circuit (adder + comparator), noiseless", "quantum, finite shots", e["counter_circuit"], f"{e['counter_circuit']['shots']:,}"),
            ["Counter circuit on the device noise model", "quantum, noisy", f"{e['counter_circuit']['noisy_shots']:,}",
             f"{e['counter_circuit']['noisy_estimate']:.6f}", f"{e['counter_circuit']['noisy_error']:+.6f}", ""],
            line("Monte Carlo", "classical, sampled", e["monte_carlo"], f"{e['monte_carlo']['trials']:,} draws"),
        ],
    )
    q = r["s3csv"]
    out["qubits"] = table(
        ["Qubit", "Customer", "Stage 1 rank", "Stage 1 probability", "RY angle (rad)", "Measured P(1), ideal", "Measured P(1), noisy", "Measured P(1), mitigated"],
        [[row.qubit, row.customer_id, row.stage1_rank, f"{row.predicted_probability:.6f}", f"{row.ry_angle_rad:.6f}",
          f"{row.measured_p1_noiseless:.4f}", f"{row.measured_p1_noisy:.4f}", f"{row.measured_p1_mitigated:.4f}"]
         for _, row in q.iterrows()],
    )
    out["sweep"] = table(
        ["Shots", "Estimate", "Error against exact", "One standard error"],
        [[f"{s['shots']:,}", f"{s['estimate']:.5f}", f"{s['error']:+.5f}", f"{s['standard_error']:.5f}"] for s in r["s3"]["shot_sweep"]],
    )
    p = r["prop"]
    out["top10"] = table(
        ["Candidate", "Share of refits that keep it in the top 10"],
        [[cid, f"{share:.0%}"] for cid, share in p["top10"]["inclusion_frequency"].items()],
    )
    s3p = p["stage3"]
    out["six"] = table(
        ["Customer", "Submitted probability", "5th percentile", "95th percentile", "Standard deviation", "dP(at least 3)/dp"],
        [[cid, f"{s3p['probability_submitted'][cid]:.3f}", f"{s3p['probability_p05'][cid]:.3f}", f"{s3p['probability_p95'][cid]:.3f}",
          f"{s3p['probability_sd'][cid]:.3f}", f"{s3p['sensitivity'][cid]:.3f}"] for cid in s3p["probability_submitted"]],
    )
    names = {"logistic": "Logistic regression", "boosting": "Gradient boosting", "rbf_kernel": "RBF kernel SVM",
             "quantum_kernel": "Quantum kernel SVM", "hybrid": "Hybrid (submitted)"}
    out["models"] = table(
        ["Stage 1 model", "Top 10 shared with submitted", "Six selected (exact Stage 2)", "Shared with submitted six",
         "Net value", "Cost", "Capacity", "P(at least 3), exact"],
        [[names[m["model"]], m["top10_shared_with_submitted"], ", ".join(m["selected_ids"]), m["selected_shared_with_submitted"],
          f"{m['objective_value']:.2f}", f"{m['budget_used']:.0f}", f"{m['capacity_used']:.0f}", f"{m['p_at_least_3']:.4f}"]
         for m in r["models"]],
    )
    seeds = r["seeds"]
    out["seeds"] = table(
        ["Random source", "Seed", "Where"],
        [
            ["Organisers' sampling and train/test split", seeds["organisers_sampling_and_split_seed"], "their workbook"],
            ["Cross-validation folds (5 x 3)", seeds["cv_split_seed"], "`config.SEED`"],
            ["Models (logistic regression, boosting), Platt calibration folds", seeds["model_seed"], "`config.SEED`"],
            ["QAOA start points", seeds["qaoa_start_point_seed"].split(" = ")[1] + " (one per depth)", "`config.QAOA_SEED` + depth"],
            ["Optimiser", "none needed", "COBYLA is deterministic"],
            ["Simulator (every sampled circuit)", seeds["simulator_seed"], "`config.SIM_SEED`"],
            ["Stage 3 shot sweep, one seed per shot count", ", ".join(map(str, seeds["stage3_shot_sweep_seeds"])), "`config.SIM_SEED` + index"],
            ["Readout calibration circuits", ", ".join(map(str, seeds["stage3_calibration_seeds"])), "`config.SIM_SEED` + 101, + 102"],
            ["Transpiler (all tried; fewest physical qubits, then fewest two-qubit gates kept)", ", ".join(map(str, seeds["transpiler_seeds_tried"])), "`config.TRANSPILE_SEEDS`"],
            ["Monte Carlo", seeds["monte_carlo_seed"], "`config.MC_SEED`"],
            ["Bootstrap for the standard error of the mitigated estimate", seeds["monte_carlo_seed"] + 1, "`config.MC_SEED` + 1"],
            ["Bootstrap refits", seeds["bootstrap_seed"], "`config.BOOTSTRAP_SEED`"],
        ],
    )
    return out


def derived(r: dict) -> dict:
    s2, s3, prop, stress = r["s2"], r["s3"], r["prop"], r["stress"]
    q, mixer, uniform = s2["qaoa"], s2["constraint_preserving_qaoa"], s2["qaoa"]["uniform_random_reference"]
    c = r["s2csv"]
    cv = r["cv"][r["cv"]["finalist"]].set_index("model")
    family = r["cv"].groupby("family").size()
    quantum = r["cv"][r["cv"]["family"] == "quantum_kernel"]
    fixed = np.array(prop["samples"]["p_fixed_six"])
    ecr = q["transpilation"]["two_qubit_gates_transpiled"]
    noise = r["seeds"]["noise_model"]
    p = np.array(s3["probabilities"])
    # best campaign of any size inside the two limits (the exactly-K rule dropped)
    bits = ((np.arange(2 ** len(c))[:, None] >> np.arange(len(c))[None, :]) & 1).astype(float)
    inside = (bits @ c["contact_cost"].to_numpy(float) <= s2["constraints"]["budget"]) & \
             (bits @ c["capacity_units"].to_numpy(float) <= s2["constraints"]["capacity"])
    free_values = np.where(inside, bits @ c["expected_net_value"].to_numpy(float), -np.inf)
    free = int(np.argmax(free_values))
    kp = r["sel"]["selection"]["quantum_kernel_params"]
    top_bin = r["sel"]["reliability_out_of_fold"]["hybrid"][-1]
    base = r["sel"]["selection"]["training_positives"] / r["sel"]["selection"]["training_rows"]
    yes = lambda flag: "yes" if flag else "no"
    return {
        "quantum_z": abs(s3["errors"]["quantum_minus_exact"]) / s3["quantum_estimate_standard_error"],
        "mode": int(np.argmax(s3["distribution_of_subscriber_count"])),
        "p_mode": float(np.max(s3["distribution_of_subscriber_count"])),
        "p_fewer_than_3": float(1.0 - s3["exact_probability"]),
        "same_selection_count": int(round(prop["stage2"]["fixed_candidates_same_selection_frequency"] * prop["rounds"])),
        "set_qubits": ", ".join(str(i) for i, ch in enumerate(s2["selection_bitstring"]["x_by_qubit"]) if ch == "1"),
        "qk_setting": f"{kp['reducer'].upper()} reduction, {kp['reps']} feature-map repetition{'s' if kp['reps'] > 1 else ''}, "
                      f"bandwidth {kp['bandwidth']}, C = {kp['C']}",
        "qk_reps": kp["reps"],
        "roc_min": float(cv["roc_auc"].min()), "roc_max": float(cv["roc_auc"].max()),
        "roc_se_min": float(cv["roc_auc_se"].min()), "roc_se_max": float(cv["roc_auc_se"].max()),
        "top_observed": top_bin["observed_rate"], "top_predicted": top_bin["mean_predicted"],
        "top_ratio": top_bin["observed_rate"] / base,
        "free_size": int(bits[free].sum()), "free_value": float(free_values[free]),
        "free_cost": float(bits[free] @ c["contact_cost"].to_numpy(float)),
        "free_capacity": float(bits[free] @ c["capacity_units"].to_numpy(float)),
        "free_ids": ", ".join(c["customer_id"][bits[free] == 1]),
        "n_positive": int((c["expected_net_value"] > 0).sum()),
        "p_lo": float(p.min()), "p_hi": float(p.max()),
        "stress_unbalanced_valid": yes(stress["unbalanced_ground_state_check"]["valid"]),
        "stress_slack_valid": yes(stress["slack_ground_state_check"]["valid"]),
        "stress_mixer_valid": yes(stress["constraint_preserving_ground_state_check"]["valid"]),
        "stress_penalty_found": yes(stress["qaoa_selection_matches_brute_force"]),
        "stress_mixer_found": yes(stress["constraint_preserving_selection_matches_brute_force"]),
        "stress_ids": ", ".join(stress["brute_force"]["ids"]),
        "bf_ids": ", ".join(s2["brute_force"]["selected_ids"]),
        "kernel_pairs": len(r["seeds"]["kernel_check_seeds"]),
        "mixer_feasible_noisy": mixer["noisy_sample"]["feasible_fraction"],
        "logistic_p3": next(m["p_at_least_3"] for m in r["models"] if m["model"] == "logistic"),
        "logistic_shared": next(m["selected_shared_with_submitted"] for m in r["models"] if m["model"] == "logistic"),
        "ids": ", ".join(s2["selected_ids"]),
        "amp_penalty": q["exact_output_distribution"]["p_optimal"] / uniform["p_optimal"],
        "amp_mixer": mixer["exact_output_distribution"]["p_optimal"] / uniform["p_optimal"],
        "amp_penalty_noisy": q["noisy_sample"]["p_optimal"] / uniform["p_optimal"],
        "amp_mixer_noisy": mixer["noisy_sample"]["p_optimal"] / uniform["p_optimal"],
        "subspace_uniform": 1.0 / s2["brute_force"]["selections_of_size_k"],
        "survival": (1.0 - noise["two_qubit_depolarizing"]) ** ecr,
        "expected_subscribers": float(p.sum()),
        "budget_left": s2["constraints"]["budget"] - s2["budget_used"],
        "capacity_left": s2["constraints"]["capacity"] - s2["capacity_used"],
        "all_cost": int(c["contact_cost"].sum()),
        "all_capacity": int(c["capacity_units"].sum()),
        "all_value": float(c["expected_net_value"].sum()),
        "min_value": float(c["expected_net_value"].min()),
        "max_value": float(c["expected_net_value"].max()),
        "value_gap": s2["brute_force"]["objective_value"] - s2["brute_force"]["second_best_objective"],
        "submitted_percentile": float((fixed < prop["stage3"]["submitted_exact"]).mean()),
        "n_logistic": int(family["logistic"]), "n_quantum": int(family["quantum_kernel"]), "n_rbf": int(family["rbf_kernel"]),
        "pca_best_pr": float(quantum[quantum["params"].str.contains("pca")]["pr_auc"].max()),
        "pls_best_pr": float(quantum[quantum["params"].str.contains("pls")]["pr_auc"].max()),
        "base_rate": r["sel"]["selection"]["training_positives"] / r["sel"]["selection"]["training_rows"],
        "p_max": float(r["s1"]["predicted_probability"].max()),
        "p_min": float(r["s1"]["predicted_probability"].min()),
        "p_mean": float(r["s1"]["predicted_probability"].mean()),
        "stress_amp_penalty": stress["qaoa_exact_distribution"]["p_optimal"] / stress["uniform_random_reference"]["p_optimal"],
        "stress_amp_mixer": stress["constraint_preserving_exact_distribution"]["p_optimal"] / stress["uniform_random_reference"]["p_optimal"],
        "shot_se": s3["quantum_estimate_standard_error"],
        "noise_bias": abs(s3["estimates"]["noisy_unmitigated"]["expected_bias"]),
        "model_sd": prop["stage3"]["fixed_six"]["sd"],
        "model_over_shot": prop["stage3"]["fixed_six"]["sd"] / s3["quantum_estimate_standard_error"],
        "crossover": noise["readout_p1_given_0"] / (noise["readout_p1_given_0"] + noise["readout_p0_given_1"]),
        "cv": {m: {k: float(cv.loc[m, k]) for k in ("roc_auc", "pr_auc", "brier", "log_loss", "ece", "calibration_slope",
                                                       "precision_at_10", "recall_at_10")} for m in cv.index},
        "gamma": ", ".join(f"{g:.4f}" for g in q["optimised_parameters"]["gamma"]),
        "beta": ", ".join(f"{b:.4f}" for b in q["optimised_parameters"]["beta"]),
        "mixer_gamma": ", ".join(f"{g:.4f}" for g in mixer["optimised_parameters"]["gamma"]),
        "mixer_beta": ", ".join(f"{b:.4f}" for b in mixer["optimised_parameters"]["beta"]),
        "pmf": ", ".join(f"{v:.4f}" for v in s3["distribution_of_subscriber_count"]),
        "slack_labels": ", ".join(s2["qubo"]["slack_variant"]["labels"][10:]),
    }


def extension_tables(r: dict) -> dict:
    corr, ae, ibm, hw = r["corr"], r["ae"], r["ibm"], r["hw"]
    out = {}
    out["correlated"] = table(
        ["Correlation rho", "Mean pairwise correlation of outcomes", f"P(at least {corr['threshold']}), exact",
         "Entangled circuit, state vector", f"Entangled circuit, {corr['shots']:,} shots", "P(none or one subscribes)",
         "P(campaign loses money)", "Standard deviation of net value", f"Mean net value in the worst {corr['revenue_tail']:.0%}"],
        [[f"{s['rho']:.1f}", f"{s['mean_pairwise_correlation']:.3f}", f"{s['exact']:.4f}", f"{s['statevector']:.4f}",
          f"{s['sampled']['estimate']:.4f} ± {s['sampled']['standard_error']:.4f}", f"{s['probability_none_or_one']:.3f}",
          f"{s['revenue']['probability_of_loss']:.3f}", f"{s['revenue']['standard_deviation']:.1f}",
          f"{s['revenue']['expected_shortfall']:.1f}"] for s in corr["scenarios"]],
    )
    out["amplitude"] = table(
        ["Grover powers used", "Applications of A", "Amplitude estimation, ideal", "Direct sampling, ideal (formula)",
         "Direct / amplitude estimation", "Amplitude estimation, noise model", "Direct sampling, noise model"],
        [[", ".join(map(str, s["grover_powers"])), f"{s['applications_of_A']:,}", f"{s['rmse']:.5f}",
          f"{s['direct_sampling_rmse_measured']:.5f} ({s['direct_sampling_rmse_formula']:.5f})",
          f"{s['error_ratio_direct_over_amplitude_estimation']:.1f}", f"{s['noisy_rmse']:.4f}",
          f"{s['noisy_direct_sampling_rmse_measured']:.4f}"] for s in ae["schedules"]],
    )
    out["amplitude_noisy"] = table(
        ["Grover power", "Two-qubit gates (heavy-hex)", "Depth", "P(good), ideal", "P(good), stated noise model"],
        [[e["power"], f"{e['two_qubit_gates_transpiled']:,}", f"{e['depth_transpiled']:,}", f"{e['good_probability_ideal']:.4f}",
          f"{e['good_probability_noisy']:.4f}"] for e in ae["noisy"]],
    )
    out["ibm_devices"] = table(
        ["Snapshot", "Qubits", "Two-qubit gate", "Median two-qubit error", "Median readout error"],
        [[b["snapshot"], b["qubits"], b["two_qubit_gate"].upper(), f"{b['median_two_qubit_error']:.4f}",
          f"{b['median_readout_error']:.4f}"] for b in ibm["backends"]],
    )
    out["ibm_stage3"] = table(
        ["Snapshot", "Physical qubits (customer 0 to 5)", "Unmitigated", "Error", "Readout mitigated", "Error", "Standard error"],
        [[b["snapshot"], ", ".join(map(str, b["stage3"]["physical_qubits"])), f"{b['stage3']['unmitigated']['estimate']:.6f}",
          f"{b['stage3']['unmitigated']['error']:+.6f}", f"{b['stage3']['mitigated']['estimate']:.6f}",
          f"{b['stage3']['mitigated']['error']:+.6f}", f"{b['stage3']['mitigated']['standard_error']:.6f}"]
         for b in ibm["backends"]],
    )
    sources = [("Stated noise model", ibm["stated_noise_model"]["stage2"])] + [(b["snapshot"], b["stage2"]) for b in ibm["backends"]]
    out["ibm_stage2"] = table(
        ["Noise source", "p", "Depth", "Two-qubit gates", "SWAPs (routing diagnostic)", "P(optimum), ideal", "P(optimum)",
         "P(six selected)", "P(optimum given six selected)"],
        [[name, e["layers"], e["depth_transpiled"], f"{e['two_qubit_gates_transpiled']} {e['two_qubit_gate'].upper()}",
          e["swap_count"], f"{e['p_optimal_ideal']:.4f}", f"{e['p_optimal']:.4f} ± {e['p_optimal_standard_error']:.4f}",
          f"{e['right_size_fraction']:.3f}",
          f"{e['p_optimal_given_right_size']:.4f} ± {e['p_optimal_given_right_size_standard_error']:.4f}"]
         for name, rows in sources for e in rows],
    )
    out["noise_limitation"] = (
        "- **Simulated noise only.** No real hardware. The stated noise model has depolarising and readout errors only; it "
        "has no decoherence during idle time, no crosstalk and no drift. The IBM snapshots of section 11.3 add thermal "
        "relaxation from real calibration data but are still simulations."
        if hw is None else
        f"- **One hardware job.** The run on {hw['backend']} in section 11.4 is a single job. Every other noisy number is a "
        "simulation: the stated noise model has depolarising and readout errors only, and the IBM snapshots of section 11.3 "
        "add thermal relaxation from past calibration data, without crosstalk or drift."
    )
    if hw is None:
        out["hardware"] = (
            "`scripts/run_ibm_hardware.py` sends the Stage 3 circuit, its two readout-calibration circuits, the "
            "correlated-customers circuit and the depth-1 QAOA circuit to a real IBM processor in one job. The returned "
            "counts get the same decoding as the simulated runs, and the Stage 3 circuit gets the same readout mitigation "
            "(`src/campaign/real_device.py`). It needs an IBM Quantum "
            "API key. I did not run it for this submission, so every noisy number in this report is a simulation."
        )
        out["hardware_summary"] = ""
    else:
        s3, cc, qa = hw["stage3"], hw["correlated"], hw["qaoa_p1"]
        rows = [
            ["Stage 3 circuit, unmitigated", f"{s3['unmitigated']['estimate']:.4f}", f"{s3['exact']:.4f}",
             f"{s3['unmitigated']['error']:+.4f}", f"{s3['unmitigated']['standard_error']:.4f}"],
            ["Stage 3 circuit, readout mitigated", f"{s3['mitigated']['estimate']:.4f}", f"{s3['exact']:.4f}",
             f"{s3['mitigated']['error']:+.4f}", f"{s3['mitigated']['standard_error']:.4f}"],
            [f"Correlated-customers circuit (rho = {cc['rho']:.1f}), unmitigated", f"{cc['unmitigated']['estimate']:.4f}",
             f"{cc['exact']:.4f}", f"{cc['unmitigated']['error']:+.4f}", f"{cc['unmitigated']['standard_error']:.4f}"],
        ]
        out["hardware"] = (
            f"I ran the circuits on **{hw['backend']}**, a real {hw['backend_qubits']}-qubit IBM processor "
            f"(job `{hw['job_id']}`, {hw['shots_per_circuit']:,} shots per circuit, raw counts in "
            "`results/hardware/ibm_hardware.json`). One job held the Stage 3 circuit, its two readout-calibration circuits "
            "on the same physical qubits, the correlated-customers circuit and the depth-1 QAOA circuit.\n\n"
            + table(["Circuit on the real device", f"P(at least {r['corr']['threshold']})", "Exact", "Error", "Standard error"], rows)
            + f"\n\nThe Stage 3 circuit ran on physical qubits {', '.join(map(str, hw['circuits']['stage3']['physical_qubit_of_customer']))}. "
            f"The depth-1 QAOA circuit ({hw['circuits']['qaoa_p1']['two_qubit_gates_transpiled']} two-qubit gates after "
            f"transpilation) returned the optimal selection in {qa['optimum_count']} of {qa['shots']:,} shots "
            f"({qa['p_optimal']:.4f}; random bitstrings give {qa['uniform_reference_p_optimal']:.4f}) and a selection of exactly "
            f"six customers in {qa['right_size_fraction']:.1%} of them. This is a single job on one day: it shows the "
            "pipeline runs end to end on hardware, not how the device behaves on average."
        )
        out["hardware_summary"] = (
            f"\n| Does it run on a real quantum computer? | Yes. On IBM's {hw['backend']} the Stage 3 circuit gives "
            f"{s3['mitigated']['estimate']:.3f} after readout mitigation (exact {s3['exact']:.3f}). |"
        )
    return out


def extension_values(r: dict) -> dict:
    corr, ae, ibm = r["corr"], r["ae"], r["ibm"]
    first, last = corr["scenarios"][0], corr["scenarios"][-1]
    best = ae["best_schedule"]
    torino, brisbane = ibm["backends"][0], ibm["backends"][1]
    reference = ibm["uniform_reference"]
    t3, b3, m3 = torino["stage2"][-1], brisbane["stage2"][-1], ibm["stated_noise_model"]["stage2"][-1]
    one_step = ae["noisy"][1]
    deep = ae["noisy"][-1]
    return {
        "rho_max": last["rho"],
        "scenarios": 2 ** corr["factor_qubits"],
        "pair_last": last["mean_pairwise_correlation"],
        "p_first": first["exact"], "p_last": last["exact"],
        "loss_first": first["revenue"]["probability_of_loss"], "loss_last": last["revenue"]["probability_of_loss"],
        "loss_ratio": last["revenue"]["probability_of_loss"] / first["revenue"]["probability_of_loss"],
        "low_first": first["probability_none_or_one"], "low_last": last["probability_none_or_one"],
        "sd_first": first["revenue"]["standard_deviation"], "sd_last": last["revenue"]["standard_deviation"],
        "es_first": first["revenue"]["expected_shortfall"], "es_last": last["revenue"]["expected_shortfall"],
        "expected_net": first["revenue"]["expected_net_value"],
        "expected_subscribers": first["expected_subscribers"],
        "max_statevector_error": max(s["statevector_max_abs_outcome_error"] for s in corr["scenarios"]),
        "max_marginal_deviation": max(s["sampled"]["max_abs_marginal_deviation"] for s in corr["scenarios"]),
        "ae_slope": ae["fitted_slope_log_rmse_vs_log_cost"],
        "ae_direct_slope": ae["fitted_slope_direct_sampling_measured"],
        "ae_points": ae["slope_fit_points"],
        "ae_cost": best["applications_of_A"], "ae_rmse": best["rmse"],
        "ae_direct": best["direct_sampling_rmse_measured"],
        "ae_ratio": best["error_ratio_direct_over_amplitude_estimation"],
        "ae_direct_shots": best["shots_direct_sampling_needs_for_this_rmse"],
        "ae_saving": best["shots_direct_sampling_needs_for_this_rmse"] / best["applications_of_A"],
        "ae_largest_power": best["largest_power"],
        "ae_repeats": best["repeats"], "ae_shots_per_power": best["shots_per_power"],
        "ae_identity": ae["identity_max_abs_difference"],
        "ae_noisy_rmse": best["noisy_rmse"], "ae_noisy_bias": best["noisy_bias"],
        "ae_noisy_direct": best["noisy_direct_sampling_rmse_measured"],
        "ae_noisy_rmse_min": min(s["noisy_rmse"] for s in ae["schedules"]),
        "ae_noisy_rmse_max": max(s["noisy_rmse"] for s in ae["schedules"]),
        "ae_one_step_gates": one_step["two_qubit_gates_transpiled"],
        "ae_one_step_ideal": one_step["good_probability_ideal"], "ae_one_step_noisy": one_step["good_probability_noisy"],
        "ae_deep_power": deep["power"], "ae_deep_noisy": deep["good_probability_noisy"],
        "ae_uniform_bits_readout": ae["uniform_bits_good_probability_after_readout"],
        "ae_deep_low": min(e["good_probability_noisy"] for e in ae["noisy"] if e["power"] >= 4),
        "ae_deep_high": max(e["good_probability_noisy"] for e in ae["noisy"] if e["power"] >= 4),
        "torino": torino, "brisbane": brisbane,
        "t3": t3, "b3": b3, "m3": m3,
        "ideal_p3": t3["p_optimal_ideal"],
        "random_bitstring": reference["p_optimal_random_bitstring"],
        "random_valid": reference["p_optimal_random_valid_selection"],
        "ideal_over_valid": t3["p_optimal_ideal"] / reference["p_optimal_random_valid_selection"],
        "random_right_size": reference["right_size_fraction_random_bitstring"],
        "valid_selections": reference["valid_selections"],
        "torino_over_stated": t3["p_optimal"] / m3["p_optimal"],
        "stated_minus_brisbane_z": (m3["p_optimal"] - b3["p_optimal"])
        / np.sqrt(m3["p_optimal_standard_error"] ** 2 + b3["p_optimal_standard_error"] ** 2),
        "error_ratio_devices": torino["median_two_qubit_error"] / brisbane["median_two_qubit_error"],
        "right_size_low": min(e["right_size_fraction"] for rows in (torino["stage2"], brisbane["stage2"], ibm["stated_noise_model"]["stage2"]) for e in rows),
        "right_size_high": max(e["right_size_fraction"] for rows in (torino["stage2"], brisbane["stage2"], ibm["stated_noise_model"]["stage2"]) for e in rows),
        "ibm_shots": torino["stage3"]["shots"],
        "ibm_qaoa_shots": t3["shots"],
        "runtime_version": ibm["qiskit_ibm_runtime"],
    }


def render(root: Path = ROOT) -> str:
    r = load(root)
    keys = ("s2", "s3", "sel", "kernel", "stress", "prop", "seeds", "env", "corr", "ae", "ibm")
    return TEMPLATE.read_text().format(
        t={**tables(r), **extension_tables(r)}, d=derived(r), x=extension_values(r), **{k: r[k] for k in keys}
    )


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        if not README.exists() or README.read_text() != text:
            print("README.md is out of date: run `python build_report.py`")
            sys.exit(1)
        print("README.md matches the result files")
    else:
        README.write_text(text)
        print(f"README.md written ({len(text.splitlines())} lines)")
