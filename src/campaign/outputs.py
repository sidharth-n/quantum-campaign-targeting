"""Write every result file. The three stage files go to the repository root, details to results/.

All floats in JSON are rounded to 12 significant digits so that reruns are byte-identical.
Customer ids and stage 1 probabilities are written exactly as they are used in stages 2 and 3.
"""
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, qubo

PACKAGES = ["qiskit", "qiskit-aer", "scikit-learn", "numpy", "scipy", "pandas", "openpyxl", "matplotlib", "joblib"]


def clean(obj):
    """Plain, JSON-safe Python objects with stable float formatting."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict(orient="records"))
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if not np.isfinite(value) else float(f"{value:.12g}")
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def seeds() -> dict:
    return {
        "organisers_sampling_and_split_seed": 42,
        "cv_split_seed": config.SEED,
        "model_seed": config.SEED,
        "platt_fold_seed": config.SEED,
        "qaoa_start_point_seed": "config.QAOA_SEED + depth = " + ", ".join(str(config.QAOA_SEED + d) for d in config.QAOA_LAYER_SWEEP),
        "optimiser": "COBYLA is deterministic; its only randomness is the seeded start points",
        "simulator_seed": config.SIM_SEED,
        "stage3_shot_sweep_seeds": [config.SIM_SEED + i for i in range(len(config.SHOT_SWEEP))],
        "stage3_calibration_seeds": [config.SIM_SEED + 101, config.SIM_SEED + 102],
        "kernel_check_seeds": [config.SIM_SEED + i for i in range(config.KERNEL_CHECK_PAIRS)],
        "transpiler_seeds_tried": list(config.TRANSPILE_SEEDS),
        "monte_carlo_seed": config.MC_SEED,
        "bootstrap_seed": config.BOOTSTRAP_SEED,
        "shots": {
            "stage2": config.SHOTS_STAGE2,
            "stage3": config.SHOTS_STAGE3,
            "stage3_submitted_estimate": config.SHOTS_STAGE3_HEADLINE,
            "stage3_counter_noisy": config.SHOTS_COUNTER_NOISY,
            "kernel_check": config.KERNEL_CHECK_SHOTS,
        },
        "monte_carlo_trials": config.MC_TRIALS,
        "bootstrap_rounds": config.BOOTSTRAP_ROUNDS,
        "noise_model": config.NOISE,
        "basis_gates": config.BASIS_GATES,
    }


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
    }


def transpile_row(circuit: str, stats: dict) -> dict:
    keys = [
        "logical_qubits", "physical_qubits_used", "physical_qubits_device", "depth_original", "depth_transpiled",
        "one_qubit_gates_original", "two_qubit_gates_original", "multi_qubit_blocks_original",
        "one_qubit_gates_transpiled", "two_qubit_gates_transpiled", "swap_count", "transpiler_seed",
    ]
    return {"circuit": circuit, **{k: stats[k] for k in keys}}


def _is_optimum(sample: dict, exact: dict) -> bool:
    best = sample["best_feasible"]
    return bool(best is not None and best["bitstring_qiskit"] == qubo.qiskit_key(exact["x"]))


def traceability(out: Path) -> pd.DataFrame:
    """Read the three stage files back from disk and line up every candidate across them.
    Raises if an id or a probability differs anywhere, so inconsistent files are never left behind."""
    stage1 = pd.read_csv(out / "stage1_predictions.csv").set_index("customer_id")
    stage2 = pd.read_csv(out / "stage2_selection.csv").set_index("customer_id")
    stage2_json = json.loads((out / "stage2_selection.json").read_text())
    stage3 = pd.read_csv(out / "stage3_simulation.csv").set_index("customer_id")
    stage3_json = json.loads((out / "stage3_simulation.json").read_text())
    rows = []
    for cid in stage1.index[stage1["is_top_10"] == 1]:
        in3 = cid in stage3.index
        row = {
            "customer_id": cid,
            "stage1_rank": int(stage1.loc[cid, "rank"]),
            "stage1_probability": float(stage1.loc[cid, "predicted_probability"]),
            "stage2_candidate": cid in stage2.index,
            "stage2_qubit": int(stage2.loc[cid, "qubit"]),
            "stage2_probability": float(stage2.loc[cid, "predicted_probability"]),
            "stage2_selected": bool(stage2.loc[cid, "selected"]),
            "stage3_qubit": int(stage3.loc[cid, "qubit"]) if in3 else "",
            "stage3_probability": float(stage3.loc[cid, "predicted_probability"]) if in3 else "",
        }
        row["consistent"] = bool(
            row["stage2_probability"] == row["stage1_probability"]
            and row["stage2_qubit"] == row["stage1_rank"] - 1
            and row["stage2_selected"] == in3
            and (not in3 or row["stage3_probability"] == row["stage1_probability"])
            and row["stage2_selected"] == (cid in stage2_json["selected_ids"])
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    selected = list(table.loc[table["stage2_selected"], "customer_id"])
    problems = []
    if not table["consistent"].all():
        problems.append("an id or probability differs between the stage files")
    if len(table) != len(stage2) or list(stage2.index) != list(table["customer_id"]):
        problems.append("stage 2 candidates are not the stage 1 top 10 in rank order")
    if selected != list(stage3.index) or selected != stage3_json["selected_ids"] or set(selected) != set(stage2_json["selected_ids"]):
        problems.append("stage 3 customers are not the stage 2 selection in rank order")
    if list(stage3["qubit"]) != list(range(len(stage3))) or stage3_json["probabilities"] != list(stage3["predicted_probability"]):
        problems.append("stage 3 qubit order or probabilities differ between its two files")
    if problems:
        raise RuntimeError("traceability check failed: " + "; ".join(problems))
    return table


# --------------------------------------------------------------------------- stage files
def stage2_files(s1, s2, constraints) -> tuple:
    selection, exact = s2["selection"], s2["exact"]
    candidates = s2["candidates"].copy()
    candidates["selected"] = candidates["customer_id"].isin(selection["ids"]).astype(int)
    candidates["brute_force_selected"] = candidates["customer_id"].isin(exact["ids"]).astype(int)
    depth = config.QAOA_LAYERS
    entry = s2["study"][depth]
    mixer = s2["mixer_study"][depth]
    encoding = s2["encoding"]
    tuned = {k: v for k, v in encoding["tuned"].items() if k != "table"}
    payload = {
        "constraints": {
            "n_candidates": constraints.n_candidates, "n_select": constraints.n_select,
            "budget": constraints.budget, "capacity": constraints.capacity,
            "source": "sheet Stage2_Constraints of the organisers' workbook",
        },
        "candidates": candidates,
        "selected_ids": selection["ids"],
        "selected_ids_comma_separated": ",".join(selection["ids"]),
        "budget_used": selection["cost"],
        "capacity_used": selection["capacity"],
        "objective_value": selection["value"],
        "selection_bitstring": {
            "qiskit_key": selection["bitstring_qiskit"],
            "x_by_qubit": selection["x_by_qubit"],
            "convention": "qubit i = candidate with stage 1 rank i + 1; a Qiskit key shows qubit 0 as its rightmost character; x_by_qubit lists qubit 0 first",
            "times_measured": selection["count"],
            "source": selection["source"],
        },
        "brute_force": {
            "selected_ids": exact["ids"],
            "objective_value": exact["value"],
            "budget_used": exact["cost"],
            "capacity_used": exact["capacity"],
            "bitstring_qiskit": qubo.qiskit_key(exact["x"]),
            "selections_of_size_k": exact["n_subsets_of_size_k"],
            "feasible_selections": exact["n_feasible"],
            "second_best_objective": exact["second_best_value"],
        },
        "optimality_gap": selection["optimality_gap"],
        "relative_optimality_gap": selection["relative_gap"],
        "matches_brute_force": selection["matches_brute_force"],
        "constraint_activity": s2["binding"],
        "qubo": {
            "encoding": "unbalanced penalisation, 10 qubits, no auxiliary qubits",
            "energy": "E(x) = -sum v_i x_i + A (sum x_i - K)^2 + sum over {budget, capacity} [ -lambda_1 g(x) + lambda_2 g(x)^2 ], g(x) = limit - sum w_i x_i",
            "penalties": tuned,
            "ground_state_check": encoding["unbalanced_check"],
            "ising_vs_qubo_max_abs_difference": encoding["ising_max_abs_difference"],
            "ising_check_note": "largest |Ising energy - QUBO energy| over all 1024 states, using h, J and the constant exactly as written below",
            "ising": encoding["ising"],
            "slack_variant": {
                "qubits": encoding["slack"].n,
                "auxiliary_qubits": encoding["slack"].n - encoding["slack"].n_decision,
                "labels": list(encoding["slack"].labels),
                "A": config.SLACK_CARDINALITY * tuned["v_max"],
                "mu": config.SLACK_PENALTY * tuned["v_max"],
                "ground_state_check": encoding["slack_check"],
                "feasible_energy_max_abs_error": encoding["slack_feasible_energy_error"],
                "zz_couplings": encoding["slack_couplings"],
            },
            "zz_couplings": encoding["unbalanced_couplings"],
        },
        "qaoa": {
            "layers": depth,
            "optimised_parameters": {"gamma": entry["gamma"], "beta": entry["beta"]},
            "cost_scale": "H_C is the Ising Hamiltonian divided by its largest coefficient",
            "objective": f"CVaR, alpha = {config.QAOA_CVAR_ALPHA}",
            "cvar_objective_value": entry["cvar_objective"],
            "optimiser": f"COBYLA, {config.QAOA_MAXITER} iterations per start, {len(entry['training_runs'])} starts",
            "best_start": entry["best_start_kind"],
            "shots": config.SHOTS_STAGE2,
            "simulator": "qiskit-aer AerSimulator, statevector method",
            "exact_output_distribution": entry["exact_distribution"],
            "noiseless_sample": {k: v for k, v in entry["ideal_sample"].items() if k != "top_bitstrings"},
            "noisy_sample": {k: v for k, v in entry["noisy_sample"].items() if k != "top_bitstrings"},
            "uniform_random_reference": s2["uniform"],
            "best_bitstrings": entry["ideal_sample"]["top_bitstrings"],
            "best_bitstrings_noisy": entry["noisy_sample"]["top_bitstrings"],
            "transpilation": entry["transpile"],
        },
        "constraint_preserving_qaoa": {
            "purpose": "cross-check with a hard cardinality constraint: Dicke initial state and XY mixer, no cardinality penalty",
            "agrees_with_selection": selection["constraint_preserving_qaoa_agrees"],
            "layers": depth,
            "budget_penalty_active": s2["mixer_encoding"]["budget_penalty_active"],
            "capacity_penalty_active": s2["mixer_encoding"]["capacity_penalty_active"],
            "ground_state_check_within_exactly_k_subspace": s2["mixer_encoding"]["check"],
            "zz_couplings": s2["mixer_encoding"]["zz_couplings"],
            "optimised_parameters": {"gamma": mixer["gamma"], "beta": mixer["beta"]},
            "exact_output_distribution": mixer["exact_distribution"],
            "noiseless_sample": {k: v for k, v in mixer["ideal_sample"].items() if k != "top_bitstrings"},
            "noisy_sample": {k: v for k, v in mixer["noisy_sample"].items() if k != "top_bitstrings"},
            "best_bitstrings": mixer["ideal_sample"]["top_bitstrings"],
            "transpilation": mixer["transpile"],
        },
        "stage1_model": s1.selection["submitted_model"],
    }
    return candidates, payload


def stage3_files(s2, s3) -> tuple:
    ranks = s2["candidates"].set_index("customer_id")["rank"]
    table = pd.DataFrame({
        "qubit": np.arange(len(s3["ids"])),
        "customer_id": s3["ids"],
        "stage1_rank": [int(ranks[c]) for c in s3["ids"]],
        "predicted_probability": s3["probabilities"],
        "ry_angle_rad": s3["angles"],
        "measured_p1_noiseless": s3["headline"]["marginals"],
        "measured_p1_noisy": s3["noisy"]["marginals"],
        "measured_p1_mitigated": s3["mitigated"]["marginals"],
    })
    exact = s3["exact"]
    estimates = {
        "exact_poisson_binomial": {"estimate": exact, "error": 0.0},
        "exact_enumeration_check": {"estimate": s3["exact_enumeration"], "error": s3["exact_enumeration"] - exact},
        "statevector": s3["statevector"],
        "noiseless_shots_submitted": {k: v for k, v in s3["headline"].items() if k != "marginals"},
        "noiseless_shots": {k: v for k, v in s3["ideal_shots"].items() if k != "marginals"},
        "noisy_unmitigated": {k: v for k, v in s3["noisy"].items() if "marginals" not in k},
        "noisy_readout_mitigated": {k: v for k, v in s3["mitigated"].items() if k != "marginals"},
        "counter_circuit": {k: v for k, v in s3["counter"].items() if k not in ("counts", "noisy_counts")},
        "monte_carlo": s3["monte_carlo"],
    }
    payload = {
        "selected_ids": s3["ids"],
        "qubit_order": [{"qubit": i, "customer_id": c} for i, c in enumerate(s3["ids"])],
        "qubit_order_rule": "qubit i = i-th selected customer in stage 1 rank order (highest probability first)",
        "bit_order": "Qiskit count keys show qubit 0 as the rightmost character; keys are reversed before use so position i is qubit i",
        "probabilities": s3["probabilities"],
        "circuit_parameters": {
            "gate": "RY(theta_i) on qubit i, theta_i = 2 asin(sqrt(p_i))",
            "theta_rad": s3["angles"],
        },
        "success_condition": f"at least {s3['threshold']} of {len(s3['ids'])} qubits read 1",
        "quantum_estimate": s3["headline"]["estimate"],
        "quantum_estimate_source": f"noiseless Aer sample, {s3['headline']['shots']} shots",
        "quantum_estimate_standard_error": s3["headline"]["standard_error"],
        "exact_probability": exact,
        "monte_carlo_estimate": s3["monte_carlo"]["estimate"],
        "errors": {
            "quantum_minus_exact": s3["headline"]["error"],
            "monte_carlo_minus_exact": s3["monte_carlo"]["error"],
            "statevector_minus_exact": s3["statevector"]["error"],
            "noisy_minus_exact": s3["noisy"]["error"],
            "mitigated_minus_exact": s3["mitigated"]["error"],
        },
        "estimates": estimates,
        "distribution_of_subscriber_count": s3["pmf"],
        "sensitivity_dP_dp": dict(zip(s3["ids"], s3["sensitivity"])),
        "shot_sweep": [{k: v for k, v in e.items() if k != "marginals"} for e in s3["shot_sweep"]],
        "measurement_counts": {
            "noiseless_submitted": s3["counts_headline"],
            "noiseless": s3["counts_ideal"],
            "noisy": s3["counts_noisy"],
            "calibration_all_zeros": s3["counts_calibration_zeros"],
            "calibration_all_ones": s3["counts_calibration_ones"],
            "counter_flag": s3["counter"]["counts"],
            "counter_flag_noisy": s3["counter"]["noisy_counts"],
        },
        "transpilation": {"campaign_circuit": s3["transpile"], "counter_circuit": s3["counter_transpile"]},
        "noise_model": config.NOISE,
    }
    return table, payload


def write_all(out: Path, s1, kernel, s2, s3, prop, comparison, constraints) -> dict:
    results = out / "results"
    # ---------------------------------------------------------------- root files
    write_csv(out / "stage1_predictions.csv", s1.predictions)
    candidates, stage2_payload = stage2_files(s1, s2, constraints)
    write_csv(out / "stage2_selection.csv", candidates)
    write_json(out / "stage2_selection.json", stage2_payload)
    stage3_table, stage3_payload = stage3_files(s2, s3)
    write_csv(out / "stage3_simulation.csv", stage3_table)
    write_json(out / "stage3_simulation.json", stage3_payload)

    # ---------------------------------------------------------------- stage 1 details
    write_csv(results / "stage1_cv_metrics.csv", s1.cv_table.round(6))
    write_csv(results / "stage1_test_predictions_all_models.csv", s1.all_model_predictions)
    write_json(results / "stage1_selection.json", {"selection": s1.selection, "reliability_out_of_fold": s1.reliability})
    write_csv(results / "stage1_kernel_circuit_check.csv", kernel["pairs"].round(6))
    write_json(results / "stage1_kernel_circuit_check.json", {k: v for k, v in kernel.items() if k != "pairs"})

    # ---------------------------------------------------------------- stage 2 details
    write_csv(results / "stage2_penalty_grid.csv", s2["encoding"]["tuned"]["table"].round(8))
    depth_rows = []
    for optimiser, study in (("penalty QAOA", s2["study"]), ("constraint-preserving QAOA", s2["mixer_study"])):
        for depth, entry in study.items():
            depth_rows.append({
                "optimiser": optimiser,
                "layers": depth,
                "p_optimal_exact": entry["exact_distribution"]["p_optimal"],
                "p_optimal_noiseless_sample": entry["ideal_sample"]["p_optimal"],
                "p_optimal_noisy_sample": entry["noisy_sample"]["p_optimal"],
                "p_optimal_given_feasible_exact": entry["exact_distribution"]["p_optimal_given_feasible"],
                "p_optimal_given_feasible_noisy_sample": entry["noisy_sample"]["p_optimal_given_feasible"],
                "feasible_exact": entry["exact_distribution"]["feasible_fraction"],
                "feasible_noiseless_sample": entry["ideal_sample"]["feasible_fraction"],
                "feasible_noisy_sample": entry["noisy_sample"]["feasible_fraction"],
                "value_ratio_exact": entry["exact_distribution"]["mean_value_of_feasible_over_optimum"],
                "value_ratio_noiseless_sample": entry["ideal_sample"]["mean_value_of_feasible_over_optimum"],
                "value_ratio_noisy_sample": entry["noisy_sample"]["mean_value_of_feasible_over_optimum"],
                "noiseless_best_feasible_is_optimum": _is_optimum(entry["ideal_sample"], s2["exact"]),
                "noisy_best_feasible_is_optimum": _is_optimum(entry["noisy_sample"], s2["exact"]),
                "depth_original": entry["transpile"]["depth_original"],
                "depth_transpiled": entry["transpile"]["depth_transpiled"],
                "two_qubit_gates_transpiled": entry["transpile"]["two_qubit_gates_transpiled"],
                "swap_count": entry["transpile"]["swap_count"],
            })
    depth_table = pd.DataFrame(depth_rows)
    write_csv(results / "stage2_qaoa_depth_study.csv", depth_table.round(8))
    write_json(results / "stage2_qaoa_runs.json", {
        name: {str(depth): {k: v for k, v in entry.items() if k not in ("ideal_counts", "noisy_counts")}
               for depth, entry in study.items()}
        for name, study in (("penalty_qaoa", s2["study"]), ("constraint_preserving_qaoa", s2["mixer_study"]))
    })
    write_json(results / "stage2_measurement_counts.json", {
        name: {str(depth): {"noiseless": entry["ideal_counts"], "noisy": entry["noisy_counts"]}
               for depth, entry in study.items()}
        for name, study in (("penalty_qaoa", s2["study"]), ("constraint_preserving_qaoa", s2["mixer_study"]))
    })
    stress = s2["stress"]
    stress_entry = stress["study"][config.QAOA_LAYERS]
    stress_mixer = stress["mixer_study"][config.QAOA_LAYERS]
    stress_mixer_tuned = stress["mixer_encoding"]["tuned"]
    write_json(results / "stage2_stress_instance.json", {
        "purpose": "the organisers' limits do not bind for these candidates, so both limits are tightened to test the constraint handling",
        "rule": (f"each limit targets the smaller of the {config.STRESS_PERCENTILE}th percentile of what the groups of exactly K "
                 "candidates need and one unit less than the optimum uses; the closest pair of limits that leaves "
                 "at least one group feasible and excludes the original optimum is used"),
        "budget": stress["problem"].budget,
        "capacity": stress["problem"].capacity_limit,
        "feasible_selections": stress["exact"]["n_feasible"],
        "brute_force": {"ids": stress["exact"]["ids"], "value": stress["exact"]["value"],
                        "cost": stress["exact"]["cost"], "capacity": stress["exact"]["capacity"]},
        "unbalanced_penalties": {k: v for k, v in stress["encoding"]["tuned"].items() if k != "table"},
        "unbalanced_ground_state_check": stress["encoding"]["unbalanced_check"],
        "slack_qubits": stress["encoding"]["slack"].n,
        "slack_ground_state_check": stress["encoding"]["slack_check"],
        "qaoa_layers": config.QAOA_LAYERS,
        "qaoa_exact_distribution": stress_entry["exact_distribution"],
        "qaoa_noiseless_sample": {k: v for k, v in stress_entry["ideal_sample"].items() if k != "top_bitstrings"},
        "qaoa_selection_matches_brute_force": _is_optimum(stress_entry["ideal_sample"], stress["exact"]),
        "constraint_preserving_penalties": (
            None if stress_mixer_tuned is None else {k: v for k, v in stress_mixer_tuned.items() if k != "table"}
        ),
        "constraint_preserving_ground_state_check": stress["mixer_encoding"]["check"],
        "constraint_preserving_exact_distribution": stress_mixer["exact_distribution"],
        "constraint_preserving_noiseless_sample": {k: v for k, v in stress_mixer["ideal_sample"].items() if k != "top_bitstrings"},
        "constraint_preserving_selection_matches_brute_force": _is_optimum(stress_mixer["ideal_sample"], stress["exact"]),
        "uniform_random_reference": stress["uniform"],
    })

    # ---------------------------------------------------------------- transpilation, propagation, environment
    rows = [transpile_row("stage 1 kernel overlap circuit", kernel["transpile"])]
    rows += [transpile_row(f"stage 2 penalty QAOA p={depth}", entry["transpile"]) for depth, entry in s2["study"].items()]
    rows += [transpile_row(f"stage 2 constraint-preserving QAOA p={depth}", entry["transpile"])
             for depth, entry in s2["mixer_study"].items()]
    rows += [transpile_row("stage 3 campaign circuit (6 RY)", s3["transpile"]),
             transpile_row("stage 3 counter circuit", s3["counter_transpile"])]
    transpile_table = pd.DataFrame(rows)
    write_csv(results / "transpile_stats.csv", transpile_table)
    write_json(results / "error_propagation.json", prop)
    write_json(results / "stage_comparison_by_model.json", comparison)
    write_csv(results / "traceability.csv", traceability(out))
    write_json(results / "seeds.json", seeds())
    write_json(results / "environment.json", environment())
    return {"depth_table": depth_table, "transpile_table": transpile_table}
