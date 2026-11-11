"""The three stage files: validity, traceability of customer ids and probabilities, and internal consistency.

Everything is recomputed from the stage 1 file and the organisers' workbook, not from pipeline objects."""
import numpy as np
import pytest

from campaign import config, qubo, stage3


# --------------------------------------------------------------------------- stage 2
def test_stage2_candidates_are_the_stage1_top_10(stage1_file, stage2_csv, stage2_json):
    top = stage1_file[stage1_file["is_top_10"] == 1].sort_values("rank")
    assert list(stage2_csv["customer_id"]) == list(top["customer_id"])
    assert list(stage2_csv["predicted_probability"]) == list(top["predicted_probability"])     # exact equality
    assert list(stage2_csv["qubit"]) == list(range(10))
    assert [c["customer_id"] for c in stage2_json["candidates"]] == list(top["customer_id"])
    assert [c["predicted_probability"] for c in stage2_json["candidates"]] == list(top["predicted_probability"])


def test_stage2_inputs_are_the_workbook_values(data, stage2_csv):
    ops = data.ops.loc[stage2_csv["customer_id"]]
    for column in ("contact_cost", "expected_deposit_value", "capacity_units"):
        assert list(stage2_csv[column]) == list(ops[column])
    value = stage2_csv["predicted_probability"] * stage2_csv["expected_deposit_value"] - stage2_csv["contact_cost"]
    np.testing.assert_allclose(stage2_csv["expected_net_value"], value, rtol=0, atol=1e-12)


def test_stage2_selection_is_valid(data, stage1_file, stage2_json):
    ids = stage2_json["selected_ids"]
    c = data.constraints
    assert len(ids) == c.n_select == 6 and len(set(ids)) == 6
    assert set(ids) <= set(stage1_file.loc[stage1_file["is_top_10"] == 1, "customer_id"])
    cost = data.ops.loc[ids, "contact_cost"].sum()
    capacity = data.ops.loc[ids, "capacity_units"].sum()
    assert cost <= c.budget and capacity <= c.capacity
    assert stage2_json["budget_used"] == cost and stage2_json["capacity_used"] == capacity
    assert stage2_json["constraints"] == {
        "n_candidates": c.n_candidates, "n_select": c.n_select, "budget": c.budget, "capacity": c.capacity,
        "source": stage2_json["constraints"]["source"],
    }
    assert stage2_json["selected_ids_comma_separated"] == ",".join(ids)


def test_stage2_objective_and_gap_recomputed_from_scratch(problem, stage2_json):
    exact = qubo.brute_force(problem)
    ids = stage2_json["selected_ids"]
    x = np.array([cid in ids for cid in problem.ids], dtype=float)
    assert stage2_json["objective_value"] == pytest.approx(float(x @ problem.value), abs=1e-9)
    assert stage2_json["brute_force"]["objective_value"] == pytest.approx(exact["value"], abs=1e-9)
    assert stage2_json["brute_force"]["selected_ids"] == exact["ids"]
    assert stage2_json["optimality_gap"] == pytest.approx(exact["value"] - float(x @ problem.value), abs=1e-9)
    assert stage2_json["optimality_gap"] >= 0
    if set(ids) == set(exact["ids"]):
        assert stage2_json["optimality_gap"] == 0.0
    assert stage2_json["matches_brute_force"] == (set(ids) == set(exact["ids"]))


def test_stage2_bitstring_decodes_to_the_selected_ids(problem, stage2_json, stage2_csv):
    key = stage2_json["selection_bitstring"]["qiskit_key"]
    x = qubo.key_to_bits(key)
    assert [problem.ids[i] for i in np.flatnonzero(x)] == stage2_json["selected_ids"]
    assert stage2_json["selection_bitstring"]["x_by_qubit"] == "".join(map(str, x))
    assert list(stage2_csv["selected"]) == x.tolist()
    assert list(stage2_csv.loc[stage2_csv["selected"] == 1, "customer_id"]) == stage2_json["selected_ids"]


def test_stage2_reported_bitstrings_are_consistent(problem, stage2_json):
    for row in stage2_json["qaoa"]["best_bitstrings"] + stage2_json["constraint_preserving_qaoa"]["best_bitstrings"]:
        x = qubo.key_to_bits(row["bitstring_qiskit"])
        ev = qubo.evaluate(problem, x[None, :])
        assert row["ids"] == [problem.ids[i] for i in np.flatnonzero(x)]
        assert row["feasible"] == bool(ev["feasible"][0])
        assert row["value"] == pytest.approx(float(ev["value"][0]), abs=1e-9)
        assert row["cost"] == ev["cost"][0] and row["capacity"] == ev["capacity"][0]
    counts = [row["count"] for row in stage2_json["qaoa"]["best_bitstrings"]]
    assert counts == sorted(counts, reverse=True)


def test_stage2_qubo_ground_state_check_recorded_as_valid(stage2_json):
    assert stage2_json["qubo"]["ground_state_check"]["valid"] is True
    assert stage2_json["qubo"]["slack_variant"]["ground_state_check"]["valid"] is True
    assert stage2_json["qubo"]["ising_vs_qubo_max_abs_difference"] < 1e-8
    assert stage2_json["constraint_preserving_qaoa"]["agrees_with_selection"] is True
    assert len(stage2_json["qaoa"]["optimised_parameters"]["gamma"]) == config.QAOA_LAYERS


# --------------------------------------------------------------------------- stage 3 and traceability
def test_same_ids_and_probabilities_in_all_three_files(stage1_file, stage2_csv, stage2_json, stage3_csv, stage3_json):
    stage1 = dict(zip(stage1_file["customer_id"], stage1_file["predicted_probability"]))
    # stage 2 -> stage 3: the same six, in stage 1 rank order
    chosen = stage2_csv[stage2_csv["selected"] == 1].sort_values("rank")
    assert list(stage3_csv["customer_id"]) == list(chosen["customer_id"]) == stage3_json["selected_ids"]
    assert set(stage3_json["selected_ids"]) == set(stage2_json["selected_ids"])
    assert list(stage3_csv["qubit"]) == list(range(6))
    assert list(stage3_csv["stage1_rank"]) == list(chosen["rank"])
    assert [q["customer_id"] for q in stage3_json["qubit_order"]] == list(stage3_csv["customer_id"])
    # the probabilities are bit-for-bit the stage 1 values
    for cid, p_csv, p_json in zip(stage3_csv["customer_id"], stage3_csv["predicted_probability"], stage3_json["probabilities"]):
        assert p_csv == stage1[cid] and p_json == stage1[cid]
    for cid, p in zip(stage2_csv["customer_id"], stage2_csv["predicted_probability"]):
        assert p == stage1[cid]


def test_stage3_exact_value_recomputed_from_the_stage1_file(stage1_file, stage3_json):
    stage1 = dict(zip(stage1_file["customer_id"], stage1_file["predicted_probability"]))
    p = [stage1[cid] for cid in stage3_json["selected_ids"]]
    exact = stage3.exact_at_least(p, 3)
    assert stage3_json["exact_probability"] == pytest.approx(exact, abs=1e-11)
    assert stage3_json["estimates"]["exact_enumeration_check"]["estimate"] == pytest.approx(exact, abs=1e-11)
    assert stage3_json["estimates"]["statevector"]["estimate"] == pytest.approx(exact, abs=1e-11)
    np.testing.assert_allclose(stage3_json["circuit_parameters"]["theta_rad"], stage3.angles(p), atol=1e-11)


def test_stage3_quantum_estimate_recomputed_from_the_recorded_counts(stage3_json):
    counts = stage3_json["measurement_counts"]["noiseless_submitted"]
    assert sum(counts.values()) == config.SHOTS_STAGE3_HEADLINE
    estimate = stage3.estimate_from_counts(counts, 6, 3)
    assert stage3_json["quantum_estimate"] == pytest.approx(estimate["estimate"], abs=1e-12)
    assert stage3_json["errors"]["quantum_minus_exact"] == pytest.approx(
        stage3_json["quantum_estimate"] - stage3_json["exact_probability"], abs=1e-11)
    # the estimate must sit within sampling error of the exact value
    assert abs(stage3_json["errors"]["quantum_minus_exact"]) < 4 * stage3_json["quantum_estimate_standard_error"]
    noisy = stage3.estimate_from_counts(stage3_json["measurement_counts"]["noisy"], 6, 3)
    assert stage3_json["estimates"]["noisy_unmitigated"]["estimate"] == pytest.approx(noisy["estimate"], abs=1e-12)


def test_stage3_measured_marginals_sit_on_the_right_customers(stage3_csv):
    p = stage3_csv["predicted_probability"].to_numpy()
    measured = stage3_csv["measured_p1_noiseless"].to_numpy()
    assert (np.abs(measured - p) < 5 * np.sqrt(p * (1 - p) / config.SHOTS_STAGE3_HEADLINE)).all()


def test_stage3_monte_carlo_and_reference_values(stage3_json):
    mc = stage3_json["estimates"]["monte_carlo"]
    assert mc["trials"] == config.MC_TRIALS and mc["seed"] == config.MC_SEED
    assert abs(mc["error"]) < 4 * mc["standard_error"]
    assert stage3_json["estimates"]["counter_circuit"]["statevector_abs_error"] < 1e-9
    assert sum(stage3_json["distribution_of_subscriber_count"]) == pytest.approx(1.0, abs=1e-11)


def test_near_tie_goes_to_the_higher_value_not_the_more_frequent_bitstring(data):
    """Two feasible bitstrings whose values differ by 2e-13: the better one must win even with fewer counts."""
    import pandas as pd
    from campaign import stage2

    frame = pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "predicted_probability": [0.5, 0.5, 0.5],
        "expected_deposit_value": [4.0, 4.0000000000004, 2.0],
        "contact_cost": [1.0, 1.0, 1.0],
        "capacity_units": [1.0, 1.0, 1.0],
    })
    problem = qubo.make_problem(frame, 1, 10, 10)
    assert 0 < problem.value[1] - problem.value[0] < 1e-12
    exact = qubo.brute_force(problem)
    assert exact["ids"] == ["B"]
    model = qubo.unbalanced_qubo(problem, 5.0, 0.0, 0.0)
    sample = stage2.sample_metrics(problem, exact, model.energies(), {"001": 900, "010": 100})
    assert sample["best_feasible"]["ids"] == ["B"]
    assert stage2.describe(problem, exact["x"])["value"] - sample["best_feasible"]["value"] == 0.0


def test_mitigated_estimate_carries_a_standard_error(stage3_json):
    mitigated = stage3_json["estimates"]["noisy_readout_mitigated"]
    plain = stage3_json["estimates"]["noisy_unmitigated"]["standard_error"]
    assert mitigated["standard_error"] >= plain * 0.9          # calibration can only add uncertainty
    assert mitigated["standard_error"] < 5 * plain
    assert abs(mitigated["error"]) < 4 * mitigated["standard_error"]


def test_published_ising_coefficients_reproduce_the_qubo(problem, stage2_json):
    """Rebuild the energies from h, J and the constant exactly as written in the result file."""
    ising = stage2_json["qubo"]["ising"]
    penalties = stage2_json["qubo"]["penalties"]
    model = qubo.unbalanced_qubo(problem, penalties["A"], penalties["lambda_1"], penalties["lambda_2"])
    energies = qubo.ising_energies(np.array(ising["h"]), np.array(ising["J"]), ising["constant"])
    assert np.abs(energies - model.energies()).max() < 1e-8
    assert stage2_json["qubo"]["ising_vs_qubo_max_abs_difference"] < 1e-8
    assert int(np.argmin(energies)) == qubo.brute_force(problem)["index"]


def test_traceability_file_lines_up_all_three_stages(stage1_file, stage2_json, stage3_json):
    import pandas as pd
    from pathlib import Path

    table = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "traceability.csv")
    assert len(table) == 10 and table["consistent"].all()
    assert list(table["customer_id"]) == list(stage1_file.sort_values("rank").head(10)["customer_id"])
    assert list(table.loc[table["stage2_selected"], "customer_id"]) == stage3_json["selected_ids"]
    assert set(stage3_json["selected_ids"]) == set(stage2_json["selected_ids"])


def test_traceability_check_refuses_inconsistent_files(tmp_path, stage1_file):
    """Corrupt one probability in a copy of the stage 2 file: the check must raise."""
    import shutil
    from pathlib import Path
    import pandas as pd
    from campaign import outputs

    root = Path(__file__).resolve().parents[1]
    for name in ("stage1_predictions.csv", "stage2_selection.csv", "stage2_selection.json",
                 "stage3_simulation.csv", "stage3_simulation.json"):
        shutil.copy(root / name, tmp_path / name)
    assert outputs.traceability(tmp_path)["consistent"].all()
    broken = pd.read_csv(tmp_path / "stage2_selection.csv")
    broken.loc[0, "predicted_probability"] += 1e-9
    broken.to_csv(tmp_path / "stage2_selection.csv", index=False)
    with pytest.raises(RuntimeError, match="traceability"):
        outputs.traceability(tmp_path)


def test_readme_numbers_match_the_result_files():
    """The report is generated from the result files; it must be up to date."""
    from pathlib import Path
    import build_report

    assert (Path(build_report.README)).read_text() == build_report.render()
