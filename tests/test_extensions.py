"""Extensions as a whole: they read the committed stage files, never change them, and rerun to
the same bytes. Also the IBM snapshot study."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import run_extensions
from campaign import config, correlated, hardware, ibm_backends, real_device, stage3

ROOT = Path(__file__).resolve().parents[1]
STAGE_FILES = ["stage1_predictions.csv", "stage2_selection.csv", "stage2_selection.json",
               "stage3_simulation.csv", "stage3_simulation.json"]
EXTENSION_FILES = ["correlated_customers.json", "amplitude_estimation.json", "ibm_snapshots.json"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def given():
    return run_extensions.inputs(ROOT)


@pytest.fixture(scope="module")
def ibm_file():
    return json.loads((ROOT / "results" / "extensions" / "ibm_snapshots.json").read_text())


def test_inputs_are_the_submitted_six_with_the_stage1_probabilities(given, stage1_file, stage2_json, stage3_json):
    assert given["ids"] == stage3_json["selected_ids"]
    assert sorted(given["ids"]) == sorted(stage2_json["selected_ids"])
    by_id = stage1_file.set_index("customer_id")["predicted_probability"]
    for cid, p in zip(given["ids"], given["probabilities"]):
        assert p == pytest.approx(by_id[cid], abs=1e-12)
    assert given["threshold"] == 3
    assert len(given["deposits"]) == len(given["costs"]) == 6


@pytest.mark.slow
def test_rerun_is_byte_identical_and_leaves_the_stage_files_alone(tmp_path):
    before = {name: digest(ROOT / name) for name in STAGE_FILES}
    run_extensions.main(ROOT, tmp_path, make_figures=False)
    assert {name: digest(ROOT / name) for name in STAGE_FILES} == before
    assert not any((tmp_path / name).exists() for name in STAGE_FILES)
    for name in EXTENSION_FILES:
        fresh = tmp_path / "results" / "extensions" / name
        assert fresh.read_bytes() == (ROOT / "results" / "extensions" / name).read_bytes(), name


# --------------------------------------------------------------------------- IBM snapshots
def test_rebuilt_qaoa_circuit_is_the_one_stage2_trained(given):
    ham = ibm_backends.ising_from_file(given["stage2"])
    optimum = int(given["stage2"]["brute_force"]["bitstring_qiskit"], 2)
    for depth in config.IBM_QAOA_LAYERS:
        run = given["qaoa_runs"][depth]
        circuit = ibm_backends.trained_circuit(ham, run["gamma"], run["beta"])
        circuit.remove_final_measurements()
        assert Statevector(circuit).probabilities()[optimum] == pytest.approx(run["p_optimal_ideal"], abs=1e-9)


def test_sample_summary_counts_the_optimum_and_the_right_size():
    counts = {"0000111111": 5, "1000101111": 3, "0000000001": 2}
    summary = ibm_backends.sample_summary(counts, "1000101111", 6)
    assert summary["shots"] == 10
    assert summary["p_optimal"] == pytest.approx(0.3)
    assert summary["right_size_fraction"] == pytest.approx(0.8)
    assert summary["p_optimal_given_right_size"] == pytest.approx(3 / 8)


@pytest.mark.parametrize("name", config.IBM_SNAPSHOTS)
def test_calibration_circuits_sit_on_the_qubits_of_the_stage3_circuit(name, given):
    backend = ibm_backends.snapshot(name)
    seed = config.TRANSPILE_SEEDS[0]
    device = ibm_backends.place(stage3.campaign_circuit(given["probabilities"]), backend, seed)
    qubits = ibm_backends.physical_qubits(device)
    assert len(set(qubits)) == 6
    for circuit in stage3.calibration_circuits(6):
        assert ibm_backends.physical_qubits(ibm_backends.place(circuit, backend, seed, layout=qubits)) == qubits
    assert ibm_backends._two_qubit_free(device, backend)


def test_committed_snapshot_study_is_consistent(ibm_file, stage3_json):
    assert [b["snapshot"] for b in ibm_file["backends"]] == list(config.IBM_SNAPSHOTS)
    reference = ibm_file["uniform_reference"]
    assert reference["p_optimal_random_bitstring"] == pytest.approx(1 / 1024)
    assert reference["p_optimal_random_valid_selection"] == pytest.approx(1 / 210)
    assert reference["right_size_fraction_random_bitstring"] == pytest.approx(210 / 1024)
    for backend in ibm_file["backends"]:
        s3 = backend["stage3"]
        assert s3["exact"] == pytest.approx(stage3_json["exact_probability"], abs=1e-11)
        assert s3["calibration_on_same_qubits"] is True
        assert s3["two_qubit_gates_transpiled"] == 0
        assert sum(s3["counts"].values()) == config.SHOTS_IBM
        assert abs(s3["mitigated"]["error"]) < s3["mitigated"]["standard_error"]      # the claim made in the report
        assert abs(s3["mitigated"]["error"]) < abs(s3["unmitigated"]["error"])
        assert 0.003 < s3["unmitigated"]["error"] < 0.005                             # "about 0.004"
        np.testing.assert_allclose(s3["mitigated"]["marginals"], stage3_json["probabilities"], atol=0.02)
    for rows in [ibm_file["stated_noise_model"]["stage2"]] + [b["stage2"] for b in ibm_file["backends"]]:
        assert [e["layers"] for e in rows] == list(config.IBM_QAOA_LAYERS)
        for entry in rows:
            assert entry["shots"] == config.SHOTS_IBM_QAOA
            assert entry["physical_qubits_used"] == 10
            assert entry["two_qubit_gates_transpiled"] > 0
            assert 0.0 <= entry["p_optimal"] <= entry["right_size_fraction"] <= 1.0
            assert entry["p_optimal"] < entry["p_optimal_ideal"]
            assert entry["p_optimal_given_right_size"] == pytest.approx(entry["p_optimal"] / entry["right_size_fraction"])
            assert entry["p_optimal_standard_error"] == pytest.approx(
                np.sqrt(entry["p_optimal"] * (1 - entry["p_optimal"]) / entry["shots"]))


def test_report_claims_about_the_snapshots_hold(ibm_file):
    """Each sentence of README section 11.3 that compares noise sources, as an inequality with its margin."""
    reference = ibm_file["uniform_reference"]
    torino, brisbane = ibm_file["backends"]
    stated = ibm_file["stated_noise_model"]["stage2"]
    t3, b3, m3 = torino["stage2"][-1], brisbane["stage2"][-1], stated[-1]

    def sigmas(a, b):
        return (a["p_optimal"] - b["p_optimal"]) / np.hypot(a["p_optimal_standard_error"], b["p_optimal_standard_error"])

    assert torino["median_two_qubit_error"] < 0.65 * brisbane["median_two_qubit_error"]
    assert sigmas(t3, m3) > 5 and sigmas(t3, b3) > 5                       # Torino is clearly the best of the three
    assert abs(sigmas(m3, b3)) < 3                                         # the stated model sits next to Brisbane
    valid = reference["p_optimal_random_valid_selection"]
    assert (t3["p_optimal"] - valid) / t3["p_optimal_standard_error"] > 5  # only Torino at p = 3 beats a random valid selection
    for rows in (stated, brisbane["stage2"]):
        assert all(e["p_optimal"] + 3 * e["p_optimal_standard_error"] < valid for e in rows)
    assert all(e["p_optimal"] < valid for e in torino["stage2"][:-1])
    for entry in (t3, b3, m3):                                             # among shots that select six, all beat it at p = 3
        assert (entry["p_optimal_given_right_size"] - valid) / entry["p_optimal_given_right_size_standard_error"] > 5
    fractions = [e["right_size_fraction"] for rows in (stated, torino["stage2"], brisbane["stage2"]) for e in rows]
    assert 0.25 < min(fractions) and max(fractions) < 0.45
    assert min(fractions) > reference["right_size_fraction_random_bitstring"]


def test_swap_diagnostic_starts_from_the_qubits_of_the_circuit_that_is_run(given):
    backend = ibm_backends.snapshot(config.IBM_SNAPSHOTS[0])
    ham = ibm_backends.ising_from_file(given["stage2"])
    run = given["qaoa_runs"][1]
    device, stats = ibm_backends.place_best(ibm_backends.trained_circuit(ham, run["gamma"], run["beta"]), backend)
    assert stats["starting_physical_qubits"] == [int(q) for q in device.layout.initial_index_layout(filter_ancillas=True)]
    assert len(set(stats["starting_physical_qubits"])) == 10
    assert stats["swap_count"] > 0


# --------------------------------------------------------------------------- real device post-processing
def test_hardware_post_processing_recovers_the_exact_values_from_ideal_counts(given):
    """The code that will read real device counts, fed with noiseless simulator counts."""
    p, threshold = given["probabilities"], given["threshold"]
    circuits = real_device.logical_circuits(p, threshold, given["stage2"], given["qaoa_runs"])
    assert list(circuits) == ["stage3", "calibration_zeros", "calibration_ones", "correlated", "qaoa_p1"]
    sim = hardware.ideal_simulator()
    counts = {
        name: hardware.run_counts(sim, hardware.to_device(circuit, config.TRANSPILE_SEEDS[0]), 20000, seed=7 + index)
        for index, (name, circuit) in enumerate(circuits.items())
    }
    assert counts["calibration_zeros"] == {"000000": 20000} and counts["calibration_ones"] == {"111111": 20000}
    summary = real_device.summarise(counts, p, threshold, given["stage2"])
    s3 = summary["stage3"]
    assert s3["exact"] == pytest.approx(stage3.exact_at_least(p, threshold), abs=1e-12)
    assert s3["mitigated"]["estimate"] == pytest.approx(s3["unmitigated"]["estimate"], abs=1e-12)   # perfect readout
    assert abs(s3["unmitigated"]["error"]) < 4 * s3["unmitigated"]["standard_error"]
    np.testing.assert_allclose(s3["unmitigated"]["marginals"], p, atol=0.02)
    cc = summary["correlated"]
    assert cc["exact"] == pytest.approx(correlated.exact_at_least(p, config.HARDWARE_RHO, config.FACTOR_QUBITS, threshold), abs=1e-12)
    assert abs(cc["unmitigated"]["error"]) < 4 * cc["unmitigated"]["standard_error"]
    assert cc["exact"] > cc["independent_model_value"]
    assert "estimate" not in str(cc.get("mitigation"))                # no mitigated number without its own calibration
    qa = summary["qaoa_p1"]
    ideal = given["qaoa_runs"][1]["p_optimal_ideal"]
    assert abs(qa["p_optimal"] - ideal) < 4 * np.sqrt(ideal * (1 - ideal) / qa["shots"])
