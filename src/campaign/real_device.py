"""The stage 3 circuits on a real IBM quantum processor (extension).

Two steps, because a job on the free plan waits in a queue:

    python scripts/run_ibm_hardware.py submit      transpile for the least busy device, send one job
    python scripts/run_ibm_hardware.py collect     fetch the counts when the job has finished

One job holds five circuits:
  stage3              the six-RY circuit of stage 3
  calibration_zeros   all qubits left in 0, on the same physical qubits  (readout calibration)
  calibration_ones    all qubits flipped to 1, on the same physical qubits
  correlated          the entangled correlated-customers circuit (see correlated.py)
  qaoa_p1             the trained depth-1 QAOA circuit of stage 2

This module holds everything that does not need an account: which circuits are sent and how the
returned counts are turned into estimates. The estimates use the same decoding, the same
readout mitigation and the same exact references as the simulated runs.
"""
import numpy as np
from qiskit import QuantumCircuit

from . import config, correlated, ibm_backends, stage3


def logical_circuits(p, threshold: int, stage2_file: dict, qaoa_runs: dict) -> dict:
    n = len(p)
    zeros, ones = stage3.calibration_circuits(n)
    ham = ibm_backends.ising_from_file(stage2_file)
    return {
        "stage3": stage3.campaign_circuit(p),
        "calibration_zeros": zeros,
        "calibration_ones": ones,
        "correlated": correlated.correlated_circuit(p, config.HARDWARE_RHO, config.FACTOR_QUBITS),
        "qaoa_p1": ibm_backends.trained_circuit(ham, qaoa_runs[1]["gamma"], qaoa_runs[1]["beta"]),
    }


def place_all(circuits: dict, backend) -> tuple:
    """Transpile for the device. The calibration circuits are pinned to the physical qubits of the
    stage 3 circuit. Returns (device circuits, facts per circuit)."""
    seed = config.TRANSPILE_SEEDS[0]
    gate = ibm_backends.two_qubit_gate(backend)
    device = {"stage3": ibm_backends.place(circuits["stage3"], backend, seed)}
    qubits = ibm_backends.physical_qubits(device["stage3"])
    for name in ("calibration_zeros", "calibration_ones"):
        device[name] = ibm_backends.place(circuits[name], backend, seed, layout=qubits)
    for name in ("correlated", "qaoa_p1"):
        device[name], _ = ibm_backends.place_best(circuits[name], backend)
    facts = {
        name: {
            "logical_qubits": int(circuits[name].num_qubits),
            "depth_transpiled": int(circuit.depth()),
            "two_qubit_gate": gate,
            "two_qubit_gates_transpiled": ibm_backends.hardware._count(circuit, [gate]),
            "physical_qubits": sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits
                                       if inst.operation.name != "barrier"}),
        }
        for name, circuit in device.items()
    }
    facts["stage3"]["physical_qubit_of_customer"] = qubits
    return device, facts


def summarise(counts: dict, p, threshold: int, stage2_file: dict) -> dict:
    """`counts`: circuit name -> {bitstring: shots}, keys in Qiskit order (rightmost = qubit 0)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    exact = stage3.exact_at_least(p, threshold)
    raw = stage3.estimate_from_counts(counts["stage3"], n, threshold)
    matrices = stage3.confusion_matrices(counts["calibration_zeros"], counts["calibration_ones"], n)
    quasi = stage3.mitigate(counts["stage3"], matrices)
    mitigated = stage3.quasi_at_least(quasi, n, threshold)
    mitigated_se = stage3.mitigated_standard_error(
        counts["stage3"], counts["calibration_zeros"], counts["calibration_ones"], n, threshold,
        config.MITIGATION_BOOTSTRAP, config.EXT_SEED + 900,
    )
    rho, m = config.HARDWARE_RHO, config.FACTOR_QUBITS
    exact_correlated = correlated.exact_at_least(p, rho, m, threshold)
    raw_correlated = stage3.estimate_from_counts(counts["correlated"], n, threshold)
    optimum = stage2_file["brute_force"]["bitstring_qiskit"]
    k = int(stage2_file["constraints"]["n_select"])
    return {
        "stage3": {
            "exact": exact,
            "shots": raw["shots"],
            "unmitigated": {"estimate": raw["estimate"], "error": raw["estimate"] - exact,
                            "standard_error": raw["standard_error"], "marginals": raw["marginals"]},
            "mitigated": {"estimate": mitigated, "error": mitigated - exact, "standard_error": mitigated_se,
                          "marginals": (quasi @ stage3.all_bitstrings(n)).tolist()},
            "stage1_probabilities": p.tolist(),
            "calibration_p1_given_0": matrices[:, 1, 0].tolist(),
            "calibration_p0_given_1": matrices[:, 0, 1].tolist(),
        },
        "correlated": {
            "rho": float(rho),
            "exact": exact_correlated,
            "shots": raw_correlated["shots"],
            "unmitigated": {"estimate": raw_correlated["estimate"], "error": raw_correlated["estimate"] - exact_correlated,
                            "standard_error": raw_correlated["standard_error"], "marginals": raw_correlated["marginals"]},
            "mitigation": "none: the readout calibration was measured on the stage 3 qubits, and this circuit "
                          "sits on other physical qubits",
            "independent_model_value": exact,
        },
        "qaoa_p1": {
            **ibm_backends.sample_summary(counts["qaoa_p1"], optimum, k),
            "uniform_reference_p_optimal": 1.0 / 2 ** len(optimum),
        },
        "counts": counts,
    }
