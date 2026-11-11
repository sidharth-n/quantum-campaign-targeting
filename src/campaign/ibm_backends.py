"""The stage 2 and stage 3 circuits on IBM device snapshots (extension).

The main pipeline uses a noise model with four stated numbers. That keeps the assumptions in
plain sight, but it is not any particular machine. Here the same circuits are transpiled to two
real IBM processors and simulated with their published calibration data:

  FakeTorino     133 qubits, Heron r1, CZ gates, heavy-hex
  FakeBrisbane   127 qubits, Eagle r3, ECR gates, heavy-hex

These are the snapshot backends shipped with qiskit-ibm-runtime. Each one carries the coupling
map, the basis gates and a calibration record of a real device (gate errors, readout errors, T1,
T2, gate times), and `AerSimulator.from_backend` turns that record into a noise model with
thermal relaxation. A snapshot is a past calibration of the device, not its state today, and the
simulation leaves out crosstalk and drift. Nothing here contacts IBM.

Every circuit here touches at most ten physical qubits, so the simulator evolves the exact
density matrix of the noisy circuit and the shots are drawn from it. The only statistical error
is the shot noise, which is why the QAOA runs can afford 2^17 shots. The stated noise model of
the main pipeline is rerun here with the same shot count so that the three noise sources are
compared like for like.

What is measured
----------------
stage 3   the six-RY circuit: unmitigated and readout-mitigated estimate of P(at least threshold),
          with the calibration circuits placed on the same physical qubits
stage 2   the trained QAOA circuit at each depth: physical qubits, depth, two-qubit gates, SWAPs,
          and how often the noisy samples return the optimum or a selection of the right size
"""
from importlib import metadata
from math import comb

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from . import config, hardware, qaoa, stage3


def snapshot(name: str):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def two_qubit_gate(backend) -> str:
    return next(name for name in ("cz", "ecr", "cx") if name in backend.target.operation_names)


def describe(backend) -> dict:
    """Size, native gates and median calibration errors of the snapshot."""
    target = backend.target
    gate = two_qubit_gate(backend)
    gate_errors = [props.error for props in target[gate].values() if props is not None and props.error is not None]
    readout = [props.error for props in target["measure"].values() if props is not None and props.error is not None]
    return {
        "name": backend.name,
        "qubits": int(backend.num_qubits),
        "two_qubit_gate": gate,
        "basis_gates": sorted(name for name in target.operation_names if name in ("cz", "ecr", "cx", "rz", "sx", "x", "id")),
        "median_two_qubit_error": float(np.median(gate_errors)),
        "median_readout_error": float(np.median(readout)),
    }


def simulator(backend) -> AerSimulator:
    """Exact density matrix of the noisy circuit (Aer drops the idle qubits of the device)."""
    return AerSimulator.from_backend(backend, method="density_matrix", seed_simulator=config.EXT_SEED)


def place(circuit: QuantumCircuit, backend, seed: int, layout=None) -> QuantumCircuit:
    return transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=seed, initial_layout=layout)


def place_best(circuit: QuantumCircuit, backend, seeds=config.TRANSPILE_SEEDS):
    """Same rule as hardware.transpile_best: fewest physical qubits, then fewest two-qubit gates,
    then lowest depth, then lowest seed."""
    gate = two_qubit_gate(backend)
    tried = []
    for seed in seeds:
        candidate = place(circuit, backend, seed)
        tried.append((hardware.active_qubits(candidate), hardware._count(candidate, [gate]), candidate.depth(), seed, candidate))
    used, two_qubit, depth, seed, best = min(tried, key=lambda t: t[:4])
    if used > config.DENSITY_MATRIX_MAX_QUBITS:
        raise RuntimeError(f"the circuit touches {used} physical qubits, too many for a density matrix")
    # routing diagnostic: a second compilation from the same starting qubits and seed that keeps SWAP
    # as a gate. The circuit that is run has its SWAPs merged into native gates, so this count is
    # reported next to the native gate count, not derived from it.
    start = [int(q) for q in best.layout.initial_index_layout(filter_ancillas=True)]
    routed = transpile(
        circuit, coupling_map=backend.coupling_map, basis_gates=list(backend.target.operation_names) + ["swap"],
        initial_layout=start, optimization_level=3, seed_transpiler=seed,
    )
    stats = {
        "logical_qubits": int(circuit.num_qubits),
        "physical_qubits_device": int(backend.num_qubits),
        "physical_qubits_used": int(used),
        "depth_original": int(circuit.depth()),
        "depth_transpiled": int(depth),
        "two_qubit_gate": gate,
        "two_qubit_gates_transpiled": int(two_qubit),
        "one_qubit_gates_transpiled": hardware._count(best, ("sx", "x", "rz")),
        "swap_count": hardware._count(routed, ["swap"]),
        "swap_count_source": "separate compilation from the same starting qubits and seed, SWAP kept as a gate",
        "starting_physical_qubits": start,
        "transpiler_seed": int(seed),
        "optimization_level": 3,
    }
    return best, stats


def physical_qubits(circuit: QuantumCircuit) -> list:
    """Physical qubit that holds logical qubit i at the end of the circuit."""
    return [int(q) for q in circuit.layout.final_index_layout()]


def _two_qubit_free(circuit: QuantumCircuit, backend) -> bool:
    return hardware._count(circuit, [two_qubit_gate(backend)]) == 0


def run_counts(sim: AerSimulator, circuit: QuantumCircuit, shots: int, seed: int) -> dict:
    counts = sim.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()
    return {key: int(counts[key]) for key in sorted(counts)}


# --------------------------------------------------------------------------- stage 3
def stage3_on(backend, p, threshold: int, seed_offset: int) -> dict:
    p = np.asarray(p, dtype=float)
    n = len(p)
    exact = stage3.exact_at_least(p, threshold)
    sim = simulator(backend)
    seed = config.TRANSPILE_SEEDS[0]
    device = place(stage3.campaign_circuit(p), backend, seed)
    qubits = physical_qubits(device)
    zeros, ones = (place(c, backend, seed, layout=qubits) for c in stage3.calibration_circuits(n))
    shots = config.SHOTS_IBM
    counts = run_counts(sim, device, shots, config.EXT_SEED + seed_offset)
    counts_zeros = run_counts(sim, zeros, shots, config.EXT_SEED + seed_offset + 1)
    counts_ones = run_counts(sim, ones, shots, config.EXT_SEED + seed_offset + 2)
    raw = stage3.estimate_from_counts(counts, n, threshold)
    matrices = stage3.confusion_matrices(counts_zeros, counts_ones, n)
    quasi = stage3.mitigate(counts, matrices)
    mitigated = stage3.quasi_at_least(quasi, n, threshold)
    mitigated_se = stage3.mitigated_standard_error(
        counts, counts_zeros, counts_ones, n, threshold, config.MITIGATION_BOOTSTRAP, config.EXT_SEED + seed_offset + 3
    )
    return {
        "exact": exact,
        "shots": int(shots),
        "physical_qubits": qubits,
        "calibration_on_same_qubits": physical_qubits(zeros) == qubits and physical_qubits(ones) == qubits,
        "depth_transpiled": int(device.depth()),
        "two_qubit_gates_transpiled": hardware._count(device, [two_qubit_gate(backend)]),
        "unmitigated": {
            "estimate": raw["estimate"], "error": raw["estimate"] - exact, "standard_error": raw["standard_error"],
            "marginals": raw["marginals"],
        },
        "mitigated": {
            "estimate": mitigated, "error": mitigated - exact, "standard_error": mitigated_se,
            "marginals": (quasi @ stage3.all_bitstrings(n)).tolist(),
        },
        "calibration_p1_given_0": matrices[:, 1, 0].tolist(),
        "calibration_p0_given_1": matrices[:, 0, 1].tolist(),
        "counts": counts,
    }


# --------------------------------------------------------------------------- stage 2
def ising_from_file(stage2_file: dict) -> qaoa.Hamiltonian:
    """The cost Hamiltonian exactly as stage 2 wrote it (fields, couplings, scale)."""
    ising = stage2_file["qubo"]["ising"]
    h = np.array(ising["h"], dtype=float)
    J = np.array(ising["J"], dtype=float)
    scale = float(max(np.abs(h).max(), np.abs(J).max()))
    return qaoa.Hamiltonian(h / scale, J / scale, scale, float(ising["constant"]), None, None)


def trained_circuit(ham: qaoa.Hamiltonian, gamma, beta) -> QuantumCircuit:
    params = np.array(list(gamma) + list(beta), dtype=float)
    return qaoa.bind(qaoa.qaoa_circuit(ham, len(gamma)), params)


def sample_summary(counts: dict, optimum_key: str, k: int) -> dict:
    """How often the samples return the optimum, and a selection of exactly k customers."""
    shots = sum(counts.values())
    right_size = sum(c for key, c in counts.items() if key.count("1") == k)
    best = counts.get(optimum_key, 0)
    p_optimal = best / shots
    conditional = best / right_size if right_size else 0.0
    return {
        "shots": int(shots),
        "p_optimal": p_optimal,
        "p_optimal_standard_error": float(np.sqrt(p_optimal * (1.0 - p_optimal) / shots)),
        "optimum_count": int(best),
        "right_size_fraction": right_size / shots,
        "p_optimal_given_right_size": conditional,
        "p_optimal_given_right_size_standard_error":
            float(np.sqrt(conditional * (1.0 - conditional) / right_size)) if right_size else 0.0,
        "distinct_bitstrings": len(counts),
    }


def stage2_on(backend, ham: qaoa.Hamiltonian, runs: dict, optimum_key: str, k: int, seed_offset: int) -> list:
    """`runs` maps depth -> {"gamma": [...], "beta": [...]} (the parameters stage 2 trained)."""
    sim = simulator(backend)
    out = []
    for index, depth in enumerate(config.IBM_QAOA_LAYERS):
        circuit = trained_circuit(ham, runs[depth]["gamma"], runs[depth]["beta"])
        device, stats = place_best(circuit, backend)
        counts = run_counts(sim, device, config.SHOTS_IBM_QAOA, config.EXT_SEED + seed_offset + index)
        out.append({"layers": int(depth), "p_optimal_ideal": runs[depth]["p_optimal_ideal"], **stats,
                    **sample_summary(counts, optimum_key, k)})
    return out


def stage2_stated_noise_model(ham: qaoa.Hamiltonian, runs: dict, optimum_key: str, k: int, seed_offset: int) -> list:
    """The same circuits on the stated noise model of the main pipeline (config.NOISE, heavy-hex
    device model), with the shot count used for the snapshots."""
    out = []
    for index, depth in enumerate(config.IBM_QAOA_LAYERS):
        circuit = trained_circuit(ham, runs[depth]["gamma"], runs[depth]["beta"])
        device, stats = hardware.transpile_best(circuit)
        counts = hardware.run_counts(
            hardware.noisy_simulator(device), device, config.SHOTS_IBM_QAOA, seed=config.EXT_SEED + seed_offset + index
        )
        out.append({
            "layers": int(depth), "p_optimal_ideal": runs[depth]["p_optimal_ideal"],
            "physical_qubits_used": stats["physical_qubits_used"], "depth_transpiled": stats["depth_transpiled"],
            "two_qubit_gate": hardware.TWO_QUBIT_GATE, "two_qubit_gates_transpiled": stats["two_qubit_gates_transpiled"],
            "swap_count": stats["swap_count"], **sample_summary(counts, optimum_key, k),
        })
    return out


def run(p, threshold: int, stage2_file: dict, qaoa_runs: dict) -> dict:
    """`qaoa_runs`: depth -> trained parameters and the ideal P(optimum), from results/stage2_qaoa_runs.json."""
    ham = ising_from_file(stage2_file)
    optimum_key = stage2_file["brute_force"]["bitstring_qiskit"]
    k = int(stage2_file["constraints"]["n_select"])
    selections = comb(ham.n, k)
    if stage2_file["brute_force"]["feasible_selections"] != selections:
        raise RuntimeError("some selections of the right size are infeasible; the right-size reference needs them all feasible")
    out = {
        "source": "snapshot backends of qiskit-ibm-runtime, simulated with AerSimulator.from_backend (density matrix)",
        "qiskit_ibm_runtime": metadata.version("qiskit-ibm-runtime"),
        "uniform_reference": {
            "p_optimal_random_bitstring": 1.0 / 2 ** ham.n,
            "p_optimal_random_valid_selection": 1.0 / selections,
            "right_size_fraction_random_bitstring": selections / 2 ** ham.n,
            "valid_selections": int(selections),
        },
        "stated_noise_model": {
            "noise": dict(config.NOISE),
            "stage2": stage2_stated_noise_model(ham, qaoa_runs, optimum_key, k, seed_offset=900),
        },
        "backends": [],
    }
    for index, name in enumerate(config.IBM_SNAPSHOTS):
        backend = snapshot(name)
        out["backends"].append({
            **describe(backend),
            "snapshot": name,
            "stage3": stage3_on(backend, p, threshold, seed_offset=100 * index),
            "stage2": stage2_on(backend, ham, qaoa_runs, optimum_key, k, seed_offset=100 * index + 10),
        })
    return out
