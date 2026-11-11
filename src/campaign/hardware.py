"""Device model used for the noisy runs: heavy-hex connectivity, IBM Eagle basis gates, and an
explicit noise model whose numbers are stated in config.NOISE.

Nothing here talks to real hardware. The noise model is an assumption in the range of public
Eagle calibrations (depolarising gate errors plus asymmetric readout errors). It leaves out
T1/T2 decay during idle time, crosstalk and drift, which real devices also have.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from . import config

ONE_QUBIT_GATES = ("sx", "x", "rz")
TWO_QUBIT_GATE = "ecr"


def coupling_map() -> CouplingMap:
    return CouplingMap.from_heavy_hex(config.HEAVY_HEX_DISTANCE)


def noise_model() -> NoiseModel:
    model = NoiseModel(basis_gates=config.BASIS_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(config.NOISE["one_qubit_depolarizing"], 1), ["sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(config.NOISE["two_qubit_depolarizing"], 2), ["ecr"])
    model.add_all_qubit_readout_error(ReadoutError(readout_matrix()))
    return model


def readout_matrix() -> np.ndarray:
    """Row = prepared state, column = reported state (Aer's convention)."""
    p01 = config.NOISE["readout_p1_given_0"]
    p10 = config.NOISE["readout_p0_given_1"]
    return np.array([[1.0 - p01, p01], [p10, 1.0 - p10]])


def ideal_simulator() -> AerSimulator:
    return AerSimulator(method="statevector", seed_simulator=config.SIM_SEED)


def noisy_simulator(circuit: QuantumCircuit = None) -> AerSimulator:
    """Simulator with the noise model. The method is chosen by the number of physical qubits the
    circuit touches (Aer drops idle qubits): an exact density matrix up to DENSITY_MATRIX_MAX_QUBITS,
    state-vector trajectories (one noisy run per shot) above that, where a density matrix would
    need 4^n numbers."""
    width = config.DENSITY_MATRIX_MAX_QUBITS if circuit is None else active_qubits(circuit)
    method = "density_matrix" if width <= config.DENSITY_MATRIX_MAX_QUBITS else "statevector"
    return AerSimulator(
        method=method, noise_model=noise_model(), basis_gates=config.BASIS_GATES, seed_simulator=config.SIM_SEED
    )


def run_noisy(circuit: QuantumCircuit, shots: int, seed: int = None) -> dict:
    """Counts of a device-basis circuit under the noise model."""
    return run_counts(noisy_simulator(circuit), circuit, shots, seed)


def run_counts(simulator: AerSimulator, circuit: QuantumCircuit, shots: int, seed: int = None) -> dict:
    """Run one circuit and return its counts with keys sorted (deterministic output order)."""
    seed = config.SIM_SEED if seed is None else seed
    counts = simulator.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()
    return {key: int(counts[key]) for key in sorted(counts)}


# --------------------------------------------------------------------------- transpilation
def _count(circuit: QuantumCircuit, names) -> int:
    ops = circuit.count_ops()
    return int(sum(ops.get(name, 0) for name in names))


def active_qubits(circuit: QuantumCircuit) -> int:
    """Physical qubits touched by at least one gate or measurement."""
    used = set()
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        used.update(circuit.find_bit(q).index for q in instruction.qubits)
    return len(used)


def to_device(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    return transpile(
        circuit, coupling_map=coupling_map(), basis_gates=config.BASIS_GATES,
        optimization_level=3, seed_transpiler=seed,
    )


def routing_swaps(circuit: QuantumCircuit, seed: int) -> int:
    """SWAPs inserted by the router, seen by keeping `swap` as a basis gate at the same seed.

    In the final circuit every SWAP costs up to three two-qubit gates, some of which the
    optimiser merges with neighbouring gates, so this count and the ECR count are reported
    side by side rather than derived from one another.
    """
    routed = transpile(
        circuit, coupling_map=coupling_map(), basis_gates=config.BASIS_GATES + ["swap"],
        optimization_level=3, seed_transpiler=seed,
    )
    return _count(routed, ["swap"])


def transpile_best(circuit: QuantumCircuit, seeds=config.TRANSPILE_SEEDS):
    """Transpile once per seed. Keep the circuit that touches the fewest physical qubits (routing
    through a spare qubit exposes it to noise too), then the fewest two-qubit gates, then the
    lowest depth, then the lowest seed. Returns (circuit, stats)."""
    logical = circuit
    tried = []
    for seed in seeds:
        candidate = to_device(circuit, seed)
        tried.append((active_qubits(candidate), _count(candidate, [TWO_QUBIT_GATE]), candidate.depth(), seed, candidate))
    _, two_qubit, depth, seed, best = min(tried, key=lambda t: t[:4])
    logical_ops = {name: int(count) for name, count in sorted(logical.count_ops().items())}
    logical_two_qubit = int(sum(1 for inst in logical.data if len(inst.qubits) == 2))
    logical_blocks = int(sum(1 for inst in logical.data if len(inst.qubits) > 2 and inst.operation.name != "barrier"))
    logical_one_qubit = int(sum(
        1 for inst in logical.data if len(inst.qubits) == 1 and inst.operation.name not in ("measure", "barrier")
    ))
    stats = {
        "logical_qubits": int(circuit.num_qubits),
        "physical_qubits_device": int(coupling_map().size()),
        "physical_qubits_used": active_qubits(best),
        "depth_original": int(logical.depth()),
        "depth_transpiled": int(depth),
        "one_qubit_gates_original": logical_one_qubit,
        "two_qubit_gates_original": logical_two_qubit,
        "multi_qubit_blocks_original": logical_blocks,
        "one_qubit_gates_transpiled": _count(best, ONE_QUBIT_GATES),
        "two_qubit_gates_transpiled": int(two_qubit),
        "swap_count": routing_swaps(circuit, seed),
        "original_gate_counts": logical_ops,
        "transpiled_gate_counts": {name: int(count) for name, count in sorted(best.count_ops().items())},
        "basis_gates": list(config.BASIS_GATES),
        "coupling_map": f"heavy-hex, distance {config.HEAVY_HEX_DISTANCE}",
        "optimization_level": 3,
        "transpiler_seed": int(seed),
        "transpiler_seeds_tried": [int(s) for s in seeds],
        "two_qubit_gates_by_seed": {str(t[3]): int(t[1]) for t in tried},
        "physical_qubits_by_seed": {str(t[3]): int(t[0]) for t in tried},
    }
    return best, stats
