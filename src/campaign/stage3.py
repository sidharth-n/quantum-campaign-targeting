"""Stage 3: probability that at least `threshold` of the selected customers subscribe.

Circuit
-------
One qubit per selected customer. Qubit i belongs to the i-th selected customer in stage 1 rank
order (highest probability first) and is prepared with RY(theta_i), theta_i = 2 asin(sqrt(p_i)),
so that P(qubit i reads 1) = sin^2(theta_i / 2) = p_i. There are no entangling gates because the
customers respond independently. A shot is a success when at least `threshold` qubits read 1.

Bit order
---------
Qiskit writes a measured bitstring with qubit 0 as the rightmost character. `decode` reverses
the key, so element i of the decoded array is the outcome of qubit i, that is of customer i.
The success test only needs the number of ones, but the per-customer marginals are also
reported: each must reproduce its own customer's probability, which only happens when the
decoding order is right.

Estimates, all compared with the exact Poisson-binomial value
------------------------------------------------------------
  statevector        the ideal output distribution of the circuit (no sampling error)
  finite shots       noiseless Aer samples; the 2^20-shot run is the submitted quantum estimate
  noisy              the same circuit transpiled to the device model with gate and readout noise
  mitigated          the noisy counts corrected with a tensored readout calibration
  counter circuit    an adder and a comparator compute "at least threshold" coherently and write
                     it on one flag qubit, so P(flag = 1) is the answer (the form of the circuit
                     that amplitude estimation would start from)
  Monte Carlo        classical Bernoulli draws, the benchmark the organisers ask for
"""
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator, WeightedAdder
from qiskit.quantum_info import Statevector

from . import config, hardware
from .qubo import all_bitstrings, key_to_bits


# --------------------------------------------------------------------------- exact references
def poisson_binomial_pmf(p) -> np.ndarray:
    """P(exactly j successes), j = 0..n, by dynamic programming over the customers."""
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.append(pmf, 0.0) * (1.0 - pi) + np.append(0.0, pmf) * pi
    return pmf


def exact_at_least(p, threshold: int) -> float:
    return float(poisson_binomial_pmf(p)[threshold:].sum())


def enumerate_at_least(p, threshold: int) -> float:
    """Independent check: sum the probability of every one of the 2^n outcomes."""
    p = np.asarray(p, dtype=float)
    X = all_bitstrings(len(p))
    weights = np.prod(np.where(X == 1, p, 1.0 - p), axis=1)
    return float(weights[X.sum(axis=1) >= threshold].sum())


def sensitivity(p, threshold: int) -> np.ndarray:
    """dP(at least threshold)/dp_i = P(exactly threshold - 1 successes among the other customers)."""
    p = np.asarray(p, dtype=float)
    return np.array([poisson_binomial_pmf(np.delete(p, i))[threshold - 1] for i in range(len(p))])


def monte_carlo(p, threshold: int, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.random((trials, len(p))) < np.asarray(p)[None, :]
    estimate = float((draws.sum(axis=1) >= threshold).mean())
    return {"estimate": estimate, "trials": int(trials), "seed": int(seed),
            "standard_error": float(np.sqrt(estimate * (1.0 - estimate) / trials))}


# --------------------------------------------------------------------------- circuits
def angles(p) -> np.ndarray:
    return 2.0 * np.arcsin(np.sqrt(np.asarray(p, dtype=float)))


def campaign_circuit(p, measure: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(len(p))
    for qubit, theta in enumerate(angles(p)):
        qc.ry(float(theta), qubit)
    if measure:
        qc.measure_all()
    return qc


def counter_circuit(p, threshold: int, measure: bool = True) -> QuantumCircuit:
    """Customers -> adder (number of ones, in binary) -> comparator (sum >= threshold) -> flag qubit."""
    n = len(p)
    adder = WeightedAdder(n, weights=[1] * n)
    n_sum = adder.num_sum_qubits
    comparator = IntegerComparator(n_sum, value=threshold, geq=True)
    customers = QuantumRegister(n, "customer")
    total = QuantumRegister(n_sum, "sum")
    flag = QuantumRegister(1, "flag")
    n_work = max(adder.num_qubits - n - n_sum, comparator.num_qubits - n_sum - 1)
    work = QuantumRegister(n_work, "work")
    qc = QuantumCircuit(customers, total, flag, work)
    for qubit, theta in enumerate(angles(p)):
        qc.ry(float(theta), customers[qubit])
    adder_work = adder.num_qubits - n - n_sum
    qc.append(adder.to_gate(), [*customers, *total, *work[:adder_work]])
    comparator_work = comparator.num_qubits - n_sum - 1
    qc.append(comparator.to_gate(), [*total, flag[0], *work[:comparator_work]])
    if measure:
        result = ClassicalRegister(1, "at_least")
        qc.add_register(result)
        qc.measure(flag[0], result[0])
    return qc


# --------------------------------------------------------------------------- decoding
def decode(counts: dict, n: int):
    """Counts -> (outcomes, shots) where outcomes[r, i] is the value read on qubit i."""
    keys = list(counts)
    outcomes = np.array([key_to_bits(key)[:n] for key in keys], dtype=np.int8).reshape(len(keys), n)
    shots = np.array([counts[key] for key in keys], dtype=float)
    return outcomes, shots


def estimate_from_counts(counts: dict, n: int, threshold: int) -> dict:
    outcomes, shots = decode(counts, n)
    total = shots.sum()
    estimate = float(shots[outcomes.sum(axis=1) >= threshold].sum() / total)
    return {
        "estimate": estimate,
        "shots": int(total),
        "standard_error": float(np.sqrt(estimate * (1.0 - estimate) / total)),
        "marginals": (shots @ outcomes / total).tolist(),      # P(qubit i reads 1), i = 0..n-1
    }


def statevector_at_least(p, threshold: int) -> float:
    probabilities = Statevector(campaign_circuit(p, measure=False)).probabilities()   # index bit i = qubit i
    ones = all_bitstrings(len(p)).sum(axis=1)
    return float(probabilities[ones >= threshold].sum())


# --------------------------------------------------------------------------- readout mitigation
def calibration_circuits(n: int):
    """Two circuits (all zeros, all ones) are enough for a tensored, per-qubit calibration."""
    zeros = QuantumCircuit(n)
    zeros.measure_all()
    ones = QuantumCircuit(n)
    ones.x(range(n))
    ones.measure_all()
    return zeros, ones


def confusion_matrices(counts_zeros: dict, counts_ones: dict, n: int) -> np.ndarray:
    """Per-qubit matrices M with M[read, prepared] estimated from the two calibration runs."""
    out0, shots0 = decode(counts_zeros, n)
    out1, shots1 = decode(counts_ones, n)
    p1_given_0 = shots0 @ out0 / shots0.sum()
    p1_given_1 = shots1 @ out1 / shots1.sum()
    return np.array([[[1.0 - a, 1.0 - b], [a, b]] for a, b in zip(p1_given_0, p1_given_1)])


def mitigate(counts: dict, matrices: np.ndarray) -> np.ndarray:
    """Apply the inverse of the tensored confusion matrix. Returns quasi-probabilities over the
    2^n outcomes (index bit i = qubit i); entries can be slightly negative."""
    n = len(matrices)
    outcomes, shots = decode(counts, n)
    observed = np.zeros(2 ** n)
    np.add.at(observed, outcomes.astype(int) @ (1 << np.arange(n)), shots / shots.sum())
    tensor = observed.reshape((2,) * n)                    # axis 0 = most significant bit = qubit n-1
    for qubit in range(n):
        axis = n - 1 - qubit
        inverse = np.linalg.inv(matrices[qubit])
        tensor = np.moveaxis(np.tensordot(inverse, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(2 ** n)


def quasi_at_least(quasi: np.ndarray, n: int, threshold: int) -> float:
    return float(quasi[all_bitstrings(n).sum(axis=1) >= threshold].sum())


def mitigated_standard_error(counts: dict, counts_zeros: dict, counts_ones: dict, n: int, threshold: int,
                             rounds: int, seed: int) -> float:
    """Standard error of the mitigated estimate by parametric bootstrap: the measured counts are
    redrawn from a multinomial and both calibration runs from per-qubit binomials, and the
    mitigation is repeated. It carries the sampling error of the run and of the calibration."""
    rng = np.random.default_rng(seed)
    keys = [format(b, f"0{n}b") for b in range(2 ** n)]
    observed = np.array([counts.get(key, 0) for key in keys], dtype=float)
    shots = int(observed.sum())
    out0, shots0 = decode(counts_zeros, n)
    out1, shots1 = decode(counts_ones, n)
    n0, n1 = int(shots0.sum()), int(shots1.sum())
    p1_given_0 = shots0 @ out0 / n0
    p1_given_1 = shots1 @ out1 / n1
    estimates = []
    for _ in range(rounds):
        redrawn = rng.multinomial(shots, observed / shots)
        a = rng.binomial(n0, p1_given_0) / n0
        b = rng.binomial(n1, p1_given_1) / n1
        matrices = np.array([[[1.0 - x, 1.0 - y], [x, y]] for x, y in zip(a, b)])
        quasi = mitigate({key: int(c) for key, c in zip(keys, redrawn) if c}, matrices)
        estimates.append(quasi_at_least(quasi, n, threshold))
    return float(np.std(estimates, ddof=1))


def readout_shifted(p) -> np.ndarray:
    """P(qubit reads 1) after the readout error of the noise model: p (1 - e10) + (1 - p) e01."""
    p = np.asarray(p, dtype=float)
    e01 = config.NOISE["readout_p1_given_0"]
    e10 = config.NOISE["readout_p0_given_1"]
    return p * (1.0 - e10) + (1.0 - p) * e01


# --------------------------------------------------------------------------- stage 3 run
def run(ids, p, threshold: int) -> dict:
    p = np.asarray(p, dtype=float)
    n = len(p)
    exact = exact_at_least(p, threshold)
    pmf = poisson_binomial_pmf(p)
    ideal = hardware.ideal_simulator()

    def with_error(entry: dict) -> dict:
        entry["error"] = entry["estimate"] - exact
        entry["abs_error"] = abs(entry["error"])
        return entry

    circuit = campaign_circuit(p)
    statevector = with_error({"estimate": statevector_at_least(p, threshold)})

    # noiseless finite-shot runs, one independent seed per shot count
    sweep = []
    for index, shots in enumerate(config.SHOT_SWEEP):
        counts = hardware.run_counts(ideal, circuit, shots, seed=config.SIM_SEED + index)
        entry = with_error(estimate_from_counts(counts, n, threshold))
        entry["seed"] = config.SIM_SEED + index
        sweep.append(entry)
    counts_ideal = hardware.run_counts(ideal, circuit, config.SHOTS_STAGE3)
    ideal_shots = with_error(estimate_from_counts(counts_ideal, n, threshold))
    counts_headline = hardware.run_counts(ideal, circuit, config.SHOTS_STAGE3_HEADLINE)
    headline = with_error(estimate_from_counts(counts_headline, n, threshold))

    # device model: transpile, run with noise, then readout mitigation
    device_circuit, transpile_stats = hardware.transpile_best(circuit)
    counts_noisy = hardware.run_noisy(device_circuit, config.SHOTS_STAGE3)
    noisy_shots = with_error(estimate_from_counts(counts_noisy, n, threshold))
    # what the readout error alone predicts (gate errors on six single-qubit rotations are ~1e-3)
    noisy_shots["expected_from_readout_model"] = exact_at_least(readout_shifted(p), threshold)
    noisy_shots["expected_bias"] = noisy_shots["expected_from_readout_model"] - exact
    noisy_shots["expected_marginals"] = readout_shifted(p).tolist()
    zeros, ones = (hardware.to_device(c, transpile_stats["transpiler_seed"]) for c in calibration_circuits(n))
    counts_zeros = hardware.run_noisy(zeros, config.SHOTS_STAGE3, seed=config.SIM_SEED + 101)
    counts_ones = hardware.run_noisy(ones, config.SHOTS_STAGE3, seed=config.SIM_SEED + 102)
    matrices = confusion_matrices(counts_zeros, counts_ones, n)
    quasi = mitigate(counts_noisy, matrices)
    mitigated = with_error({
        "estimate": quasi_at_least(quasi, n, threshold),
        "shots": config.SHOTS_STAGE3,
        "standard_error": mitigated_standard_error(
            counts_noisy, counts_zeros, counts_ones, n, threshold, config.MITIGATION_BOOTSTRAP, config.MC_SEED + 1
        ),
        "standard_error_method": f"parametric bootstrap of the run and the calibration, {config.MITIGATION_BOOTSTRAP} rounds",
        "calibration_shots_per_circuit": config.SHOTS_STAGE3,
        "calibration_p1_given_0": matrices[:, 1, 0].tolist(),
        "calibration_p0_given_1": matrices[:, 0, 1].tolist(),
        "most_negative_quasi_probability": float(quasi.min()),
        "marginals": (quasi @ all_bitstrings(n)).tolist(),
    })

    # coherent counter: one flag qubit carries the answer
    counter = counter_circuit(p, threshold)
    counter_flat = counter.decompose(reps=6)
    flag_probability = Statevector(counter_circuit(p, threshold, measure=False)).probabilities(
        [counter.qubits.index(counter.qregs[2][0])]
    )[1]
    counter_counts = hardware.run_counts(ideal, hardware.to_device(counter, config.TRANSPILE_SEEDS[0]), config.SHOTS_STAGE3)
    counter_estimate = counter_counts.get("1", 0) / config.SHOTS_STAGE3
    counter_device, counter_stats = hardware.transpile_best(counter)
    counter_noisy = hardware.run_noisy(counter_device, config.SHOTS_COUNTER_NOISY)
    counter_noisy_estimate = counter_noisy.get("1", 0) / config.SHOTS_COUNTER_NOISY
    counter_entry = with_error({
        "estimate": float(counter_estimate),
        "shots": config.SHOTS_STAGE3,
        "standard_error": float(np.sqrt(counter_estimate * (1 - counter_estimate) / config.SHOTS_STAGE3)),
        "counts": counter_counts,
        "statevector_flag_probability": float(flag_probability),
        "statevector_abs_error": float(abs(flag_probability - exact)),
        "qubits": int(counter.num_qubits),
        "registers": {reg.name: reg.size for reg in counter.qregs},
        "depth_elementary_gates": int(counter_flat.depth()),
        "noisy_estimate": float(counter_noisy_estimate),
        "noisy_error": float(counter_noisy_estimate - exact),
        "noisy_shots": config.SHOTS_COUNTER_NOISY,
        "noisy_counts": counter_noisy,
    })

    mc = with_error(monte_carlo(p, threshold, config.MC_TRIALS, config.MC_SEED))

    return {
        "ids": list(ids),
        "probabilities": p.tolist(),
        "threshold": int(threshold),
        "angles": angles(p).tolist(),
        "pmf": pmf.tolist(),
        "exact": exact,
        "exact_enumeration": enumerate_at_least(p, threshold),
        "sensitivity": sensitivity(p, threshold).tolist(),
        "statevector": statevector,
        "ideal_shots": ideal_shots,
        "headline": headline,
        "shot_sweep": sweep,
        "noisy": noisy_shots,
        "mitigated": mitigated,
        "counter": counter_entry,
        "monte_carlo": mc,
        "counts_ideal": counts_ideal,
        "counts_headline": counts_headline,
        "counts_noisy": counts_noisy,
        "counts_calibration_zeros": counts_zeros,
        "counts_calibration_ones": counts_ones,
        "transpile": transpile_stats,
        "counter_transpile": counter_stats,
    }
