"""Amplitude estimation of the stage 3 probability (extension, not the submitted number).

Why
---
The stage 3 circuit is sampled directly: N shots give an error that falls like 1 / sqrt(N). The
same quantity can be read with an error that falls close to 1 / N in the number of circuit
applications, which is the textbook quadratic speed-up of amplitude estimation. This module
builds that circuit for the six selected customers and measures the scaling.

Circuit
-------
A      the stage 3 state preparation, RY(theta_i) on customer qubit i, so that the probability of
       a "good" outcome (at least `threshold` ones) is a = sin^2(phi)
S_chi  flips the sign of every good basis state (a diagonal gate)
S_0    flips the sign of |00...0>
Q      = A S_0 A^-1 S_chi, the Grover operator (up to a global sign)

After k applications of Q the probability of a good outcome is sin^2((2k + 1) phi). The test
suite checks this identity on the state vector for every power used.

Estimator
---------
Maximum-likelihood amplitude estimation (Suzuki et al., Quantum Information Processing 19, 75,
2020): circuits Q^k A |0> are sampled for k = 0, 1, 2, 4, ..., and phi is the maximiser of the
joint binomial likelihood. No phase estimation register and no controlled Q are needed.

Cost
----
Every circuit application is counted in units of A: a shot of Q^k A uses 2k + 1 of them (k in Q,
k in its inverse part, one for the initial preparation). Direct sampling with the same number
of applications is the comparison, measured the same way and also given by its formula. This
count is a query count. It leaves out the two reflections inside Q, which is where almost all
of the physical gates are, so it says nothing about run time on a device.

Noise
-----
The whole study is repeated on the stated device noise model (config.NOISE), with the same
schedules and the same estimator, so that the statement about noise is a measured error and
not an extrapolation.

S_chi is written from the truth table of "at least threshold ones", which is fine for six
qubits. The scalable form is the adder and comparator of `stage3.counter_circuit`, whose flag
qubit marks the same states with a number of gates polynomial in the number of customers.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

from . import config, hardware, stage3
from .qubo import all_bitstrings


def good_states(n: int, threshold: int) -> np.ndarray:
    """Boolean mask over the 2^n basis states (index bit i = qubit i): at least `threshold` ones."""
    return all_bitstrings(n).sum(axis=1) >= threshold


def grover_powers(schedule: int) -> tuple:
    """Schedule j uses the powers 0, 1, 2, 4, ..., 2^(j-1) (j + 1 circuits)."""
    return (0,) + tuple(2 ** k for k in range(schedule))


def state_preparation(p) -> QuantumCircuit:
    return stage3.campaign_circuit(p, measure=False)


def grover_operator(p, threshold: int) -> QuantumCircuit:
    """Q = A S_0 A^-1 S_chi on the customer qubits."""
    n = len(p)
    prep = state_preparation(p)
    mark_good = np.where(good_states(n, threshold), -1.0, 1.0)
    mark_zero = np.ones(2 ** n)
    mark_zero[0] = -1.0
    qc = QuantumCircuit(n)
    qc.append(DiagonalGate(mark_good.tolist()), range(n))
    qc.compose(prep.inverse(), inplace=True)
    qc.append(DiagonalGate(mark_zero.tolist()), range(n))
    qc.compose(prep, inplace=True)
    return qc


def amplified_circuit(p, threshold: int, power: int, measure: bool = True) -> QuantumCircuit:
    """Q^power A |0>, measured on every customer qubit."""
    qc = state_preparation(p)
    q = grover_operator(p, threshold)
    for _ in range(power):
        qc.compose(q, inplace=True)
    if measure:
        qc.measure_all()
    return qc


def good_probability_statevector(p, threshold: int, power: int) -> float:
    probabilities = Statevector(amplified_circuit(p, threshold, power, measure=False)).probabilities()
    return float(probabilities[good_states(len(p), threshold)].sum())


def good_probability_theory(amplitude: float, power: int) -> float:
    """sin^2((2k + 1) phi) with a = sin^2(phi)."""
    return float(np.sin((2 * power + 1) * np.arcsin(np.sqrt(amplitude))) ** 2)


def applications(powers, shots_per_power: int) -> int:
    """Number of applications of A (or its inverse) used by one estimate."""
    return int(sum(shots_per_power * (2 * k + 1) for k in powers))


# --------------------------------------------------------------------------- estimator
def log_likelihood(phi: np.ndarray, powers, hits, shots_per_power: int) -> np.ndarray:
    phi = np.atleast_1d(phi)
    total = np.zeros_like(phi, dtype=float)
    for k, h in zip(powers, hits):
        angle = (2 * k + 1) * phi
        total += h * np.log(np.clip(np.sin(angle) ** 2, 1e-300, None))
        total += (shots_per_power - h) * np.log(np.clip(np.cos(angle) ** 2, 1e-300, None))
    return total


def maximum_likelihood(powers, hits, shots_per_power: int, grid: int = 20001) -> float:
    """Amplitude a = sin^2(phi) at the maximum of the likelihood. The likelihood oscillates with a
    period set by the largest power, so the search is a fine grid over [0, pi/2] followed by a
    golden-section refinement inside the best cell. The grid is much finer than that period."""
    phis = np.linspace(0.0, np.pi / 2.0, grid)
    values = log_likelihood(phis, powers, hits, shots_per_power)
    best = int(np.argmax(values))
    lo = phis[max(best - 1, 0)]
    hi = phis[min(best + 1, grid - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a = hi - ratio * (hi - lo)
        b = lo + ratio * (hi - lo)
        if log_likelihood(a, powers, hits, shots_per_power)[0] > log_likelihood(b, powers, hits, shots_per_power)[0]:
            hi = b
        else:
            lo = a
    return float(np.sin((lo + hi) / 2.0) ** 2)


# --------------------------------------------------------------------------- sampling
def good_flags(memory, threshold: int) -> np.ndarray:
    """One 0/1 flag per shot (1 = at least `threshold` customer qubits read 1), in shot order."""
    return (np.array([key.count("1") for key in memory]) >= threshold).astype(np.int64)


def sample_good(simulator, circuit: QuantumCircuit, threshold: int, shots: int, seed: int) -> np.ndarray:
    memory = simulator.run(circuit, shots=shots, seed_simulator=seed, memory=True).result().get_memory()
    return good_flags(memory, threshold)


def direct_estimates(simulator, circuit: QuantumCircuit, n: int, threshold: int, shots: int, repeats: int, seed: int) -> np.ndarray:
    """`repeats` independent direct-sampling estimates of P(good), each from `shots` shots of the
    stage 3 circuit. Repetition r is its own simulator run with seed `seed + r`."""
    estimates = np.empty(repeats)
    for r in range(repeats):
        counts = simulator.run(circuit, shots=shots, seed_simulator=seed + r).result().get_counts()
        estimates[r] = stage3.estimate_from_counts(counts, n, threshold)["estimate"]
    return estimates


def rmse(estimates, exact: float) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimates) - exact) ** 2)))


def run(p, threshold: int) -> dict:
    p = np.asarray(p, dtype=float)
    n = len(p)
    exact = stage3.exact_at_least(p, threshold)
    repeats, shots = config.AE_REPEATS, config.AE_SHOTS_PER_POWER
    all_powers = grover_powers(max(config.AE_SCHEDULES))
    ideal = hardware.ideal_simulator()

    # each power is sampled once with repeats x shots shots, then cut into `repeats` independent
    # blocks; once on the ideal simulator and once on the stated noise model
    hits, noisy_hits, checks, noise_checks = {}, {}, [], []
    for index, power in enumerate(all_powers):
        device = hardware.to_device(amplified_circuit(p, threshold, power), config.TRANSPILE_SEEDS[0])
        flags = sample_good(ideal, device, threshold, repeats * shots, config.EXT_SEED + index)
        hits[power] = flags.reshape(repeats, shots).sum(axis=1)
        noisy_flags = sample_good(hardware.noisy_simulator(device), device, threshold, repeats * shots,
                                  config.EXT_SEED + 50 + index)
        noisy_hits[power] = noisy_flags.reshape(repeats, shots).sum(axis=1)
        theory = good_probability_theory(exact, power)
        state = good_probability_statevector(p, threshold, power)
        facts = {
            "power": int(power),
            "depth_transpiled": int(device.depth()),
            "two_qubit_gates_transpiled": int(hardware._count(device, [hardware.TWO_QUBIT_GATE])),
            "physical_qubits_used": int(hardware.active_qubits(device)),
        }
        checks.append({
            **facts,
            "good_probability_theory": theory,
            "good_probability_statevector": state,
            "abs_difference": abs(state - theory),
            "good_probability_sampled": float(flags.mean()),
            "shots": int(repeats * shots),
            "seed": config.EXT_SEED + index,
        })
        noise_checks.append({
            **facts,
            "good_probability_ideal": theory,
            "good_probability_noisy": float(noisy_flags.mean()),
            "shift": float(noisy_flags.mean() - theory),
            "shots": int(repeats * shots),
            "seed": config.EXT_SEED + 50 + index,
        })

    direct_circuit = hardware.to_device(amplified_circuit(p, threshold, 0), config.TRANSPILE_SEEDS[0])
    noisy = hardware.noisy_simulator(direct_circuit)
    schedules = []
    for position, schedule in enumerate(config.AE_SCHEDULES):
        powers = grover_powers(schedule)
        cost = applications(powers, shots)
        estimates = np.array([maximum_likelihood(powers, [hits[k][r] for k in powers], shots) for r in range(repeats)])
        noisy_estimates = np.array([
            maximum_likelihood(powers, [noisy_hits[k][r] for k in powers], shots) for r in range(repeats)
        ])
        direct = direct_estimates(ideal, direct_circuit, n, threshold, cost, repeats, config.EXT_SEED + 1000 * (position + 1))
        noisy_direct = direct_estimates(noisy, direct_circuit, n, threshold, cost, repeats, config.EXT_SEED + 1000 * (position + 11))
        error = rmse(estimates, exact)
        direct_formula = float(np.sqrt(exact * (1.0 - exact) / cost))
        schedules.append({
            "schedule": int(schedule),
            "grover_powers": [int(k) for k in powers],
            "largest_power": int(max(powers)),
            "applications_of_A": cost,
            "shots_per_power": int(shots),
            "repeats": int(repeats),
            "mean_estimate": float(estimates.mean()),
            "bias": float(estimates.mean() - exact),
            "rmse": error,
            "direct_sampling_rmse_measured": rmse(direct, exact),
            "direct_sampling_rmse_formula": direct_formula,
            "error_ratio_direct_over_amplitude_estimation": rmse(direct, exact) / error,
            "shots_direct_sampling_needs_for_this_rmse": float(exact * (1.0 - exact) / error ** 2),
            "noisy_rmse": rmse(noisy_estimates, exact),
            "noisy_bias": float(noisy_estimates.mean() - exact),
            "noisy_direct_sampling_rmse_measured": rmse(noisy_direct, exact),
            "noisy_direct_sampling_bias": float(noisy_direct.mean() - exact),
        })

    log_cost = np.log([s["applications_of_A"] for s in schedules])
    slope = float(np.polyfit(log_cost, np.log([s["rmse"] for s in schedules]), 1)[0])
    direct_slope = float(np.polyfit(log_cost, np.log([s["direct_sampling_rmse_measured"] for s in schedules]), 1)[0])
    return {
        "exact": exact,
        "threshold": int(threshold),
        "qubits": int(n),
        "method": "maximum-likelihood amplitude estimation, Grover powers 0, 1, 2, 4, ... (Suzuki et al. 2020)",
        "cost_unit": "applications of the state preparation A; a shot of Q^k A uses 2k + 1 (the reflections are not counted)",
        "identity_check": checks,
        "identity_max_abs_difference": float(max(c["abs_difference"] for c in checks)),
        "schedules": schedules,
        "fitted_slope_log_rmse_vs_log_cost": slope,
        "fitted_slope_direct_sampling_measured": direct_slope,
        "direct_sampling_slope_theory": -0.5,
        "slope_fit_points": len(schedules),
        "best_schedule": schedules[-1],
        "noise_model": dict(config.NOISE),
        "uniform_bits_good_probability": float(good_states(n, threshold).mean()),
        "uniform_bits_good_probability_after_readout": stage3.exact_at_least(stage3.readout_shifted([0.5] * n), threshold),
        "noisy": noise_checks,
    }
