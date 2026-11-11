"""Beyond the brief: Stage 3 with correlated customers (one-factor Gaussian copula).

Why
---
Stage 3 prepares one qubit per selected customer with no entangling gates, because the brief
says customers respond independently. In a real campaign they do not: a good or bad month for
rates, wages or markets moves every customer the same way. Banks model this with a one-factor
Gaussian copula (the Vasicek model behind credit portfolio risk). Woerner and Egger (2019) and
Egger et al. (2021) load exactly this model on a quantum computer for credit risk; here it is
applied to the six selected customers.

Model
-----
A market factor Z takes 2^m equally likely values z_k = Phi^-1((k + 1/2) / 2^m), k = 0..2^m-1.
Given Z = z_k the customers respond independently and customer i subscribes with probability

    p_i(z_k) = Phi((c_i + sqrt(rho) z_k) / sqrt(1 - rho))          (higher factor = better market)

The threshold c_i is solved numerically so that the scenario average equals the stage 1
probability exactly: mean_k p_i(z_k) = p_i. Every customer's marginal probability is therefore
unchanged and rho only adds dependence. For rho = 0 every p_i(z_k) = p_i and the model is the
independent Stage 3 model. Exact references:

    P(x)                = mean_k prod_i p_i(z_k)^x_i (1 - p_i(z_k))^(1 - x_i)
    P(exactly j ones)   = mean_k PoissonBinomial(p_1(z_k), .., p_n(z_k))[j]

Circuit
-------
Registers `factor` (m qubits, first) and `customer` (n qubits). A Hadamard on every factor qubit
gives the 2^m equally likely scenarios (factor qubit j = bit j of k). Customer qubit i gets a
uniformly controlled RY (UCRYGate, target first, then the factor qubits as controls with the
first control as the least significant bit) whose angle is 2 asin(sqrt(p_i(z_k))) when the
factor register holds k. Every customer qubit is entangled with the factor register, and that
shared register is the only thing that correlates the customers with each other. Only the
customer qubits are measured, into an n-bit register, so the count
keys decode with stage3.decode exactly like the Stage 3 circuit: bit i = customer i.

Because the scenarios are equally likely, relabelling them (for example reversing the order of
the factor qubits) leaves the customer distribution unchanged. The scenario-to-angle mapping is
therefore checked on the joint distribution of both registers in the tests.

Revenue at risk
---------------
Every selected customer is contacted, so the full contact cost is always paid. The net value of
an outcome x is  V(x) = sum_i deposit_i x_i - sum_i cost_i. Its mean does not depend on rho (the
marginals do not change); its spread and its lower tail do. For a tail level a:

    value at risk       VaR_a = the smallest v with P(V <= v) >= a (the a-quantile of V)
    expected shortfall  ES_a  = [ sum_{v < VaR_a} v P(V = v) + VaR_a (a - P(V < VaR_a)) ] / a

ES_a is the mean of V over its worst a fraction; the second term takes only the part of the
atom at VaR_a that falls inside that fraction.
"""
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import Statevector
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from . import config, hardware, stage3
from .qubo import all_bitstrings

CALIBRATION_TOLERANCE = 1e-12   # |mean_k p_i(z_k) - p_i| allowed after solving for c_i
QUANTILE_TOLERANCE = 1e-12      # cumulative probabilities within this of the tail level count as reaching it
REFERENCES = [
    "S. Woerner and D. J. Egger, Quantum risk analysis, npj Quantum Information 5, 15 (2019)",
    "D. J. Egger, R. Garcia Gutierrez, J. Mestre and S. Woerner, Credit risk analysis using quantum computers, "
    "IEEE Transactions on Computers 70(12), 2136-2145 (2021)",
]


# --------------------------------------------------------------------------- model
def factor_values(m: int) -> np.ndarray:
    """The 2^m equally likely market scenarios z_k = Phi^-1((k + 1/2) / 2^m)."""
    size = 2 ** int(m)
    return ndtri((np.arange(size) + 0.5) / size)


def _check(p, rho: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not ((p > 0.0) & (p < 1.0)).all():
        raise ValueError("probabilities must be a 1-d array with every entry strictly between 0 and 1")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return p


def thresholds(p, rho: float, m: int) -> np.ndarray:
    """c_i such that mean_k Phi((c_i + sqrt(rho) z_k) / sqrt(1 - rho)) = p_i, by bracketed root finding.

    The scenario average is strictly increasing in c_i, so the root is unique. With |z_k| < 4
    (m <= 14) it is below 1e-200 at c_i = -40 and equal to 1 at c_i = +40, so the bracket holds
    the root for any p_i not within 1e-16 of 0 or 1."""
    p = _check(p, rho)
    z = factor_values(m)
    loading, scale = np.sqrt(rho), np.sqrt(1.0 - rho)
    out = np.empty(len(p))
    for i, target in enumerate(p):
        def gap(c, target=target):
            return float(ndtr((c + loading * z) / scale).mean() - target)
        out[i] = brentq(gap, -40.0, 40.0, xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    return out


def conditional_probabilities(p, rho: float, m: int) -> np.ndarray:
    """Shape (2^m, n): entry [k, i] = p_i(z_k), the subscription probability of customer i in scenario k."""
    p = _check(p, rho)
    c = thresholds(p, rho, m)
    q = ndtr((c[None, :] + np.sqrt(rho) * factor_values(m)[:, None]) / np.sqrt(1.0 - rho))
    drift = float(np.abs(q.mean(axis=0) - p).max())
    if drift > CALIBRATION_TOLERANCE:
        raise RuntimeError(f"marginal calibration failed: max |mean_k p_i(z_k) - p_i| = {drift:.3e}")
    return q


# --------------------------------------------------------------------------- exact references
def exact_pmf(p, rho: float, m: int) -> np.ndarray:
    """P(exactly j customers subscribe), j = 0..n: the scenario average of the Poisson-binomial pmf."""
    q = conditional_probabilities(p, rho, m)
    return np.mean([stage3.poisson_binomial_pmf(row) for row in q], axis=0)


def exact_at_least(p, rho: float, m: int, threshold: int) -> float:
    return float(exact_pmf(p, rho, m)[threshold:].sum())


def outcome_distribution(p, rho: float, m: int) -> np.ndarray:
    """Exact probability of each of the 2^n joint outcomes. Index b has bit i = customer i."""
    q = conditional_probabilities(p, rho, m)
    X = all_bitstrings(q.shape[1])
    per_scenario = np.prod(np.where(X[None, :, :] == 1, q[:, None, :], 1.0 - q[:, None, :]), axis=2)
    return per_scenario.mean(axis=0)


def correlation_from_distribution(distribution, n: int) -> np.ndarray:
    """(n, n) Pearson correlation matrix of the customer outcomes under a distribution over 2^n outcomes."""
    distribution = np.asarray(distribution, dtype=float)
    X = all_bitstrings(n).astype(float)
    mean = distribution @ X
    second = X.T @ (distribution[:, None] * X)
    covariance = second - np.outer(mean, mean)
    sd = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sd, sd)


def pairwise_correlation(p, rho: float, m: int) -> np.ndarray:
    return correlation_from_distribution(outcome_distribution(p, rho, m), len(p))


def mean_off_diagonal(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    return float(matrix[~np.eye(n, dtype=bool)].mean())


def count_moments(pmf) -> tuple:
    """(mean, variance) of the number of subscribers."""
    pmf = np.asarray(pmf, dtype=float)
    j = np.arange(len(pmf))
    mean = float(pmf @ j)
    return mean, float(pmf @ j ** 2 - mean ** 2)


# --------------------------------------------------------------------------- circuit
def correlated_circuit(p, rho: float, m: int, measure: bool = True) -> QuantumCircuit:
    q = conditional_probabilities(p, rho, m)
    n = q.shape[1]
    factor = QuantumRegister(m, "factor")
    customer = QuantumRegister(n, "customer")
    qc = QuantumCircuit(factor, customer)
    qc.h(factor)
    for i in range(n):
        scenario_angles = [float(a) for a in stage3.angles(q[:, i])]     # index k = integer held by the factor register
        qc.append(UCRYGate(scenario_angles), [customer[i], *factor])      # target first; factor qubit 0 = bit 0 of k
    if measure:
        outcome = ClassicalRegister(n, "outcome")
        qc.add_register(outcome)
        qc.measure(customer, outcome)                                     # clbit i = customer i
    return qc


def statevector_distribution(p, rho: float, m: int) -> np.ndarray:
    """Marginal distribution of the customer qubits from the ideal state (index bit i = customer i)."""
    circuit = correlated_circuit(p, rho, m, measure=False)
    customers = [circuit.find_bit(qubit).index for qubit in circuit.qregs[1]]
    return Statevector(circuit).probabilities(customers)


def empirical_distribution(counts: dict, n: int) -> np.ndarray:
    """Counts -> relative frequency of each of the 2^n outcomes (index bit i = customer i)."""
    outcomes, shots = stage3.decode(counts, n)
    distribution = np.zeros(2 ** n)
    np.add.at(distribution, outcomes.astype(np.int64) @ (1 << np.arange(n)), shots)
    return distribution / shots.sum()


# --------------------------------------------------------------------------- revenue at risk
def net_values(deposits, costs) -> np.ndarray:
    """Net value of each of the 2^n outcomes: deposits of the subscribers minus every contact cost."""
    deposits = np.asarray(deposits, dtype=float)
    return all_bitstrings(len(deposits)) @ deposits - float(np.sum(costs))


def revenue_stats(distribution, deposits, costs, tail: float) -> dict:
    distribution = np.asarray(distribution, dtype=float)
    n = len(deposits)
    if distribution.shape != (2 ** n,):
        raise ValueError(f"distribution must have 2^{n} entries")
    if (distribution < 0).any() or abs(distribution.sum() - 1.0) > 1e-9:
        raise ValueError("distribution must be non-negative and sum to 1")
    if not 0.0 < tail < 1.0:
        raise ValueError("tail must lie in (0, 1)")
    net = net_values(deposits, costs)
    expected = float(distribution @ net)
    values, position = np.unique(net, return_inverse=True)               # ascending, outcomes with equal value merged
    mass = np.bincount(position, weights=distribution, minlength=len(values))
    cumulative = np.cumsum(mass)
    q = int(np.argmax(cumulative >= tail - QUANTILE_TOLERANCE))           # first value whose CDF reaches the tail level
    var = float(values[q])
    below = float(cumulative[q - 1]) if q > 0 else 0.0                    # P(V < VaR)
    shortfall = (float(mass[:q] @ values[:q]) + var * (tail - below)) / tail
    support = values[mass > 0]
    return {
        "tail": float(tail),
        "expected_net_value": expected,
        "standard_deviation": float(np.sqrt(max(distribution @ (net - expected) ** 2, 0.0))),
        "probability_of_loss": float(distribution[net < 0].sum()),
        "value_at_risk": var,
        "expected_shortfall": float(shortfall),
        "min_net_value": float(support.min()),
        "max_net_value": float(support.max()),
    }


def revenue_stats_from_counts(counts: dict, n: int, deposits, costs, tail: float) -> dict:
    return revenue_stats(empirical_distribution(counts, n), deposits, costs, tail)


# --------------------------------------------------------------------------- extension run
def circuit_facts(circuit: QuantumCircuit, device: QuantumCircuit, seed: int) -> dict:
    """Size of the logical circuit, of its decomposition with all-to-all connectivity and of the
    heavy-hex circuit (two-qubit gates counted as in hardware.transpile_best)."""
    flat = transpile(circuit, basis_gates=config.BASIS_GATES, optimization_level=3, seed_transpiler=seed)
    return {
        "logical_qubits": int(circuit.num_qubits),
        "factor_qubits": int(circuit.qregs[0].size),
        "customer_qubits": int(circuit.qregs[1].size),
        "depth_original": int(circuit.depth()),
        "uniformly_controlled_rotations": int(circuit.count_ops().get("ucry", 0)),
        "depth_all_to_all": int(flat.depth()),
        "two_qubit_gates_all_to_all": hardware._count(flat, [hardware.TWO_QUBIT_GATE]),
        "physical_qubits_used": hardware.active_qubits(device),
        "depth_transpiled": int(device.depth()),
        "one_qubit_gates_transpiled": hardware._count(device, hardware.ONE_QUBIT_GATES),
        "two_qubit_gates_transpiled": hardware._count(device, [hardware.TWO_QUBIT_GATE]),
        "swap_count": hardware.routing_swaps(circuit, seed),
        "transpiled_gate_counts": {name: int(count) for name, count in sorted(device.count_ops().items())},
        "basis_gates": list(config.BASIS_GATES),
        "coupling_map": f"heavy-hex, distance {config.HEAVY_HEX_DISTANCE}",
        "optimization_level": 3,
        "transpiler_seed": int(seed),
    }


def run(ids, p, deposits, costs, threshold: int) -> dict:
    p = np.asarray(p, dtype=float)
    deposits = np.asarray(deposits, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n, m = len(p), config.FACTOR_QUBITS
    ones = all_bitstrings(n).sum(axis=1)
    ideal = hardware.ideal_simulator()
    seed = config.TRANSPILE_SEEDS[0]
    rho_max = max(config.CORRELATION_GRID)
    scenarios, facts = [], None
    for index, rho in enumerate(config.CORRELATION_GRID):
        pmf = exact_pmf(p, rho, m)
        exact = float(pmf[threshold:].sum())
        distribution = outcome_distribution(p, rho, m)
        state = statevector_distribution(p, rho, m)
        statevector_value = float(state[ones >= threshold].sum())
        circuit = correlated_circuit(p, rho, m)
        device = hardware.to_device(circuit, seed)
        counts = hardware.run_counts(ideal, device, config.SHOTS_CORRELATED, seed=config.EXT_SEED + index)
        sampled = stage3.estimate_from_counts(counts, n, threshold)
        marginal_gap = np.abs(np.array(sampled["marginals"]) - p)
        mean_count, variance_count = count_moments(pmf)
        if rho == rho_max:
            facts = circuit_facts(circuit, device, seed)
        scenarios.append({
            "rho": float(rho),
            "thresholds_c": thresholds(p, rho, m).tolist(),
            "conditional_probabilities": conditional_probabilities(p, rho, m).tolist(),   # [scenario k][customer i]
            "pmf": pmf.tolist(),
            "exact": exact,
            "exact_enumeration": float(distribution[ones >= threshold].sum()),
            "statevector": statevector_value,
            "statevector_abs_error": abs(statevector_value - exact),
            "statevector_max_abs_outcome_error": float(np.abs(state - distribution).max()),
            "sampled": {
                "estimate": sampled["estimate"],
                "shots": sampled["shots"],
                "seed": int(config.EXT_SEED + index),
                "standard_error": sampled["standard_error"],
                "error": sampled["estimate"] - exact,
                "abs_error": abs(sampled["estimate"] - exact),
                "marginals": sampled["marginals"],
                "max_abs_marginal_deviation": float(marginal_gap.max()),
                "max_marginal_z": float((marginal_gap / np.sqrt(p * (1.0 - p) / sampled["shots"])).max()),
                "mean_pairwise_correlation": mean_off_diagonal(
                    correlation_from_distribution(empirical_distribution(counts, n), n)
                ),
                "revenue": revenue_stats_from_counts(counts, n, deposits, costs, config.REVENUE_TAIL),
                "counts": counts,
            },
            "probability_none_or_one": float(pmf[0] + pmf[1]),
            "expected_subscribers": mean_count,
            "variance_subscribers": variance_count,
            "mean_pairwise_correlation": mean_off_diagonal(pairwise_correlation(p, rho, m)),
            "revenue": revenue_stats(distribution, deposits, costs, config.REVENUE_TAIL),
        })
    return {
        "model": "one-factor Gaussian copula: market factor on 2^m equally likely scenarios "
                 "z_k = Phi^-1((k + 1/2) / 2^m), p_i(z_k) = Phi((c_i + sqrt(rho) z_k) / sqrt(1 - rho)) "
                 "with c_i solved so that mean_k p_i(z_k) = p_i, customers independent given the factor",
        "references": list(REFERENCES),
        "ids": list(ids),
        "probabilities": p.tolist(),
        "deposits": deposits.tolist(),
        "costs": costs.tolist(),
        "threshold": int(threshold),
        "factor_qubits": int(m),
        "factor_values": factor_values(m).tolist(),
        "correlation_grid": [float(r) for r in config.CORRELATION_GRID],
        "revenue_tail": float(config.REVENUE_TAIL),
        "shots": int(config.SHOTS_CORRELATED),
        "bit_order": "factor qubit j = bit j of the scenario index k; count key rightmost character = customer 0",
        "circuit_rho": float(rho_max),
        "circuit": facts,
        "scenarios": scenarios,
    }
