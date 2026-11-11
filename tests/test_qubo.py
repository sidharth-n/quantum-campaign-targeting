"""Stage 2 formulation: exact solver, both QUBO encodings, the Ising map and bit order."""
import itertools

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from campaign import config, qubo


def random_problem(seed, n=8, k=4, tight=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "customer_id": [f"R{i:03d}" for i in range(n)],
        "predicted_probability": rng.uniform(0.05, 0.6, n),
        "expected_deposit_value": rng.integers(12, 31, n),
        "contact_cost": rng.integers(1, 5, n),
        "capacity_units": rng.integers(1, 5, n),
    })
    budget, capacity = (2.6 * k, 2.4 * k) if tight else (4.0 * k, 4.0 * k)
    return qubo.make_problem(frame, k, np.floor(budget), np.floor(capacity))


def reference_optimum(problem):
    """Independent brute force with itertools, no shared code with qubo.brute_force."""
    best = None
    for combo in itertools.combinations(range(problem.n), problem.k):
        idx = list(combo)
        if problem.cost[idx].sum() <= problem.budget and problem.capacity[idx].sum() <= problem.capacity_limit:
            value = problem.value[idx].sum()
            if best is None or value > best[0] + 1e-12:
                best = (value, combo)
    return best


# --------------------------------------------------------------------------- bit order
def test_bitstring_index_convention():
    X = qubo.all_bitstrings(3)
    assert X[1].tolist() == [1, 0, 0]          # index 1 -> qubit 0 set
    assert X[4].tolist() == [0, 0, 1]          # index 4 -> qubit 2 set
    for b in range(8):
        assert qubo.bits_to_index(X[b]) == b


def test_qiskit_key_round_trip_and_orientation():
    x = np.array([1, 0, 0, 1, 1, 0])
    key = qubo.qiskit_key(x)
    assert key == "011001"                      # qubit 0 is the rightmost character
    assert qubo.key_to_bits(key).tolist() == x.tolist()


def test_decoding_matches_what_qiskit_actually_returns():
    """Flip qubit 2 of five, measure, and decode the key Qiskit hands back."""
    circuit = QuantumCircuit(5)
    circuit.x(2)
    key = max(Statevector(circuit).probabilities_dict(), key=Statevector(circuit).probabilities_dict().get)
    assert key == "00100"
    assert qubo.key_to_bits(key).tolist() == [0, 0, 1, 0, 0]
    circuit = QuantumCircuit(5)
    circuit.x(0)
    key = max(Statevector(circuit).probabilities_dict(), key=Statevector(circuit).probabilities_dict().get)
    assert qubo.key_to_bits(key).tolist() == [1, 0, 0, 0, 0]


# --------------------------------------------------------------------------- exact solver
def test_brute_force_matches_an_independent_enumeration(problem):
    exact = qubo.brute_force(problem)
    value, combo = reference_optimum(problem)
    assert exact["value"] == pytest.approx(value, abs=1e-12)
    assert sorted(np.flatnonzero(exact["x"])) == list(combo)
    assert exact["n_subsets_of_size_k"] == 210


@pytest.mark.parametrize("seed", range(6))
def test_brute_force_on_random_problems(seed):
    problem = random_problem(seed, tight=seed % 2 == 0)
    reference = reference_optimum(problem)
    if reference is None:
        with pytest.raises(ValueError):
            qubo.brute_force(problem)
        return
    exact = qubo.brute_force(problem)
    assert exact["value"] == pytest.approx(reference[0], abs=1e-12)


# --------------------------------------------------------------------------- QUBO encodings
def test_unbalanced_qubo_energy_equals_its_formula(problem):
    A, l1, l2 = 3.1, 0.7, 0.4
    model = qubo.unbalanced_qubo(problem, A, l1, l2)
    X = qubo.all_bitstrings(problem.n).astype(float)
    g_budget = problem.budget - X @ problem.cost
    g_capacity = problem.capacity_limit - X @ problem.capacity
    direct = (-(X @ problem.value) + A * (X.sum(axis=1) - problem.k) ** 2
              - l1 * g_budget + l2 * g_budget ** 2 - l1 * g_capacity + l2 * g_capacity ** 2)
    np.testing.assert_allclose(model.energies(), direct, rtol=0, atol=1e-9)
    assert model.energy(X[37]) == pytest.approx(direct[37], abs=1e-9)


def test_tuned_unbalanced_qubo_ground_state_is_the_brute_force_optimum(problem):
    exact = qubo.brute_force(problem)
    tuned = qubo.tune_unbalanced(problem, exact, config.CARDINALITY_GRID, config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID)
    model = qubo.unbalanced_qubo(problem, tuned["A"], tuned["lambda_1"], tuned["lambda_2"])
    energies = model.energies()
    assert len(energies) == 1024
    assert int(np.argmin(energies)) == exact["index"]
    assert np.sort(energies)[1] - energies.min() > 1e-6, "the ground state must be unique"


def stress_instance(problem):
    """The stress instance of stage2.run."""
    from campaign import stage2
    return problem.with_limits(*stage2.stress_limits(problem, qubo.brute_force(problem)))


def test_stress_limits_bind_and_exclude_the_original_optimum(problem):
    free = qubo.brute_force(problem)
    tight = stress_instance(problem)
    exact = qubo.brute_force(tight)
    assert 0 < exact["n_feasible"] < exact["n_subsets_of_size_k"]
    assert free["cost"] > tight.budget or free["capacity"] > tight.capacity_limit
    assert set(exact["ids"]) != set(free["ids"])


@pytest.mark.parametrize("seed", range(8))
def test_stress_limits_hold_their_properties_on_random_problems(seed):
    """Whatever the instance, the rule must return binding limits that exclude the optimum, or raise."""
    from campaign import stage2
    problem = random_problem(200 + seed)
    free = qubo.brute_force(problem)
    try:
        budget, capacity = stage2.stress_limits(problem, free)
    except RuntimeError:
        return
    tight = problem.with_limits(budget, capacity)
    exact = qubo.brute_force(tight)
    assert exact["n_feasible"] >= 1
    assert free["cost"] > budget or free["capacity"] > capacity
    assert exact["index"] != free["index"]


@pytest.mark.parametrize("seed", range(6))
def test_unbalanced_tuning_on_random_instances_is_valid_or_says_it_is_not(seed):
    """Random instances: when the rule reports a valid setting, the QUBO ground state is the
    brute-force optimum; when it reports failure, it is not. Most random instances must be valid."""
    problem = random_problem(300 + seed)
    exact = qubo.brute_force(problem)
    tuned = qubo.tune_unbalanced(problem, exact, config.CARDINALITY_GRID, config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID)
    model = qubo.unbalanced_qubo(problem, tuned["A"], tuned["lambda_1"], tuned["lambda_2"])
    ground_is_optimum = int(np.argmin(model.energies())) == exact["index"]
    assert tuned["valid"] == ground_is_optimum
    assert tuned["valid"], "loose random instances should have a valid penalty setting on the grid"


def test_tuned_unbalanced_qubo_on_the_tightened_instance(problem):
    """Limits that bind: the rule must report honestly whether the optimum is the ground state."""
    free = qubo.brute_force(problem)
    tight = stress_instance(problem)
    exact = qubo.brute_force(tight)
    assert exact["n_feasible"] < exact["n_subsets_of_size_k"], "the stress instance must have binding limits"
    assert set(exact["ids"]) != set(free["ids"])
    tuned = qubo.tune_unbalanced(tight, exact, config.CARDINALITY_GRID, config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID)
    model = qubo.unbalanced_qubo(tight, tuned["A"], tuned["lambda_1"], tuned["lambda_2"])
    assert tuned["valid"] == (int(np.argmin(model.energies())) == exact["index"])
    assert tuned["valid"] == (tuned["selections_below_optimum"] == 0)


def test_tuning_reports_failure_instead_of_raising(problem):
    """A grid that cannot work (no cardinality penalty at all) must come back flagged, not crash."""
    exact = qubo.brute_force(problem)
    tuned = qubo.tune_unbalanced(problem, exact, (0.0,), (0.001,), (0.0,))
    assert tuned["valid"] is False and tuned["selections_below_optimum"] > 0


@pytest.mark.parametrize("tight", [False, True])
def test_slack_qubo_is_exact(problem, tight):
    if tight:
        problem = stress_instance(problem)
    exact = qubo.brute_force(problem)
    v_max = np.abs(problem.value).max()
    mu = config.SLACK_PENALTY * v_max
    model = qubo.slack_qubo(problem, config.SLACK_CARDINALITY * v_max, mu)
    energies = model.energies()
    decision = np.arange(len(energies)) & (2 ** problem.n - 1)
    assert int(decision[np.argmin(energies)]) == exact["index"]
    assert energies.min() == pytest.approx(-exact["value"], abs=1e-8)

    best = np.full(2 ** problem.n, np.inf)
    np.minimum.at(best, decision, energies)
    ev = qubo.evaluate(problem, qubo.all_bitstrings(problem.n))
    # feasible selections: some slack assignment removes every penalty
    np.testing.assert_allclose(best[ev["feasible"]], -ev["value"][ev["feasible"]], rtol=0, atol=1e-8)
    # selections of the right size that break a limit: at least one unit of violation, so at least mu
    broken = (ev["count"] == problem.k) & ~ev["feasible"]
    if broken.any():
        assert (best[broken] + ev["value"][broken] >= mu - 1e-8).all()


@pytest.mark.parametrize("upper", [0, 1, 2, 5, 9, 15, 16, 19])
def test_slack_coefficients_cover_exactly_the_range(upper):
    coefficients = qubo.slack_coefficients(upper)
    reachable = {sum(c for c, bit in zip(coefficients, bits) if bit)
                 for bits in itertools.product([0, 1], repeat=len(coefficients))}
    assert reachable == set(range(upper + 1))


@pytest.mark.parametrize("seed", [100, 101, 103, 105, 106, 107])
def test_random_instances_slack_encoding_recovers_the_optimum(seed):
    problem = random_problem(seed, tight=True)
    assert reference_optimum(problem) is not None, "seed chosen so that a feasible selection exists"
    exact = qubo.brute_force(problem)
    v_max = np.abs(problem.value).max()
    slack = qubo.slack_qubo(problem, config.SLACK_CARDINALITY * v_max, config.SLACK_PENALTY * v_max)
    assert qubo.check_ground_state(problem, slack, exact)["valid"]


# --------------------------------------------------------------------------- Ising map
def test_ising_energies_equal_qubo_energies_on_all_1024_states(problem):
    model = qubo.unbalanced_qubo(problem, 2.0, 0.3, 0.2)
    h, J, constant = qubo.to_ising(model)
    np.testing.assert_allclose(qubo.ising_energies(h, J, constant), model.energies(), rtol=0, atol=1e-9)
    assert np.allclose(J, np.triu(J, 1)), "couplings are stored once, upper triangular"


def test_ising_energies_equal_qubo_energies_for_the_slack_encoding(problem):
    v_max = np.abs(problem.value).max()
    model = qubo.slack_qubo(problem, 2 * v_max, 2 * v_max)
    h, J, constant = qubo.to_ising(model)
    np.testing.assert_allclose(qubo.ising_energies(h, J, constant), model.energies(), rtol=1e-12, atol=1e-6)


def test_can_bind_matches_enumeration(problem):
    exact = qubo.brute_force(problem)
    budget, capacity = qubo.can_bind(problem)
    assert budget == (exact["max_cost_any_k_subset"] > problem.budget)
    assert capacity == (exact["max_capacity_any_k_subset"] > problem.capacity_limit)
