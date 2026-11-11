"""The emulators used for training must be the Qiskit circuits, and the circuits must do what they claim."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from campaign import config, qaoa, qubo


@pytest.fixture(scope="module")
def ham(problem):
    exact = qubo.brute_force(problem)
    tuned = qubo.tune_unbalanced(problem, exact, config.CARDINALITY_GRID, config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID)
    return qaoa.hamiltonian(qubo.unbalanced_qubo(problem, tuned["A"], tuned["lambda_1"], tuned["lambda_2"]))


def overlap(a, b):
    return abs(np.vdot(a, b))


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_penalty_qaoa_emulator_equals_qiskit_statevector(ham, layers):
    params = np.random.default_rng(layers).uniform(0, 1.2, 2 * layers)
    circuit = qaoa.bind(qaoa.qaoa_circuit(ham, layers, measure=False), params)
    reference = Statevector(circuit).data
    emulated = qaoa.emulate(ham.diagonal, params)
    assert overlap(reference, emulated) == pytest.approx(1.0, abs=1e-10)      # equal up to a global phase
    np.testing.assert_allclose(np.abs(emulated) ** 2, np.abs(reference) ** 2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_constraint_preserving_emulator_equals_qiskit_statevector(problem, layers):
    model = qubo.unbalanced_qubo(problem, 0.0, 0.4, 0.3)          # cost with ZZ terms, no cardinality term
    ham = qaoa.hamiltonian(model)
    params = np.random.default_rng(10 + layers).uniform(0, 1.2, 2 * layers)
    circuit = qaoa.bind(qaoa.xy_circuit(ham, layers, problem.k, measure=False), params)
    reference = Statevector(circuit).data
    emulated = qaoa.emulate_xy(ham.diagonal, params, problem.k)
    assert overlap(reference, emulated) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(np.abs(emulated) ** 2, np.abs(reference) ** 2, rtol=0, atol=1e-12)


def test_constraint_preserving_circuit_only_outputs_exactly_k_ones(problem):
    ham = qaoa.hamiltonian(qubo.unbalanced_qubo(problem, 0.0, 0.0, 0.0, (False, False)))
    params = np.random.default_rng(5).uniform(0, 1.5, 6)
    probabilities = np.abs(Statevector(qaoa.bind(qaoa.xy_circuit(ham, 3, problem.k, measure=False), params)).data) ** 2
    ones = qubo.all_bitstrings(problem.n).sum(axis=1)
    assert probabilities[ones != problem.k].sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n,k", [(2, 1), (4, 2), (5, 2), (6, 3), (7, 5), (10, 4), (10, 6), (6, 6), (5, 0)])
def test_dicke_circuit_prepares_the_dicke_state(n, k):
    np.testing.assert_allclose(Statevector(qaoa.dicke_circuit(n, k)).data, qaoa.dicke_state(n, k), rtol=0, atol=1e-12)


def test_hamiltonian_diagonal_is_the_scaled_qubo_energy(ham):
    np.testing.assert_allclose(ham.diagonal * ham.scale + ham.constant, ham.energies, rtol=0, atol=1e-9)
    assert max(np.abs(ham.h).max(), np.abs(ham.J).max()) == pytest.approx(1.0)


def test_zero_angles_leave_the_uniform_superposition(ham):
    probabilities = np.abs(qaoa.emulate(ham.diagonal, np.zeros(6))) ** 2
    np.testing.assert_allclose(probabilities, 1.0 / len(ham.diagonal), atol=1e-12)


def test_bind_assigns_by_name_not_by_position(ham):
    circuit = qaoa.qaoa_circuit(ham, 2, measure=False)
    params = np.array([0.11, 0.22, 0.33, 0.44])                    # gamma0, gamma1, beta0, beta1
    bound = qaoa.bind(circuit, params)
    assert not bound.parameters
    rx_angles = [float(inst.operation.params[0]) for inst in bound.data if inst.operation.name == "rx"]
    assert rx_angles[0] == pytest.approx(2 * 0.33) and rx_angles[-1] == pytest.approx(2 * 0.44)


def test_cvar_limits():
    energies = np.array([3.0, -1.0, 2.0, 0.0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert qaoa.cvar(probabilities, energies, 1.0) == pytest.approx(probabilities @ energies)
    assert qaoa.cvar(probabilities, energies, 0.2) == pytest.approx(-1.0)          # exactly the best state
    assert qaoa.cvar(probabilities, energies, 0.4) == pytest.approx((0.2 * -1.0 + 0.2 * 0.0) / 0.4)


def test_training_is_deterministic_and_beats_uniform_sampling(problem, ham):
    exact = qubo.brute_force(problem)
    first = qaoa.train(ham, 1, config.QAOA_CVAR_ALPHA, 3, 150, config.QAOA_SEED)
    second = qaoa.train(ham, 1, config.QAOA_CVAR_ALPHA, 3, 150, config.QAOA_SEED)
    assert first["params"].tolist() == second["params"].tolist()
    probability = qaoa.exact_distribution(ham, first["params"])[exact["index"]]
    assert probability > 1.0 / 1024


def test_interpolated_start_keeps_the_schedule_shape():
    stretched = qaoa.interpolate(np.array([0.2, 0.6, 0.9, 0.3]), 3)
    assert len(stretched) == 6
    assert stretched[0] == pytest.approx(0.2) and stretched[2] == pytest.approx(0.6)
    assert stretched[3] == pytest.approx(0.9) and stretched[5] == pytest.approx(0.3)
