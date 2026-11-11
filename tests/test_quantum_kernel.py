"""The NumPy feature-map routine must be the Qiskit circuit, and the kernel must be a valid kernel."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from campaign.models import feature_map_circuit, feature_map_states, fidelity_kernel, kernel_circuit


@pytest.mark.parametrize("n_qubits,reps", [(2, 1), (4, 1), (6, 1), (6, 2), (5, 3)])
def test_emulated_states_equal_qiskit_statevector(n_qubits, reps):
    rng = np.random.default_rng(n_qubits * 10 + reps)
    angles = rng.normal(scale=1.5, size=(6, n_qubits))
    circuit = feature_map_circuit(n_qubits, reps)
    parameters = list(circuit.parameters)
    reference = np.array([
        Statevector(circuit.assign_parameters(dict(zip(parameters, row)))).data for row in angles
    ])
    np.testing.assert_allclose(feature_map_states(angles, reps), reference, rtol=0, atol=1e-12)


def test_kernel_is_symmetric_unit_diagonal_and_positive_semidefinite():
    angles = np.random.default_rng(1).normal(size=(40, 6))
    states = feature_map_states(angles, 2)
    K = fidelity_kernel(states, states)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert ((K >= -1e-12) & (K <= 1 + 1e-12)).all()


def test_overlap_circuit_all_zeros_probability_is_the_kernel_entry():
    rng = np.random.default_rng(2)
    for reps in (1, 2):
        a, b = rng.normal(size=6), rng.normal(size=6)
        circuit = kernel_circuit(a, b, reps)
        circuit.remove_final_measurements()
        probability = abs(Statevector(circuit).data[0]) ** 2
        states = feature_map_states(np.array([a, b]), reps)
        assert probability == pytest.approx(fidelity_kernel(states[[0]], states[[1]])[0, 0], abs=1e-12)


def test_submitted_quantum_model_uses_training_angles_only(data, s1):
    quantum = s1.model.quantum_
    assert quantum.train_angles_.shape == (len(data.train), quantum.n_qubits)
