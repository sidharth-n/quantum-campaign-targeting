"""Stage 3: exact references, circuit, bit order, readout mitigation and the counter circuit."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from campaign import config, hardware, stage3


@pytest.mark.parametrize("seed", range(5))
def test_dynamic_programme_equals_enumeration_and_statevector(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.9, 6)
    for threshold in (1, 3, 6):
        exact = stage3.exact_at_least(p, threshold)
        assert exact == pytest.approx(stage3.enumerate_at_least(p, threshold), abs=1e-13)
        assert exact == pytest.approx(stage3.statevector_at_least(p, threshold), abs=1e-12)
    assert stage3.poisson_binomial_pmf(p).sum() == pytest.approx(1.0, abs=1e-13)


def test_equal_probabilities_reduce_to_the_binomial():
    from scipy.stats import binom
    assert stage3.exact_at_least([0.3] * 6, 3) == pytest.approx(binom.sf(2, 6, 0.3), abs=1e-13)


def test_ry_angle_gives_the_customer_probability():
    p = np.array([0.05, 0.2, 0.37, 0.5, 0.81, 0.999])
    np.testing.assert_allclose(np.sin(stage3.angles(p) / 2) ** 2, p, atol=1e-13)
    state = Statevector(stage3.campaign_circuit(p, measure=False))
    for qubit in range(6):
        assert state.probabilities([qubit])[1] == pytest.approx(p[qubit], abs=1e-12)


def test_decoding_keeps_customer_i_on_qubit_i():
    """Six clearly different probabilities: each measured marginal must sit on its own customer."""
    p = np.array([0.05, 0.2, 0.35, 0.5, 0.7, 0.9])
    shots = 40000
    counts = hardware.run_counts(hardware.ideal_simulator(), stage3.campaign_circuit(p), shots)
    estimate = stage3.estimate_from_counts(counts, 6, 3)
    tolerance = 5 * np.sqrt(p * (1 - p) / shots)
    assert (np.abs(np.array(estimate["marginals"]) - p) < tolerance).all()
    assert (np.abs(np.array(estimate["marginals"])[::-1] - p) > tolerance).any(), "reversed order must not fit"
    assert abs(estimate["estimate"] - stage3.exact_at_least(p, 3)) < 5 * estimate["standard_error"]


def test_decode_reverses_the_qiskit_key():
    outcomes, shots = stage3.decode({"000001": 3, "100000": 5}, 6)
    assert outcomes[0].tolist() == [1, 0, 0, 0, 0, 0] and shots[0] == 3     # rightmost character = qubit 0
    assert outcomes[1].tolist() == [0, 0, 0, 0, 0, 1] and shots[1] == 5


def test_readout_mitigation_inverts_a_known_confusion_exactly():
    """Push the exact distribution through per-qubit readout errors, then undo them."""
    p = np.array([0.46, 0.42, 0.28, 0.26, 0.23, 0.22])
    n = len(p)
    X = stage3.all_bitstrings(n)
    true = np.prod(np.where(X == 1, p, 1 - p), axis=1)
    e01 = np.array([0.01, 0.02, 0.015, 0.03, 0.005, 0.012])     # P(read 1 | 0) per qubit
    e10 = np.array([0.025, 0.04, 0.03, 0.02, 0.05, 0.035])      # P(read 0 | 1) per qubit
    matrices = np.array([[[1 - a, b], [a, 1 - b]] for a, b in zip(e01, e10)])
    observed = true.reshape((2,) * n)
    for qubit in range(n):
        axis = n - 1 - qubit
        observed = np.moveaxis(np.tensordot(matrices[qubit], observed, axes=([1], [axis])), 0, axis)
    observed = observed.reshape(-1)
    scale = 10 ** 9
    counts = {format(b, f"0{n}b"): observed[b] * scale for b in range(2 ** n)}
    quasi = stage3.mitigate(counts, matrices)
    np.testing.assert_allclose(quasi, true, rtol=0, atol=1e-12)
    assert stage3.quasi_at_least(quasi, n, 3) == pytest.approx(stage3.exact_at_least(p, 3), abs=1e-12)
    # and the unmitigated value is the one predicted by shifting each probability
    shifted = p * (1 - e10) + (1 - p) * e01
    noisy_value = observed[X.sum(axis=1) >= 3].sum()
    assert noisy_value == pytest.approx(stage3.exact_at_least(shifted, 3), abs=1e-12)


def test_confusion_matrices_from_calibration_counts():
    matrices = stage3.confusion_matrices({"00": 90, "01": 10}, {"11": 80, "10": 20}, 2)
    # keys show qubit 0 on the right. qubit 0: P(1|0) = 0.1, P(1|1) = 0.8 ; qubit 1: P(1|0) = 0.0, P(1|1) = 1.0
    np.testing.assert_allclose(matrices[0], [[0.9, 0.2], [0.1, 0.8]])
    np.testing.assert_allclose(matrices[1], [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("threshold", [1, 3, 5])
def test_counter_circuit_flag_probability_is_the_exact_answer(threshold):
    p = np.random.default_rng(threshold).uniform(0.1, 0.8, 6)
    circuit = stage3.counter_circuit(p, threshold, measure=False)
    flag = circuit.qubits.index(circuit.qregs[2][0])
    assert circuit.qregs[2].name == "flag"
    probability = Statevector(circuit).probabilities([flag])[1]
    assert probability == pytest.approx(stage3.exact_at_least(p, threshold), abs=1e-10)


def test_sensitivity_is_the_derivative():
    p = np.array([0.46, 0.42, 0.28, 0.26, 0.23, 0.22])
    gradient = stage3.sensitivity(p, 3)
    step = 1e-6
    for i in range(6):
        up, down = p.copy(), p.copy()
        up[i] += step
        down[i] -= step
        numeric = (stage3.exact_at_least(up, 3) - stage3.exact_at_least(down, 3)) / (2 * step)
        assert gradient[i] == pytest.approx(numeric, abs=1e-8)


def test_monte_carlo_is_seeded_and_close():
    p = [0.46, 0.42, 0.28, 0.26, 0.23, 0.22]
    first = stage3.monte_carlo(p, 3, 200_000, config.MC_SEED)
    assert first == stage3.monte_carlo(p, 3, 200_000, config.MC_SEED)
    assert abs(first["estimate"] - stage3.exact_at_least(p, 3)) < 5 * first["standard_error"]


def test_noise_model_readout_matrix_rows_are_prepared_states():
    matrix = hardware.readout_matrix()
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 1] == config.NOISE["readout_p1_given_0"] and matrix[1, 0] == config.NOISE["readout_p0_given_1"]
    shifted = stage3.readout_shifted([0.0, 1.0])
    assert shifted[0] == config.NOISE["readout_p1_given_0"] and shifted[1] == 1 - config.NOISE["readout_p0_given_1"]
