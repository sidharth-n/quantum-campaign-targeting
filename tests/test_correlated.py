"""Correlated customers: copula calibration, exact references, circuit bit order and revenue at risk."""
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import Statevector
from scipy.special import ndtri

from campaign import config, correlated, stage3
from campaign.qubo import all_bitstrings

ROOT = Path(__file__).resolve().parents[1]
M = config.FACTOR_QUBITS
RHOS = (0.1, 0.3, 0.5, 0.9)


def random_p(seed: int, n: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.02, 0.9, n)


def product_distribution(p) -> np.ndarray:
    X = all_bitstrings(len(p))
    return np.prod(np.where(X == 1, p, 1.0 - p), axis=1)


@pytest.fixture(scope="module")
def real_inputs():
    s3 = json.loads((ROOT / "stage3_simulation.json").read_text())
    s2 = json.loads((ROOT / "stage2_selection.json").read_text())
    by_id = {row["customer_id"]: row for row in s2["candidates"]}
    ids = list(s3["selected_ids"])
    return {
        "ids": ids,
        "p": np.array(s3["probabilities"], dtype=float),
        "deposits": np.array([by_id[i]["expected_deposit_value"] for i in ids], dtype=float),
        "costs": np.array([by_id[i]["contact_cost"] for i in ids], dtype=float),
        "threshold": 3,
        "exact_independent": float(s3["exact_probability"]),
        "objective": float(s2["objective_value"]),
    }


@pytest.fixture(scope="module")
def real_run(real_inputs):
    r = real_inputs
    return correlated.run(r["ids"], r["p"], r["deposits"], r["costs"], r["threshold"])


# --------------------------------------------------------------------------- 1. rho = 0 is the independent model
@pytest.mark.parametrize("seed", range(4))
def test_zero_correlation_reproduces_the_independent_model(seed):
    p = random_p(seed)
    for threshold in (1, 3, 5):
        assert correlated.exact_at_least(p, 0.0, M, threshold) == pytest.approx(
            stage3.exact_at_least(p, threshold), abs=1e-12)
    np.testing.assert_allclose(correlated.outcome_distribution(p, 0.0, M), product_distribution(p), rtol=0, atol=1e-14)
    q = correlated.conditional_probabilities(p, 0.0, M)
    np.testing.assert_allclose(q, np.tile(p, (2 ** M, 1)), rtol=0, atol=1e-14)
    np.testing.assert_allclose(correlated.thresholds(p, 0.0, M), ndtri(p), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- 2. marginals are the stage 1 probabilities
def test_factor_values_are_symmetric_midpoint_quantiles():
    z = correlated.factor_values(M)
    assert len(z) == 2 ** M and (np.diff(z) > 0).all()
    np.testing.assert_allclose(z, -z[::-1], atol=1e-14)
    assert z[0] == pytest.approx(ndtri(0.5 / 2 ** M), abs=1e-15)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_scenario_average_equals_the_customer_probability(seed, m):
    p = random_p(seed)
    for rho in RHOS:
        q = correlated.conditional_probabilities(p, rho, m)
        assert q.shape == (2 ** m, len(p))
        np.testing.assert_allclose(q.mean(axis=0), p, rtol=0, atol=1e-10)
        assert (np.diff(q, axis=0) > 0).all(), "a higher factor must raise every customer's probability"


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        correlated.conditional_probabilities([0.2, 1.0], 0.3, M)
    with pytest.raises(ValueError):
        correlated.conditional_probabilities([0.2, 0.4], 1.0, M)


# --------------------------------------------------------------------------- 3. circuit and bit order
def test_ucry_convention_target_first_control_zero_least_significant():
    """UCRYGate(angles) on [target, c_0, c_1]: angle k is applied when the controls hold k = c_0 + 2 c_1."""
    angles = [0.3, 0.9, 1.7, 2.6]
    for k in range(4):
        qc = QuantumCircuit(3)
        for j in range(2):
            if (k >> j) & 1:
                qc.x(j)
        qc.append(UCRYGate(angles), [2, 0, 1])
        assert Statevector(qc).probabilities([2])[1] == pytest.approx(np.sin(angles[k] / 2) ** 2, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_statevector_equals_the_exact_outcome_distribution(seed):
    p = np.sort(random_p(seed))[::-1] * np.linspace(1.0, 0.5, 6)    # clearly different, descending
    for rho in (0.0, *RHOS):
        exact = correlated.outcome_distribution(p, rho, M)
        state = correlated.statevector_distribution(p, rho, M)
        np.testing.assert_allclose(state, exact, rtol=0, atol=1e-10)
    # the reversed customer order is a different distribution, so the check above can fail
    reversed_order = correlated.outcome_distribution(p[::-1], 0.3, M)
    assert np.abs(reversed_order - correlated.outcome_distribution(p, 0.3, M)).max() > 1e-3


@pytest.mark.parametrize("seed", range(3))
def test_factor_register_holds_the_scenario_index(seed):
    """Joint distribution of both registers: index = k + 2^m b, with factor qubit j = bit j of k.

    Relabelling the equally likely scenarios leaves the customer marginal unchanged, so this joint
    check is the one that pins the scenario-to-angle mapping."""
    p = random_p(seed)
    n, rho = len(p), 0.4
    q = correlated.conditional_probabilities(p, rho, M)
    X = all_bitstrings(n)
    joint = np.prod(np.where(X[None, :, :] == 1, q[:, None, :], 1.0 - q[:, None, :]), axis=2) / 2 ** M   # [k, b]
    circuit = correlated.correlated_circuit(p, rho, M, measure=False)
    assert [reg.name for reg in circuit.qregs] == ["factor", "customer"]
    measured = Statevector(circuit).probabilities().reshape(2 ** n, 2 ** M).T                         # [k, b]
    np.testing.assert_allclose(measured, joint, rtol=0, atol=1e-12)
    flipped = joint[[int(format(k, f"0{M}b")[::-1], 2) for k in range(2 ** M)]]   # factor bits in the wrong order
    assert np.abs(measured - flipped).max() > 1e-3


def test_measured_keys_decode_to_customers():
    circuit = correlated.correlated_circuit([0.3, 0.6], 0.2, 2)
    assert circuit.num_clbits == 2 and circuit.cregs[0].name == "outcome"
    measures = [inst for inst in circuit.data if inst.operation.name == "measure"]
    for inst in measures:
        qubit = circuit.find_bit(inst.qubits[0]).index
        clbit = circuit.find_bit(inst.clbits[0]).index
        assert qubit == 2 + clbit                     # customer i (after 2 factor qubits) -> clbit i
    distribution = correlated.empirical_distribution({"000001": 3, "100000": 1}, 6)
    assert distribution[1] == 0.75 and distribution[32] == 0.25


# --------------------------------------------------------------------------- 4. pmf and tail
@pytest.mark.parametrize("seed", range(4))
def test_pmf_matches_the_outcome_distribution(seed):
    p = random_p(seed)
    ones = all_bitstrings(len(p)).sum(axis=1)
    for rho in (0.0, *RHOS):
        pmf = correlated.exact_pmf(p, rho, M)
        distribution = correlated.outcome_distribution(p, rho, M)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-13)
        assert distribution.sum() == pytest.approx(1.0, abs=1e-13)
        np.testing.assert_allclose(pmf, np.bincount(ones, weights=distribution), rtol=0, atol=1e-13)
        for threshold in (1, 3, 6):
            assert correlated.exact_at_least(p, rho, M, threshold) == pytest.approx(
                distribution[ones >= threshold].sum(), abs=1e-13)


# --------------------------------------------------------------------------- 5. correlation widens the count
@pytest.mark.parametrize("seed", range(4))
def test_positive_correlation_widens_the_subscriber_count(seed):
    p = random_p(seed)
    n = len(p)
    off_diagonal = ~np.eye(n, dtype=bool)
    assert np.abs(correlated.pairwise_correlation(p, 0.0, M)[off_diagonal]).max() < 1e-12
    variances = []
    for rho in (0.0, *RHOS):
        mean, variance = correlated.count_moments(correlated.exact_pmf(p, rho, M))
        assert mean == pytest.approx(p.sum(), abs=1e-12)
        variances.append(variance)
        if rho > 0:
            assert (correlated.pairwise_correlation(p, rho, M)[off_diagonal] > 0).all()
    assert variances[0] == pytest.approx((p * (1 - p)).sum(), abs=1e-12)
    assert (np.diff(variances) > 1e-6).all()


# --------------------------------------------------------------------------- 6. revenue at risk
def test_revenue_stats_hand_computed_with_an_atom_at_the_quantile():
    # outcomes (bit 0 = customer 0): 00 -> -7, 10 -> 3, 01 -> 13, 11 -> 23
    stats = correlated.revenue_stats([0.03, 0.04, 0.50, 0.43], [10, 20], [3, 4], 0.05)
    assert stats["expected_net_value"] == pytest.approx(16.3, abs=1e-12)
    assert stats["standard_deviation"] == pytest.approx(np.sqrt(313.8 - 16.3 ** 2), abs=1e-12)
    assert stats["probability_of_loss"] == pytest.approx(0.03, abs=1e-15)
    assert stats["value_at_risk"] == 3.0                         # P(V <= -7) = 0.03 < 0.05 <= P(V <= 3) = 0.07
    assert stats["expected_shortfall"] == pytest.approx((0.03 * -7 + 0.02 * 3) / 0.05, abs=1e-12)   # = -3
    assert (stats["min_net_value"], stats["max_net_value"]) == (-7.0, 23.0)
    # two outcomes with the same value form one atom: 10 -> 3 and 01 -> 3
    tied = correlated.revenue_stats([0.02, 0.01, 0.03, 0.94], [10, 10], [3, 4], 0.05)
    assert tied["value_at_risk"] == 3.0
    assert tied["expected_shortfall"] == pytest.approx((0.02 * -7 + 0.03 * 3) / 0.05, abs=1e-12)   # = -1
    # the tail level falls exactly on the end of the lowest atom
    edge = correlated.revenue_stats([0.05, 0.05, 0.45, 0.45], [10, 20], [3, 4], 0.05)
    assert edge["value_at_risk"] == -7.0 and edge["expected_shortfall"] == pytest.approx(-7.0, abs=1e-12)
    # an outcome with zero probability is not part of the support
    assert correlated.revenue_stats([0.0, 0.5, 0.5, 0.0], [10, 20], [3, 4], 0.05)["min_net_value"] == 3.0


def greedy_tail(distribution, net, tail):
    """Independent reference: walk the outcomes from worst to best and fill the tail mass."""
    order = np.argsort(net, kind="stable")
    taken, total, var = 0.0, 0.0, None
    for b in order:
        if distribution[b] == 0:
            continue
        take = min(distribution[b], tail - taken)
        total += take * net[b]
        taken += take
        if taken >= tail - 1e-12:
            var = net[b]
            break
    return var, total / tail


@pytest.mark.parametrize("seed", range(5))
def test_value_at_risk_and_shortfall_match_a_greedy_tail_fill(seed):
    rng = np.random.default_rng(seed)
    p = random_p(seed)
    deposits = rng.integers(12, 31, 6).astype(float)
    costs = rng.integers(1, 5, 6).astype(float)
    net = correlated.net_values(deposits, costs)
    for rho in (0.0, 0.3):
        distribution = correlated.outcome_distribution(p, rho, M)
        for tail in (0.01, 0.05, 0.2, 0.5):
            stats = correlated.revenue_stats(distribution, deposits, costs, tail)
            var, shortfall = greedy_tail(distribution, net, tail)
            assert stats["value_at_risk"] == var
            assert stats["expected_shortfall"] == pytest.approx(shortfall, abs=1e-10)
            assert stats["expected_shortfall"] <= stats["value_at_risk"] + 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_expected_net_value_does_not_depend_on_correlation(seed):
    rng = np.random.default_rng(seed)
    p = random_p(seed)
    deposits = rng.integers(12, 31, 6).astype(float)
    costs = rng.integers(1, 5, 6).astype(float)
    expected = float(p @ deposits - costs.sum())
    deviations = []
    for rho in (0.0, *RHOS):
        stats = correlated.revenue_stats(correlated.outcome_distribution(p, rho, M), deposits, costs, 0.05)
        assert stats["expected_net_value"] == pytest.approx(expected, abs=1e-10)
        deviations.append(stats["standard_deviation"])
    assert (np.diff(deviations) > 0).all()


# --------------------------------------------------------------------------- 7. the extension run on the real inputs
def test_run_is_deterministic_and_serialisable(real_inputs, real_run):
    r = real_inputs
    again = correlated.run(r["ids"], r["p"], r["deposits"], r["costs"], r["threshold"])
    assert json.dumps(real_run, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert real_run == again


def test_run_agrees_with_the_stage_files_and_with_exact_maths(real_inputs, real_run):
    r = real_inputs
    p = r["p"]
    assert real_run["ids"] == r["ids"] and real_run["probabilities"] == p.tolist()
    assert [s["rho"] for s in real_run["scenarios"]] == list(config.CORRELATION_GRID)
    independent = real_run["scenarios"][0]
    assert independent["rho"] == 0.0
    assert independent["exact"] == pytest.approx(r["exact_independent"], abs=1e-12)
    assert independent["mean_pairwise_correlation"] == pytest.approx(0.0, abs=1e-12)
    for s in real_run["scenarios"]:
        assert s["exact"] == pytest.approx(s["exact_enumeration"], abs=1e-13)
        assert s["statevector_abs_error"] < 1e-10 and s["statevector_max_abs_outcome_error"] < 1e-10
        assert s["revenue"]["expected_net_value"] == pytest.approx(r["objective"], abs=1e-9)
        assert s["probability_none_or_one"] == pytest.approx(s["pmf"][0] + s["pmf"][1], abs=1e-15)
        sampled = s["sampled"]
        assert sampled["shots"] == config.SHOTS_CORRELATED == sum(sampled["counts"].values())
        assert abs(sampled["error"]) < 5 * sampled["standard_error"]
        assert sampled["max_marginal_z"] < 5, "each sampled marginal must sit on its own stage 1 probability"
    circuit = real_run["circuit"]
    assert circuit["logical_qubits"] == M + len(p) and circuit["uniformly_controlled_rotations"] == len(p)
    assert circuit["two_qubit_gates_transpiled"] > 0 and circuit["depth_transpiled"] > circuit["depth_original"]
