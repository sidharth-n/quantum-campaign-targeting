"""Amplitude estimation extension: Grover identity, likelihood estimator, cost accounting, scaling."""
import json
from pathlib import Path

import numpy as np
import pytest

from campaign import amplitude, config, stage3

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", range(4))
def test_grover_power_gives_sin_squared_of_odd_multiples(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.8, 6)
    for threshold in (2, 3, 5):
        exact = stage3.exact_at_least(p, threshold)
        for power in (0, 1, 2, 3, 5):
            state = amplitude.good_probability_statevector(p, threshold, power)
            assert state == pytest.approx(amplitude.good_probability_theory(exact, power), abs=1e-10)


def test_power_zero_is_the_stage3_circuit():
    p = [0.58, 0.56, 0.35, 0.24, 0.19, 0.17]
    assert amplitude.good_probability_statevector(p, 3, 0) == pytest.approx(stage3.statevector_at_least(p, 3), abs=1e-12)
    assert amplitude.amplified_circuit(p, 3, 0, measure=False) == stage3.campaign_circuit(p, measure=False)


def test_good_states_follow_the_bit_convention():
    mask = amplitude.good_states(3, 2)
    # index b, bit i = qubit i: 3 = 011, 5 = 101, 6 = 110, 7 = 111 have two or more ones
    assert [int(b) for b in np.flatnonzero(mask)] == [3, 5, 6, 7]


def test_cost_counts_every_application_of_the_state_preparation():
    assert amplitude.grover_powers(3) == (0, 1, 2, 4)
    assert amplitude.applications((0, 1, 2), 64) == 64 * (1 + 3 + 5)


@pytest.mark.parametrize("amplitude_true", [0.05, 0.338494912988, 0.7])
def test_maximum_likelihood_recovers_the_amplitude_from_expected_counts(amplitude_true):
    powers = amplitude.grover_powers(5)
    shots = 1_000_000
    hits = [round(shots * amplitude.good_probability_theory(amplitude_true, k)) for k in powers]
    assert amplitude.maximum_likelihood(powers, hits, shots) == pytest.approx(amplitude_true, abs=2e-5)


def test_likelihood_peak_is_at_the_truth_for_exact_frequencies():
    powers = (0, 1, 2)
    phi = np.arcsin(np.sqrt(0.3))
    hits = [1000 * np.sin((2 * k + 1) * phi) ** 2 for k in powers]
    grid = np.linspace(0.01, np.pi / 2 - 0.01, 4001)
    values = amplitude.log_likelihood(grid, powers, hits, 1000)
    assert grid[int(np.argmax(values))] == pytest.approx(phi, abs=1e-3)


@pytest.fixture(scope="module")
def study():
    return json.loads((ROOT / "results" / "extensions" / "amplitude_estimation.json").read_text())


def test_committed_study_shows_the_quadratic_speed_up_on_the_ideal_simulator(study):
    assert study["identity_max_abs_difference"] < 1e-9
    assert study["slope_fit_points"] == len(config.AE_SCHEDULES) == len(study["schedules"])
    assert study["fitted_slope_log_rmse_vs_log_cost"] < -0.8
    assert study["fitted_slope_direct_sampling_measured"] == pytest.approx(-0.5, abs=0.08)
    best = study["best_schedule"]
    assert best["error_ratio_direct_over_amplitude_estimation"] > 3.0
    assert abs(best["bias"]) < 3 * best["rmse"] / np.sqrt(best["repeats"]) + 1e-4
    costs = [s["applications_of_A"] for s in study["schedules"]]
    assert costs == sorted(costs)
    for entry in study["schedules"]:
        assert entry["applications_of_A"] == amplitude.applications(entry["grover_powers"], config.AE_SHOTS_PER_POWER)
        # measured direct sampling agrees with its formula (300 repetitions: about 4% relative scatter)
        assert entry["direct_sampling_rmse_measured"] == pytest.approx(entry["direct_sampling_rmse_formula"], rel=0.15)
    # the report says amplitude estimation is the worse of the two at the smallest budget
    assert study["schedules"][0]["error_ratio_direct_over_amplitude_estimation"] < 1.0


def test_committed_study_measures_the_failure_under_noise(study):
    for entry in study["schedules"]:
        assert entry["noisy_rmse"] > 0.1                                   # the estimator returns a wrong number
        assert entry["noisy_direct_sampling_rmse_measured"] < 0.05
        assert entry["noisy_rmse"] > 3 * entry["noisy_direct_sampling_rmse_measured"]
    best = study["best_schedule"]
    assert abs(best["noisy_bias"]) > 0.9 * best["noisy_rmse"]              # "almost all of it bias"
    by_power = {entry["power"]: entry for entry in study["noisy"]}
    assert sorted(by_power) == sorted(amplitude.grover_powers(max(config.AE_SCHEDULES)))
    assert by_power[0]["two_qubit_gates_transpiled"] == 0
    assert abs(by_power[0]["shift"]) < 0.02                                # no two-qubit gates: only readout error
    assert abs(by_power[1]["shift"]) > 0.1                                 # one Grover step is already lost
    assert study["uniform_bits_good_probability"] == pytest.approx(42 / 64)
    assert study["uniform_bits_good_probability_after_readout"] == pytest.approx(
        stage3.exact_at_least(stage3.readout_shifted([0.5] * 6), 3), abs=1e-11)
    for power, entry in by_power.items():                                  # "powers 4 and above stay close to random bits"
        if power >= 4:
            assert abs(entry["good_probability_noisy"] - study["uniform_bits_good_probability_after_readout"]) < 0.015


def test_direct_estimates_are_independent_seeded_runs():
    from campaign import hardware
    circuit = stage3.campaign_circuit([0.3, 0.6, 0.2, 0.5])
    sim = hardware.ideal_simulator()
    first = amplitude.direct_estimates(sim, circuit, 4, 2, 200, 5, seed=11)
    again = amplitude.direct_estimates(sim, circuit, 4, 2, 200, 5, seed=11)
    np.testing.assert_array_equal(first, again)
    assert len(set(first.tolist())) > 1
    assert amplitude.rmse([0.1, 0.3], 0.2) == pytest.approx(0.1)
