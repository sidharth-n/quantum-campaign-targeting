"""Stage 2: choose exactly K of the N stage 1 candidates with QAOA, and check against brute force.

Pipeline of this stage
----------------------
  1. Candidates = the N highest stage 1 probabilities, in rank order. Candidate i sits on qubit i.
  2. Exact optimum by brute force over all 2^N bitstrings (classical reference).
  3. QUBO with unbalanced penalties (N qubits); penalty weights chosen by the rule in
     qubo.tune_unbalanced and verified against the brute-force optimum over all 2^N states.
  4. The slack-variable QUBO is built and verified too, as the exact but larger alternative.
  5. QAOA is trained on an exact emulator with a CVaR objective for several depths, sampled on
     Aer without noise, then transpiled to the device model and sampled with noise.
  6. Decoding: every measured bitstring is checked for feasibility and the feasible one with the
     highest expected net value is the selection (ties: more counts, then smaller basis index).
     The selection comes from the noiseless sample at depth config.QAOA_LAYERS.
  7. A second optimiser, the constraint-preserving QAOA (Dicke state + XY mixer), keeps "exactly
     K" inside the circuit instead of in the cost function. It is the cross-check for step 6 and
     the comparison between soft (penalty) and hard (mixer) constraint handling.
  8. A stress instance with tightened limits repeats the study, because the organisers' limits
     turn out not to bind for these candidates.
"""
import numpy as np
import pandas as pd

from . import config, hardware, qaoa, qubo


def candidates_table(predictions: pd.DataFrame, ops: pd.DataFrame, n: int) -> pd.DataFrame:
    """The top-n stage 1 rows joined with their stage 2 inputs, in rank order."""
    top = predictions[predictions["is_top_10"] == 1].sort_values("rank").head(n)
    table = top[["customer_id", "rank", "predicted_probability"]].join(ops, on="customer_id").reset_index(drop=True)
    table["expected_net_value"] = (
        table["predicted_probability"] * table["expected_deposit_value"] - table["contact_cost"]
    )
    table.insert(2, "qubit", np.arange(len(table)))
    return table


# --------------------------------------------------------------------------- decoding and metrics
def describe(problem: qubo.Problem, x) -> dict:
    ev = qubo.evaluate(problem, np.asarray(x)[None, :])
    return {
        "bitstring_qiskit": qubo.qiskit_key(x[: problem.n]),
        "x_by_qubit": "".join(str(int(b)) for b in x[: problem.n]),
        "ids": [problem.ids[i] for i in np.flatnonzero(x[: problem.n])],
        "n_selected": int(ev["count"][0]),
        "cost": float(ev["cost"][0]),
        "capacity": float(ev["capacity"][0]),
        "feasible": bool(ev["feasible"][0]),
        "value": float(ev["value"][0]),
    }


def sample_metrics(problem: qubo.Problem, exact: dict, energies: np.ndarray, counts: dict, top: int = 10) -> dict:
    """Decode a counts dictionary (Qiskit keys) and score it against the exact optimum."""
    keys = list(counts)
    X = np.array([qubo.key_to_bits(key) for key in keys], dtype=np.int8)
    shots = np.array([counts[key] for key in keys], dtype=float)
    total = shots.sum()
    index = X.astype(int) @ (1 << np.arange(X.shape[1]))
    ev = qubo.evaluate(problem, X)
    feasible = ev["feasible"]
    is_optimum = (index & (2 ** problem.n - 1)) == exact["index"]
    mean_energy = float(shots @ energies[index] / total)

    rows = []
    for r in np.lexsort((index, -shots))[:top]:
        row = describe(problem, X[r])
        row["count"] = int(shots[r])
        row["is_exact_optimum"] = bool(is_optimum[r])
        rows.append(row)

    best = None
    if feasible.any():
        candidates = np.flatnonzero(feasible)
        r = min(candidates, key=lambda i: (-float(ev["value"][i]), -shots[i], index[i]))    # full precision
        best = describe(problem, X[r])
        best["count"] = int(shots[r])
    return {
        "shots": int(total),
        "distinct_bitstrings": int(len(keys)),
        "p_optimal": float(shots[is_optimum].sum() / total),
        "feasible_fraction": float(shots[feasible].sum() / total),
        "p_optimal_given_feasible": (
            float(shots[is_optimum].sum() / shots[feasible].sum()) if feasible.any() else float("nan")
        ),
        "mean_value_of_feasible_over_optimum": (
            float(shots[feasible] @ ev["value"][feasible] / shots[feasible].sum() / exact["value"])
            if feasible.any() else float("nan")
        ),
        "energy_ratio": float((energies.max() - mean_energy) / (energies.max() - energies.min())),
        "best_feasible": best,
        "top_bitstrings": rows,
    }


def distribution_metrics(problem: qubo.Problem, exact: dict, energies: np.ndarray, probabilities: np.ndarray) -> dict:
    """The same quantities from the exact output distribution (no sampling error)."""
    X = qubo.all_bitstrings(int(round(np.log2(len(probabilities)))))
    ev = qubo.evaluate(problem, X)
    feasible = ev["feasible"]
    is_optimum = (np.arange(len(probabilities)) & (2 ** problem.n - 1)) == exact["index"]
    mean_energy = float(probabilities @ energies)
    return {
        "p_optimal": float(probabilities[is_optimum].sum()),
        "feasible_fraction": float(probabilities[feasible].sum()),
        "p_optimal_given_feasible": float(probabilities[is_optimum].sum() / probabilities[feasible].sum()),
        "mean_value_of_feasible_over_optimum": float(
            probabilities[feasible] @ ev["value"][feasible] / probabilities[feasible].sum() / exact["value"]
        ),
        "energy_ratio": float((energies.max() - mean_energy) / (energies.max() - energies.min())),
    }


def uniform_baseline(problem: qubo.Problem, exact: dict, shots: int) -> dict:
    """What uniformly random bitstrings would give: the reference for every QAOA number."""
    dim = 2 ** problem.n
    X = qubo.all_bitstrings(problem.n)
    ev = qubo.evaluate(problem, X)
    return {
        "p_optimal": 1.0 / dim,
        "feasible_fraction": float(ev["feasible"].mean()),
        "p_optimal_given_feasible": 1.0 / float(ev["feasible"].sum()),
        "mean_value_of_feasible_over_optimum": float(ev["value"][ev["feasible"]].mean() / exact["value"]),
        "p_optimum_seen_in_shots": float(1.0 - (1.0 - 1.0 / dim) ** shots),
    }


# --------------------------------------------------------------------------- QAOA study
def qaoa_study(problem: qubo.Problem, exact: dict, model: qubo.Qubo, depths, noisy: bool = True,
               preserve_cardinality: bool = False) -> dict:
    """Train, sample (ideal) and optionally sample (noisy) for each depth.

    preserve_cardinality=False: penalty QAOA (|+> start, X mixer).
    preserve_cardinality=True:  Dicke start and XY mixer; `model` then carries no cardinality term.
    """
    ham = qaoa.hamiltonian(model)
    ideal_sim = hardware.ideal_simulator()
    if preserve_cardinality:
        def emulator(params):
            return qaoa.emulate_xy(ham.diagonal, params, problem.k)
    else:
        def emulator(params):
            return qaoa.emulate(ham.diagonal, params)
    out = {}
    previous = None
    for depth in depths:
        trained = qaoa.train(
            ham, depth, config.QAOA_CVAR_ALPHA, config.QAOA_RESTARTS, config.QAOA_MAXITER,
            config.QAOA_SEED + depth, previous, emulator,
        )
        previous = trained["params"]
        template = qaoa.xy_circuit(ham, depth, problem.k) if preserve_cardinality else qaoa.qaoa_circuit(ham, depth)
        circuit = qaoa.bind(template, trained["params"])
        counts = hardware.run_counts(ideal_sim, circuit.decompose(reps=4), config.SHOTS_STAGE2)
        entry = {
            "layers": int(depth),
            "gamma": trained["params"][:depth].tolist(),
            "beta": trained["params"][depth:].tolist(),
            "cvar_objective": trained["objective"],
            "best_start": trained["best_start"],
            "best_start_kind": trained["best_start_kind"],
            "training_runs": trained["runs"],
            "exact_distribution": distribution_metrics(
                problem, exact, ham.energies, np.abs(emulator(trained["params"])) ** 2
            ),
            "ideal_sample": sample_metrics(problem, exact, ham.energies, counts),
            "ideal_counts": counts,
        }
        if noisy:
            device_circuit, stats = hardware.transpile_best(circuit)
            noisy_counts = hardware.run_noisy(device_circuit, config.SHOTS_STAGE2)
            entry["transpile"] = stats
            entry["noisy_sample"] = sample_metrics(problem, exact, ham.energies, noisy_counts)
            entry["noisy_counts"] = noisy_counts
        out[int(depth)] = entry
    return out


def encode_for_mixer(problem: qubo.Problem, exact: dict) -> dict:
    """Cost function for the constraint-preserving circuit: no cardinality term. A limit that no
    selection of K candidates can exceed is dropped (it is implied by the cardinality constraint);
    a limit that can bind keeps its unbalanced penalty, tuned inside the exactly-K subspace."""
    active = qubo.can_bind(problem)
    tuned = None
    if any(active):
        tuned = qubo.tune_unbalanced(
            problem, exact, (0.0,), config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID,
            within_cardinality=True, active=active,
        )
        model = qubo.unbalanced_qubo(problem, 0.0, tuned["lambda_1"], tuned["lambda_2"], active)
    else:
        model = qubo.unbalanced_qubo(problem, 0.0, 0.0, 0.0, (False, False))
    _, J, _ = qubo.to_ising(model)
    return {
        "model": model,
        "budget_penalty_active": bool(active[0]),
        "capacity_penalty_active": bool(active[1]),
        "tuned": tuned,
        "check": qubo.check_ground_state(problem, model, exact, within_cardinality=True),
        "zz_couplings": int(np.count_nonzero(J)),
    }


def _published(value: float) -> float:
    return float(f"{float(value):.12g}")


def encode(problem: qubo.Problem, exact: dict) -> dict:
    """Both QUBO encodings with their verification against the brute-force optimum."""
    tuned = qubo.tune_unbalanced(
        problem, exact, config.CARDINALITY_GRID, config.UNBALANCED_L2_GRID, config.UNBALANCED_SLACK_GRID
    )
    unbalanced = qubo.unbalanced_qubo(problem, tuned["A"], tuned["lambda_1"], tuned["lambda_2"])
    unbalanced_check = qubo.check_ground_state(problem, unbalanced, exact)
    h, J, constant = qubo.to_ising(unbalanced)
    # the check is made with the coefficients exactly as they are written to the result file
    # (12 significant digits), so the published numbers certify themselves
    h, J, constant = np.vectorize(_published)(h), np.vectorize(_published)(J), _published(constant)
    ising_error = float(np.abs(qubo.ising_energies(h, J, constant) - unbalanced.energies()).max())

    v_max = tuned["v_max"]
    slack = qubo.slack_qubo(problem, config.SLACK_CARDINALITY * v_max, config.SLACK_PENALTY * v_max)
    slack_energies = slack.energies()
    slack_check = qubo.check_ground_state(problem, slack, exact, slack_energies)
    # for every feasible selection the best slack assignment must cost exactly minus its value
    decision = np.arange(len(slack_energies)) & (2 ** problem.n - 1)
    best_per_selection = np.full(2 ** problem.n, np.inf)
    np.minimum.at(best_per_selection, decision, slack_energies)
    ev = qubo.evaluate(problem, qubo.all_bitstrings(problem.n))
    slack_exactness = float(np.abs(best_per_selection[ev["feasible"]] + ev["value"][ev["feasible"]]).max())
    slack_h, slack_J, _ = qubo.to_ising(slack)
    return {
        "tuned": tuned,
        "unbalanced": unbalanced,
        "unbalanced_check": unbalanced_check,
        "ising_max_abs_difference": ising_error,
        "ising": {"h": h.tolist(), "J": J.tolist(), "constant": constant},
        "unbalanced_couplings": int(np.count_nonzero(J)),
        "slack": slack,
        "slack_check": slack_check,
        "slack_feasible_energy_error": slack_exactness,
        "slack_couplings": int(np.count_nonzero(slack_J)),
    }


def stress_limits(problem: qubo.Problem, exact: dict) -> tuple:
    """Budget and capacity for the stress instance.

    Target: each limit is the smaller of (a) the STRESS_PERCENTILE-th percentile of what the
    groups of exactly K candidates need and (b) one unit less than the optimum uses. The limits
    returned are the pair closest to that target (sum of absolute differences; ties to the
    smaller budget, then the smaller capacity) for which both required properties hold: at
    least one group of K is feasible, and the original optimum is not. Raises if no pair does.
    """
    groups = qubo.evaluate(problem, qubo.all_bitstrings(problem.n))
    of_size_k = groups["count"] == problem.k
    cost, capacity = groups["cost"][of_size_k], groups["capacity"][of_size_k]
    target = (
        min(np.floor(np.percentile(cost, config.STRESS_PERCENTILE)), exact["cost"] - 1),
        min(np.floor(np.percentile(capacity, config.STRESS_PERCENTILE)), exact["capacity"] - 1),
    )
    best = None
    for budget in np.arange(cost.min(), cost.max() + 1):
        for limit in np.arange(capacity.min(), capacity.max() + 1):
            feasible = bool(((cost <= budget) & (capacity <= limit)).any())
            optimum_excluded = exact["cost"] > budget or exact["capacity"] > limit
            if feasible and optimum_excluded:
                key = (abs(budget - target[0]) + abs(limit - target[1]), budget, limit)
                best = key if best is None or key < best else best
    if best is None:
        raise RuntimeError("no stress instance exists: every pair of limits either allows the optimum or allows nothing")
    return float(best[1]), float(best[2])


def run(predictions: pd.DataFrame, ops: pd.DataFrame, constraints) -> dict:
    candidates = candidates_table(predictions, ops, constraints.n_candidates)
    problem = qubo.make_problem(candidates, constraints.n_select, constraints.budget, constraints.capacity)
    exact = qubo.brute_force(problem)
    encoding = encode(problem, exact)
    if not encoding["unbalanced_check"]["valid"]:
        raise RuntimeError("no penalty setting on the grid makes the brute-force optimum the QUBO ground state")
    study = qaoa_study(problem, exact, encoding["unbalanced"], config.QAOA_LAYER_SWEEP)
    mixer_encoding = encode_for_mixer(problem, exact)
    mixer_study = qaoa_study(problem, exact, mixer_encoding["model"], config.XY_LAYER_SWEEP, preserve_cardinality=True)

    chosen = study[config.QAOA_LAYERS]["ideal_sample"]["best_feasible"]
    if chosen is None:
        raise RuntimeError("QAOA returned no feasible bitstring; the selection cannot be made")
    selection = dict(chosen)
    # both values come from the same routine, so an identical selection gives a gap of exactly 0
    optimum_value = describe(problem, exact["x"])["value"]
    gap = float(optimum_value - chosen["value"])
    selection["optimality_gap"] = gap
    selection["relative_gap"] = gap / abs(optimum_value)
    selection["matches_brute_force"] = bool(set(chosen["ids"]) == set(exact["ids"]))
    selection["source"] = f"noiseless Aer sample, penalty QAOA depth p={config.QAOA_LAYERS}, {config.SHOTS_STAGE2} shots"
    cross_check = mixer_study[config.QAOA_LAYERS]["ideal_sample"]["best_feasible"]
    selection["constraint_preserving_qaoa_agrees"] = bool(
        cross_check is not None and set(cross_check["ids"]) == set(chosen["ids"])
    )

    # stress instance: limits tightened until they bind and the unconstrained optimum is infeasible
    stress_problem = problem.with_limits(*stress_limits(problem, exact))
    stress_exact = qubo.brute_force(stress_problem)
    stress_encoding = encode(stress_problem, stress_exact)
    stress_study = qaoa_study(
        stress_problem, stress_exact, stress_encoding["unbalanced"], (config.QAOA_LAYERS,), noisy=False
    )
    stress_mixer_encoding = encode_for_mixer(stress_problem, stress_exact)
    stress_mixer_study = qaoa_study(
        stress_problem, stress_exact, stress_mixer_encoding["model"], (config.QAOA_LAYERS,), noisy=False,
        preserve_cardinality=True,
    )

    return {
        "candidates": candidates,
        "problem": problem,
        "exact": exact,
        "encoding": encoding,
        "study": study,
        "mixer_encoding": mixer_encoding,
        "mixer_study": mixer_study,
        "selection": selection,
        "uniform": uniform_baseline(problem, exact, config.SHOTS_STAGE2),
        "binding": {
            "max_cost_of_any_selection": exact["max_cost_any_k_subset"],
            "budget": problem.budget,
            "budget_can_bind": bool(exact["max_cost_any_k_subset"] > problem.budget),
            "max_capacity_of_any_selection": exact["max_capacity_any_k_subset"],
            "capacity": problem.capacity_limit,
            "capacity_can_bind": bool(exact["max_capacity_any_k_subset"] > problem.capacity_limit),
            "feasible_selections": exact["n_feasible"],
            "selections_of_size_k": exact["n_subsets_of_size_k"],
        },
        "stress": {
            "problem": stress_problem,
            "exact": stress_exact,
            "encoding": stress_encoding,
            "study": stress_study,
            "mixer_encoding": stress_mixer_encoding,
            "mixer_study": stress_mixer_study,
            "uniform": uniform_baseline(stress_problem, stress_exact, config.SHOTS_STAGE2),
        },
    }
