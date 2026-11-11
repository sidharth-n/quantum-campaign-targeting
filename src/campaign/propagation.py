"""Error propagation: how stage 1 uncertainty moves through stages 2 and 3.

The submitted stage 1 model is refitted on bootstrap resamples of the 300 training rows
(stratified by label, so every resample keeps the 35 subscribers' share). Each refit gives a new
set of 100 test probabilities, which are pushed through the same ranking, the same exact stage 2
solver and the same exact stage 3 formula. No test label is involved anywhere.

Three questions are answered:
  1. Stage 1 -> top 10: how often does each submitted candidate stay in the top 10?
  2. Stage 1 -> stage 2: with the ten candidates held fixed, how often is the submitted six still
     the optimum, and how much value is lost by keeping it when it is not (regret)?
  3. Stage 1 -> stage 3: with the six customers held fixed, how wide is the spread of
     P(at least 3), and does a first-order (delta method) calculation reproduce it?
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from . import config, qubo, stage3
from .data import FEATURE_COLUMNS
from .stage1 import rank_customers


def _refit(estimator, X, y, rows, X_test):
    p = clone(estimator).fit(X.iloc[rows], y[rows]).predict_proba(X_test)[:, 1]
    return np.round(p, config.PROBABILITY_DECIMALS)


def bootstrap_rows(y: np.ndarray, rounds: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [
        np.concatenate([rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))])
        for _ in range(rounds)
    ]


def run(data, estimator, predictions: pd.DataFrame, candidates: pd.DataFrame, selected_ids, n_jobs: int = -1) -> dict:
    X, y = data.train[FEATURE_COLUMNS], data.train["y"].to_numpy()
    X_test = data.test[FEATURE_COLUMNS]
    test_ids = data.test["customer_id"].to_numpy()
    constraints = data.constraints
    threshold = int(np.ceil(constraints.n_select / 2))
    resamples = bootstrap_rows(y, config.BOOTSTRAP_ROUNDS, config.BOOTSTRAP_SEED)
    probabilities = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_refit)(estimator, X, y, rows, X_test) for rows in resamples
    ))                                                    # (rounds, 100), columns in test-sheet order
    column = {cid: i for i, cid in enumerate(test_ids)}

    candidate_ids = list(candidates["customer_id"])
    selected_ids = list(selected_ids)
    candidate_columns = [column[c] for c in candidate_ids]
    selected_columns = [column[c] for c in selected_ids]
    submitted = predictions.set_index("customer_id")["predicted_probability"]

    in_top = np.zeros(len(candidate_ids))
    overlap, same_selection_free, regret, same_selection_fixed, p_fixed, p_free = [], [], [], [], [], []
    selected_free = np.zeros(len(selected_ids))
    for row in probabilities:
        ranked = rank_customers(test_ids, row, constraints.n_candidates)
        top = set(ranked.loc[ranked["is_top_10"] == 1, "customer_id"])
        in_top += np.array([c in top for c in candidate_ids])
        overlap.append(len(top & set(candidate_ids)))

        # (a) whole pipeline rerun on the replicate's own top 10
        own = ranked[ranked["is_top_10"] == 1].join(data.ops, on="customer_id")
        own_problem = qubo.make_problem(own, constraints.n_select, constraints.budget, constraints.capacity)
        own_exact = qubo.brute_force(own_problem)
        same_selection_free.append(set(own_exact["ids"]) == set(selected_ids))
        selected_free += np.array([c in own_exact["ids"] for c in selected_ids])
        p_free.append(stage3.exact_at_least(own_problem.probability[own_exact["x"] == 1], threshold))

        # (b) the ten submitted candidates held fixed, probabilities from the replicate
        fixed = candidates.copy()
        fixed["predicted_probability"] = row[candidate_columns]
        fixed_problem = qubo.make_problem(fixed, constraints.n_select, constraints.budget, constraints.capacity)
        fixed_exact = qubo.brute_force(fixed_problem)
        mask = np.array([c in selected_ids for c in candidate_ids], dtype=float)
        regret.append(fixed_exact["value"] - float(mask @ fixed_problem.value))
        same_selection_fixed.append(set(fixed_exact["ids"]) == set(selected_ids))

        # (c) the six submitted customers held fixed
        p_fixed.append(stage3.exact_at_least(row[selected_columns], threshold))

    six = probabilities[:, selected_columns]
    p_submitted = submitted.loc[selected_ids].to_numpy()
    gradient = stage3.sensitivity(p_submitted, threshold)
    covariance = np.cov(six, rowvar=False)
    delta_sd = float(np.sqrt(gradient @ covariance @ gradient))
    p_fixed, p_free, regret = np.array(p_fixed), np.array(p_free), np.array(regret)
    exact = stage3.exact_at_least(p_submitted, threshold)

    def spread(values):
        return {
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "p05": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "rounds": config.BOOTSTRAP_ROUNDS,
        "seed": config.BOOTSTRAP_SEED,
        "resampling": "bootstrap of the 300 training rows, stratified by label",
        "top10": {
            "mean_overlap_with_submitted_top10": float(np.mean(overlap)),
            "inclusion_frequency": dict(zip(candidate_ids, (in_top / len(probabilities)).tolist())),
        },
        "stage2": {
            "fixed_candidates_same_selection_frequency": float(np.mean(same_selection_fixed)),
            "fixed_candidates_regret": spread(regret),
            "fixed_candidates_regret_zero_frequency": float(np.mean(regret < 1e-9)),
            "full_rerun_same_selection_frequency": float(np.mean(same_selection_free)),
            "full_rerun_selected_inclusion_frequency": dict(zip(selected_ids, (selected_free / len(probabilities)).tolist())),
        },
        "stage3": {
            "submitted_exact": exact,
            "fixed_six": spread(p_fixed),
            "full_rerun": spread(p_free),
            "sensitivity": dict(zip(selected_ids, gradient.tolist())),
            "delta_method_sd": delta_sd,
            "probability_sd": dict(zip(selected_ids, six.std(axis=0, ddof=1).tolist())),
            "probability_p05": dict(zip(selected_ids, np.percentile(six, 5, axis=0).tolist())),
            "probability_p95": dict(zip(selected_ids, np.percentile(six, 95, axis=0).tolist())),
            "probability_submitted": dict(zip(selected_ids, p_submitted.tolist())),
        },
        "samples": {"p_fixed_six": p_fixed.tolist(), "p_full_rerun": p_free.tolist(), "regret": regret.tolist()},
    }


def model_comparison(data, all_model_predictions: pd.DataFrame, submitted: str) -> list:
    """What stages 2 and 3 would give with each stage 1 model's probabilities.

    Stages 2 and 3 are solved exactly here (brute force, Poisson-binomial), so the differences
    are caused by stage 1 alone. It needs test features only, no labels, and nothing was chosen
    from it: it shows how much of the final answer depends on which classifier is used.
    """
    constraints = data.constraints
    threshold = int(np.ceil(constraints.n_select / 2))
    rows = []
    reference = None
    for model in [submitted] + [c for c in all_model_predictions.columns if c not in ("customer_id", submitted)]:
        ranked = rank_customers(all_model_predictions["customer_id"], all_model_predictions[model], constraints.n_candidates)
        top = ranked[ranked["is_top_10"] == 1].join(data.ops, on="customer_id")
        problem = qubo.make_problem(top, constraints.n_select, constraints.budget, constraints.capacity)
        exact = qubo.brute_force(problem)
        entry = {
            "model": model,
            "submitted": model == submitted,
            "top10": list(top["customer_id"]),
            "selected_ids": exact["ids"],
            "objective_value": exact["value"],
            "budget_used": exact["cost"],
            "capacity_used": exact["capacity"],
            "feasible_selections": exact["n_feasible"],
            "p_at_least_3": stage3.exact_at_least(problem.probability[exact["x"] == 1], threshold),
        }
        reference = entry if reference is None else reference
        entry["top10_shared_with_submitted"] = len(set(entry["top10"]) & set(reference["top10"]))
        entry["selected_shared_with_submitted"] = len(set(entry["selected_ids"]) & set(reference["selected_ids"]))
        rows.append(entry)
    return rows
