"""Stage 1: subscription probabilities for the 100 test customers.

Everything here uses the 300 training rows only, through repeated stratified cross-validation
with the same folds for every model.

How the submitted model is chosen
---------------------------------
  1. Within each tuned family (logistic regression, quantum kernel SVM, Gaussian kernel SVM)
     keep the setting with the highest mean cross-validated PR-AUC. Settings within PR_AUC_TIE of
     the best count as tied and the lowest mean Brier score wins; any remaining tie goes to the
     earlier grid entry.
  2. The hybrid is the fixed 50/50 average of the tuned logistic regression and the tuned
     quantum kernel SVM. The weight is never tuned.
  3. The submitted model is the hybrid. This is a design decision (config.SUBMITTED_MODEL), not
     a score-based one: the challenge asks for a quantum or hybrid classifier and treats the
     classical model as the baseline to compare against.
  4. The hybrid is compared with the best classical model on the same folds, metric by metric
     (mean paired difference and its standard error). The comparison is reported as it comes
     out; it does not change step 3.

The submitted model is then fitted once on all 300 training rows and predicts the test rows once.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config, hardware
from .data import FEATURE_COLUMNS, Dataset
from .metrics import METRICS, calibration_line, expected_calibration_error, reliability_table, score
from .models import (
    Hybrid, boosting, feature_map_states, fidelity_kernel, kernel_circuit, logistic, quantum_kernel_svm,
    rbf_kernel_svm,
)


# --------------------------------------------------------------------------- cross-validation
def cv_splits(y):
    splitter = RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS, random_state=config.SEED
    )
    return list(splitter.split(np.zeros(len(y)), y))


def cross_validate(estimator, X: pd.DataFrame, y: np.ndarray, splits) -> dict:
    """Fit on each training fold, score on the held-out fold. Returns fold metrics and OOF predictions."""
    folds = []
    oof = np.full((config.CV_REPEATS, len(y)), np.nan)
    for index, (fit_rows, held_rows) in enumerate(splits):
        model = clone(estimator).fit(X.iloc[fit_rows], y[fit_rows])
        p = model.predict_proba(X.iloc[held_rows])[:, 1]
        folds.append(score(y[held_rows], p))
        oof[index // config.CV_SPLITS, held_rows] = p
    return {"folds": pd.DataFrame(folds), "oof": oof}


def summarise(name: str, family: str, params: dict, cv: dict, y: np.ndarray) -> dict:
    folds = cv["folds"]
    row = {"model": name, "family": family, "params": params}
    for metric in METRICS:
        row[metric] = float(folds[metric].mean())
        row[f"{metric}_se"] = float(folds[metric].std(ddof=1) / np.sqrt(len(folds)))
    pooled = cv["oof"].ravel()
    pooled_y = np.tile(y, config.CV_REPEATS)
    row["ece"] = expected_calibration_error(pooled_y, pooled)
    row["calibration_slope"], row["calibration_intercept"] = calibration_line(pooled_y, pooled)
    return row


# --------------------------------------------------------------------------- selection rule
def best_of(rows: list) -> dict:
    """The ordering of the rule: the top PR-AUC group, then lowest Brier, then earliest entry."""
    top = max(r["pr_auc"] for r in rows)
    tied = [r for r in rows if top - r["pr_auc"] <= config.PR_AUC_TIE]
    return min(tied, key=lambda r: r["brier"])


def paired_comparison(model_folds: pd.DataFrame, reference_folds: pd.DataFrame) -> dict:
    """Step 4: per-metric mean of the paired fold differences (model minus reference) and its SE."""
    out = {}
    for metric in METRICS:
        diff = model_folds[metric].to_numpy() - reference_folds[metric].to_numpy()
        out[metric] = {
            "difference": float(diff.mean()),
            "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        }
    return out


# --------------------------------------------------------------------------- stage 1 run
@dataclass
class Stage1Result:
    predictions: pd.DataFrame        # customer_id, predicted_probability, rank, is_top_10 (100 rows)
    all_model_predictions: pd.DataFrame
    cv_table: pd.DataFrame           # one row per evaluated setting
    selection: dict                  # which model was submitted and why
    reliability: dict                # pooled out-of-fold reliability tables for the finalists
    model: object                    # fitted submitted model
    estimator: object                # unfitted clone of the submitted model (for bootstrap refits)
    cv_folds: dict                   # fold metrics of the finalists


def rank_customers(customer_ids, probabilities, n_top: int) -> pd.DataFrame:
    """Rank high to low; ties go to the lower customer_id. Flags the first `n_top` rows."""
    frame = pd.DataFrame({"customer_id": list(customer_ids), "predicted_probability": np.asarray(probabilities, float)})
    frame = frame.sort_values(["predicted_probability", "customer_id"], ascending=[False, True], kind="mergesort")
    frame = frame.reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["is_top_10"] = (frame["rank"] <= n_top).astype(int)
    return frame


def run(data: Dataset, n_jobs: int = -1) -> Stage1Result:
    X = data.train[FEATURE_COLUMNS]
    y = data.train["y"].to_numpy()
    splits = cv_splits(y)

    # every candidate setting, declared in config.py
    jobs = [("logistic", "logistic", params, logistic(**params)) for params in config.LOGISTIC_GRID]
    jobs += [("quantum_kernel", "quantum_kernel", params, quantum_kernel_svm(**params)) for params in config.KERNEL_GRID]
    jobs += [("rbf_kernel", "rbf_kernel", params, rbf_kernel_svm(**params)) for params in config.RBF_GRID]
    jobs += [("boosting", "boosting", {}, boosting())]
    cvs = Parallel(n_jobs=n_jobs)(delayed(cross_validate)(est, X, y, splits) for _, _, _, est in jobs)
    rows = [summarise(name, family, params, cv, y) for (name, family, params, _), cv in zip(jobs, cvs)]
    by_row = {id(row): cv for row, cv in zip(rows, cvs)}

    # step 1: best setting per tuned family
    best_logistic = best_of([r for r in rows if r["family"] == "logistic"])
    best_kernel = best_of([r for r in rows if r["family"] == "quantum_kernel"])
    best_rbf = best_of([r for r in rows if r["family"] == "rbf_kernel"])
    boosting_row = next(r for r in rows if r["family"] == "boosting")

    # step 2: fixed hybrid of the two
    hybrid_estimator = Hybrid(logistic(**best_logistic["params"]), quantum_kernel_svm(**best_kernel["params"]))
    hybrid_cv = cross_validate(hybrid_estimator, X, y, splits)
    hybrid_row = summarise(
        "hybrid", "hybrid",
        {"weight": config.HYBRID_WEIGHT, "logistic": best_logistic["params"], "quantum_kernel": best_kernel["params"]},
        hybrid_cv, y,
    )
    rows.append(hybrid_row)
    by_row[id(hybrid_row)] = hybrid_cv

    finalists = {
        "logistic": (best_logistic, logistic(**best_logistic["params"])),
        "boosting": (boosting_row, boosting()),
        "rbf_kernel": (best_rbf, rbf_kernel_svm(**best_rbf["params"])),
        "quantum_kernel": (best_kernel, quantum_kernel_svm(**best_kernel["params"])),
        "hybrid": (hybrid_row, hybrid_estimator),
    }

    # step 3: the submitted model is fixed by design; step 4: report it against the best classical model
    classical_row = best_of([best_logistic, boosting_row, best_rbf])
    submitted_name = config.SUBMITTED_MODEL
    submitted_row, estimator = finalists[submitted_name]
    comparison = paired_comparison(by_row[id(submitted_row)]["folds"], by_row[id(classical_row)]["folds"])

    # final fit on all training rows, one prediction of the test rows
    X_test = data.test[FEATURE_COLUMNS]
    fitted = {name: clone(est).fit(X, y) for name, (_, est) in finalists.items()}
    # rounded once here; these exact values are what the CSV holds and what stages 2 and 3 use
    test_probabilities = {
        name: np.round(model.predict_proba(X_test)[:, 1], config.PROBABILITY_DECIMALS) for name, model in fitted.items()
    }
    predictions = rank_customers(
        data.test["customer_id"], test_probabilities[submitted_name], data.constraints.n_candidates
    )
    all_models = pd.DataFrame({"customer_id": data.test["customer_id"], **test_probabilities})
    all_models = all_models.sort_values("customer_id").reset_index(drop=True)

    cv_table = pd.DataFrame(rows)
    cv_table["params"] = cv_table["params"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
    cv_table["finalist"] = [any(r is f[0] for f in finalists.values()) for r in rows]
    cv_table["submitted"] = [r is submitted_row for r in rows]

    pooled_y = np.tile(y, config.CV_REPEATS)
    reliability = {
        name: reliability_table(pooled_y, by_row[id(row)]["oof"].ravel()) for name, (row, _) in finalists.items()
    }
    selection = {
        "submitted_model": submitted_name,
        "submitted_params": submitted_row["params"],
        "chosen_by": "design (config.SUBMITTED_MODEL), not by score",
        "classical_comparator": classical_row["family"],
        "submitted_minus_classical": comparison,
        "logistic_params": best_logistic["params"],
        "quantum_kernel_params": best_kernel["params"],
        "rbf_kernel_params": best_rbf["params"],
        "cv": {"splits": config.CV_SPLITS, "repeats": config.CV_REPEATS, "seed": config.SEED},
        "training_rows": int(len(y)),
        "training_positives": int(y.sum()),
    }
    return Stage1Result(
        predictions=predictions,
        all_model_predictions=all_models,
        cv_table=cv_table,
        selection=selection,
        reliability=reliability,
        model=fitted[submitted_name],
        estimator=estimator,
        cv_folds={name: by_row[id(row)]["folds"] for name, (row, _) in finalists.items()},
    )


# --------------------------------------------------------------------------- kernel circuit check
def kernel_circuit_report(quantum_model) -> dict:
    """Re-estimate a few kernel entries by running the overlap circuit, and report its cost.

    The classifier uses exact (statevector) kernel values. Here randomly chosen pairs of training
    customers are pushed through the real circuit U(x_b)^dagger U(x_a) on Aer: the fraction of
    all-zero outcomes estimates the same entry with shot noise, first without and then with the
    device noise model.
    """
    angles = quantum_model.train_angles_
    reps = quantum_model.reps
    n = angles.shape[1]
    rng = np.random.default_rng(config.SEED)
    pairs = [tuple(int(v) for v in rng.choice(len(angles), size=2, replace=False))
             for _ in range(config.KERNEL_CHECK_PAIRS)]
    states = feature_map_states(angles, reps)
    ideal = hardware.ideal_simulator()
    zeros = "0" * n
    rows = []
    stats = None
    for index, (a, b) in enumerate(pairs):
        circuit = kernel_circuit(angles[a], angles[b], reps)
        exact = float(fidelity_kernel(states[[a]], states[[b]])[0, 0])
        device_circuit, pair_stats = hardware.transpile_best(circuit, seeds=config.TRANSPILE_SEEDS[:3])
        stats = pair_stats if stats is None else stats
        ideal_counts = hardware.run_counts(ideal, circuit.decompose(reps=3), config.KERNEL_CHECK_SHOTS, seed=config.SIM_SEED + index)
        noisy_counts = hardware.run_noisy(device_circuit, config.KERNEL_CHECK_SHOTS, seed=config.SIM_SEED + index)
        rows.append({
            "train_row_a": a, "train_row_b": b, "exact": exact,
            "ideal_shots": ideal_counts.get(zeros, 0) / config.KERNEL_CHECK_SHOTS,
            "noisy_shots": noisy_counts.get(zeros, 0) / config.KERNEL_CHECK_SHOTS,
        })
    table = pd.DataFrame(rows)
    return {
        "pairs": table,
        "shots": config.KERNEL_CHECK_SHOTS,
        "max_abs_error_ideal": float((table["ideal_shots"] - table["exact"]).abs().max()),
        "mean_abs_error_ideal": float((table["ideal_shots"] - table["exact"]).abs().mean()),
        "mean_abs_error_noisy": float((table["noisy_shots"] - table["exact"]).abs().mean()),
        "mean_signed_error_noisy": float((table["noisy_shots"] - table["exact"]).mean()),
        "transpile": stats,
        "n_qubits": int(n),
        "reps": int(reps),
        "kernel_offdiagonal_mean": float(_offdiagonal(fidelity_kernel(states, states)).mean()),
        "kernel_offdiagonal_std": float(_offdiagonal(fidelity_kernel(states, states)).std()),
    }


def _offdiagonal(K: np.ndarray) -> np.ndarray:
    return K[~np.eye(len(K), dtype=bool)]
