"""Stage 1 metrics: the ones the organisers list (ranking and probability quality)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-12
METRICS = ["roc_auc", "pr_auc", "precision_at_10", "recall_at_10", "brier", "log_loss"]


def at_k(y, p, k=10):
    """Precision and recall among the k highest probabilities (ties keep the input order)."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")[:k]
    hits = int(y[order].sum())
    return hits / k, hits / max(int(y.sum()), 1)


def score(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    precision, recall = at_k(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision_at_10": float(precision),
        "recall_at_10": float(recall),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def reliability_table(y, p, bins=5):
    """Equal-count bins of predicted probability: mean prediction against observed rate."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({
            "n": int(len(chunk)),
            "p_min": float(p[chunk].min()),
            "p_max": float(p[chunk].max()),
            "mean_predicted": float(p[chunk].mean()),
            "observed_rate": float(y[chunk].mean()),
        })
    return rows


def expected_calibration_error(y, p, bins=5):
    table = reliability_table(y, p, bins)
    total = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in table) / total)


def calibration_line(y, p):
    """Slope and intercept of a logistic fit of the labels on logit(p). Ideal: slope 1, intercept 0."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(logit, np.asarray(y))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])
