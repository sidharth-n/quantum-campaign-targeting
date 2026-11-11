"""Ranking rule and the stage 1 result file."""
import numpy as np

from campaign import config
from campaign.stage1 import best_of, rank_customers


def test_ties_go_to_the_lower_customer_id():
    ranked = rank_customers(["C0009", "C0002", "C0005", "C0001"], [0.3, 0.7, 0.3, 0.3], n_top=2)
    assert list(ranked["customer_id"]) == ["C0002", "C0001", "C0005", "C0009"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]
    assert list(ranked["is_top_10"]) == [1, 1, 0, 0]


def test_a_tie_at_the_top_10_boundary_is_resolved_by_customer_id():
    ids = [f"C{i:04d}" for i in range(20, 0, -1)]
    probabilities = [0.5] * 20
    ranked = rank_customers(ids, probabilities, n_top=10)
    assert list(ranked.loc[ranked["is_top_10"] == 1, "customer_id"]) == [f"C{i:04d}" for i in range(1, 11)]


def test_stage1_file_schema(data, stage1_file):
    f = stage1_file
    assert list(f.columns) == ["customer_id", "predicted_probability", "rank", "is_top_10"]
    assert len(f) == 100
    assert set(f["customer_id"]) == set(data.test["customer_id"]) and f["customer_id"].is_unique
    assert ((f["predicted_probability"] > 0) & (f["predicted_probability"] < 1)).all()
    assert f["predicted_probability"].nunique() > 50, "probabilities must be continuous, not labels"
    assert sorted(f["rank"]) == list(range(1, 101))
    assert f["is_top_10"].sum() == data.constraints.n_candidates
    assert set(f.loc[f["is_top_10"] == 1, "rank"]) == set(range(1, 11))


def test_stage1_file_is_ranked_high_to_low_with_the_tie_rule(stage1_file):
    f = stage1_file.sort_values("rank")
    again = rank_customers(f["customer_id"], f["predicted_probability"], 10)
    assert list(again["customer_id"]) == list(f["customer_id"])
    assert (np.diff(f["predicted_probability"]) <= 0).all()


def test_stage1_file_matches_a_fresh_run(s1, stage1_file):
    assert s1.predictions.to_dict("list") == stage1_file.to_dict("list")


def test_submitted_model_is_the_declared_one(s1):
    assert s1.selection["submitted_model"] == config.SUBMITTED_MODEL == "hybrid"
    assert s1.cv_table["submitted"].sum() == 1


def test_selection_ordering_prefers_pr_auc_then_brier():
    rows = [{"pr_auc": 0.30, "brier": 0.11}, {"pr_auc": 0.298, "brier": 0.10}, {"pr_auc": 0.20, "brier": 0.05}]
    assert best_of(rows) is rows[1]          # within the tie band, lower Brier wins
    assert best_of([rows[0], rows[2]]) is rows[0]
