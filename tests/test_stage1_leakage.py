"""Nothing fitted in stage 1 may depend on the test rows."""
import numpy as np
import pandas as pd

from campaign.data import FEATURE_COLUMNS
from campaign.features import NUMERIC, engineer


def test_test_sheet_has_no_labels_and_no_shared_customers(data):
    assert "y" not in data.test.columns
    assert not set(data.train["customer_id"]) & set(data.test["customer_id"])
    assert len(data.train) == 300 and len(data.test) == 100
    assert set(data.train["y"].unique()) == {0, 1}


def test_constraints_come_from_the_workbook(data):
    book = pd.read_excel(data_file(), sheet_name="Stage2_Constraints")
    rules = dict(zip(book["symbol"], book["value"]))
    c = data.constraints
    assert (c.n_candidates, c.n_select, c.budget, c.capacity) == (int(rules["N"]), int(rules["K"]), float(rules["C"]), float(rules["H"]))


def data_file():
    from campaign.config import DATA_FILE
    return DATA_FILE


def test_feature_engineering_is_row_wise(data):
    """A row's engineered features must not change with the rows around it."""
    whole = engineer(data.test)
    for i in (0, 17, 99):
        alone = engineer(data.test.iloc[[i]])
        pd.testing.assert_frame_equal(alone, whole.iloc[[i]])


def test_scaler_statistics_are_training_statistics(data, s1):
    """The fitted scaler holds the mean and spread of the 300 training rows, nothing else."""
    scaler = s1.model.classical_.pipe_.named_steps["columntransformer"].named_transformers_["num"]
    train = engineer(data.train)[NUMERIC].to_numpy(float)
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaler.scale_, train.std(axis=0), rtol=0, atol=1e-12)
    both = engineer(pd.concat([data.train, data.test]))[NUMERIC].to_numpy(float)
    assert not np.allclose(scaler.mean_, both.mean(axis=0))


def test_one_hot_levels_are_training_levels(data, s1):
    encoder = s1.model.classical_.pipe_.named_steps["columntransformer"].named_transformers_["cat"]
    for column, levels in zip(["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
                              encoder.categories_):
        assert set(levels) == set(data.train[column])


def test_a_prediction_depends_only_on_its_own_row(data, s1):
    """Predicting a customer alone, in a shuffled batch, or next to corrupted customers gives the
    same probability. Any transformer fitted on the prediction batch would break this."""
    X = data.test[FEATURE_COLUMNS]
    together = s1.model.predict_proba(X)[:, 1]

    alone = np.array([s1.model.predict_proba(X.iloc[[i]])[0, 1] for i in range(0, 100, 7)])
    np.testing.assert_allclose(alone, together[::7], rtol=0, atol=1e-12)

    order = np.random.default_rng(0).permutation(len(X))
    shuffled = s1.model.predict_proba(X.iloc[order])[:, 1]
    np.testing.assert_allclose(shuffled, together[order], rtol=0, atol=1e-12)

    corrupted = X.copy()
    corrupted.loc[corrupted.index[1:], ["age", "balance", "campaign"]] *= 50
    np.testing.assert_allclose(s1.model.predict_proba(corrupted)[0, 1], together[0], rtol=0, atol=1e-12)


def test_refitting_on_training_rows_reproduces_the_submitted_probabilities(data, s1, stage1_file):
    """The file at the repository root is exactly: fit on the 300 training rows, predict the test rows."""
    from sklearn.base import clone
    from campaign import config

    model = clone(s1.estimator).fit(data.train[FEATURE_COLUMNS], data.train["y"].to_numpy())
    p = np.round(model.predict_proba(data.test[FEATURE_COLUMNS])[:, 1], config.PROBABILITY_DECIMALS)
    fresh = dict(zip(data.test["customer_id"], p))
    on_file = dict(zip(stage1_file["customer_id"], stage1_file["predicted_probability"]))
    assert fresh == on_file


def test_cross_validation_uses_only_training_rows_and_every_row_once_per_repeat(data):
    from campaign import config, stage1

    y = data.train["y"].to_numpy()
    splits = stage1.cv_splits(y)
    assert len(splits) == config.CV_SPLITS * config.CV_REPEATS
    for repeat in range(config.CV_REPEATS):
        held = np.concatenate([splits[repeat * config.CV_SPLITS + k][1] for k in range(config.CV_SPLITS)])
        assert sorted(held) == list(range(len(y)))
    for fit_rows, held_rows in splits:
        assert not set(fit_rows) & set(held_rows)
        assert max(fit_rows.max(), held_rows.max()) < len(y)


def test_platt_calibration_scores_never_see_their_own_row_or_label(data):
    """Inside the kernel model, the score used to calibrate a row must come from a chain (scaler,
    PLS, kernel, SVM) fitted without that row. Flipping the labels of one calibration fold, or
    corrupting its features, must leave the scores of the other folds' fits untouched, and the
    fold's own scores must not react to its own labels."""
    from campaign.models import quantum_kernel_svm

    model = quantum_kernel_svm(reducer="pls", reps=1, bandwidth=0.25, C=1.0)
    X = data.train[FEATURE_COLUMNS]
    y = data.train["y"].to_numpy()
    folds = model.calibration_folds(X, y)
    held = folds[0][1]
    assert not set(folds[0][0]) & set(held)
    assert sorted(np.concatenate([f[1] for f in folds])) == list(range(len(y)))
    base = model.out_of_fold_scores(X, y, folds)

    flipped = y.copy()
    flipped[held] = 1 - flipped[held]
    after = model.out_of_fold_scores(X, flipped, folds)
    # the held-out rows' own labels were never used to produce their scores
    np.testing.assert_allclose(after[held], base[held], rtol=0, atol=1e-12)
    # and the same labels do change the scores of rows in the other folds, so the check has teeth
    others = np.setdiff1d(np.arange(len(y)), held)
    assert np.abs(after[others] - base[others]).max() > 1e-6


def test_identical_rows_share_a_calibration_fold(data):
    """Bootstrap resamples repeat customers; a repeated customer must not straddle a calibration split."""
    from campaign.models import quantum_kernel_svm

    rows = np.random.default_rng(0).choice(len(data.train), len(data.train))
    X = data.train[FEATURE_COLUMNS].iloc[rows].reset_index(drop=True)
    y = data.train["y"].to_numpy()[rows]
    for fit_rows, held_rows in quantum_kernel_svm().calibration_folds(X, y):
        assert not set(rows[fit_rows]) & set(rows[held_rows])
