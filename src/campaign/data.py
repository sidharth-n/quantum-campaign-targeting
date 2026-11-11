"""Load the organisers' workbook and validate it.

The test sheet has no labels, so nothing in this project can look at test outcomes. The
constraint values (N, K, budget, capacity) are read from the Stage2_Constraints sheet.
"""
from dataclasses import dataclass

import pandas as pd

from .config import DATA_FILE

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "campaign", "pdays", "previous", "poutcome",
]
OPS_COLUMNS = ["contact_cost", "expected_deposit_value", "capacity_units", "customer_segment"]


@dataclass(frozen=True)
class Constraints:
    n_candidates: int   # N: size of the candidate pool taken from the stage 1 ranking
    n_select: int       # K: exact number of customers to contact
    budget: float       # C: sum of contact_cost must not exceed this
    capacity: float     # H: sum of capacity_units must not exceed this


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame          # 300 rows, features + y (0/1)
    test: pd.DataFrame           # 100 rows, features only
    ops: pd.DataFrame            # stage 2 inputs, indexed by customer_id
    constraints: Constraints


def load(path=DATA_FILE) -> Dataset:
    book = pd.read_excel(path, sheet_name=None)
    for sheet in ("Stage1_Train", "Stage1_Test", "Stage2_Data", "Stage2_Constraints"):
        if sheet not in book:
            raise ValueError(f"sheet {sheet!r} missing from {path}")

    train = book["Stage1_Train"].copy()
    test = book["Stage1_Test"].copy()
    ops = book["Stage2_Data"].copy()
    cons = book["Stage2_Constraints"].copy()

    for frame in (train, test, ops):
        frame["customer_id"] = frame["customer_id"].astype(str).str.strip()

    labels = train["y"].astype(str).str.strip().str.lower()
    if not labels.isin(["yes", "no"]).all():
        raise ValueError("unexpected label in Stage1_Train.y")
    train["y"] = (labels == "yes").astype(int)

    missing = [c for c in FEATURE_COLUMNS if c not in train.columns or c not in test.columns]
    if missing:
        raise ValueError(f"feature columns missing: {missing}")
    if "y" in test.columns:
        raise ValueError("Stage1_Test must not carry labels")
    if train[FEATURE_COLUMNS].isna().any().any() or test[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("missing values in the feature columns")
    for name, frame in (("train", train), ("test", test), ("ops", ops)):
        if not frame["customer_id"].is_unique:
            raise ValueError(f"duplicate customer_id in {name}")
    if set(train["customer_id"]) & set(test["customer_id"]):
        raise ValueError("train and test share customer ids")
    if not test["customer_id"].isin(ops["customer_id"]).all():
        raise ValueError("a test customer has no Stage2_Data row")

    rules = dict(zip(cons["symbol"].astype(str).str.strip(), cons["value"]))
    constraints = Constraints(
        n_candidates=int(rules["N"]),
        n_select=int(rules["K"]),
        budget=float(rules["C"]),
        capacity=float(rules["H"]),
    )
    return Dataset(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        ops=ops.set_index("customer_id")[OPS_COLUMNS],
        constraints=constraints,
    )
