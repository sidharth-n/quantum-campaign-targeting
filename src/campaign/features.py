"""Feature engineering and preprocessing.

`engineer` applies row-wise transforms only: no statistic of any other row enters a row, so it
cannot carry information between train and test. Everything that learns from data (scaling,
one-hot levels) lives in the ColumnTransformer and is fitted on training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC = ["age", "balance_slog", "day", "campaign_log", "pdays_log", "previous_log", "contacted_before"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    balance = out["balance"].astype(float)
    out["balance_slog"] = np.sign(balance) * np.log1p(np.abs(balance))   # signed log, balance can be negative
    out["campaign_log"] = np.log1p(out["campaign"].astype(float))
    out["previous_log"] = np.log1p(out["previous"].astype(float))
    contacted = out["pdays"] != -1                                       # pdays = -1 means never contacted before
    out["contacted_before"] = contacted.astype(int)
    out["pdays_log"] = np.log1p(out["pdays"].where(contacted, 0).astype(float))
    return out


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
