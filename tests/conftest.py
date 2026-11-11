"""Shared fixtures. Tests that read the result files check the files committed at the repository root."""
import json
from pathlib import Path

import pandas as pd
import pytest

from campaign import qubo, stage1, stage2
from campaign.data import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data():
    return load()


@pytest.fixture(scope="session")
def s1(data):
    """A fresh stage 1 run (cross-validation, final fit, test predictions)."""
    return stage1.run(data)


@pytest.fixture(scope="session")
def stage1_file():
    return pd.read_csv(ROOT / "stage1_predictions.csv")


@pytest.fixture(scope="session")
def stage2_csv():
    return pd.read_csv(ROOT / "stage2_selection.csv")


@pytest.fixture(scope="session")
def stage2_json():
    return json.loads((ROOT / "stage2_selection.json").read_text())


@pytest.fixture(scope="session")
def stage3_csv():
    return pd.read_csv(ROOT / "stage3_simulation.csv")


@pytest.fixture(scope="session")
def stage3_json():
    return json.loads((ROOT / "stage3_simulation.json").read_text())


@pytest.fixture(scope="session")
def problem(data, stage1_file):
    """The stage 2 problem rebuilt from the stage 1 file and the organisers' workbook only."""
    candidates = stage2.candidates_table(stage1_file, data.ops, data.constraints.n_candidates)
    return qubo.make_problem(candidates, data.constraints.n_select, data.constraints.budget, data.constraints.capacity)
