"""Two full runs must give byte-identical result files."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOT_COMPARED = {"environment.json"}          # records the machine, so it legitimately differs elsewhere


@pytest.mark.slow
def test_rerun_reproduces_every_result_file_byte_for_byte(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--out", str(tmp_path), "--no-figures"],
        check=True, cwd=ROOT, capture_output=True,
    )
    produced = sorted(p for p in tmp_path.rglob("*") if p.is_file() and p.name not in NOT_COMPARED)
    assert len(produced) >= 5 + 15          # five stage files plus the detail files in results/
    different = [str(p.relative_to(tmp_path)) for p in produced
                 if p.read_bytes() != (ROOT / p.relative_to(tmp_path)).read_bytes()]
    assert not different, f"files differ from the committed ones: {different}"
    for name in ("stage1_predictions.csv", "stage2_selection.csv", "stage2_selection.json",
                 "stage3_simulation.csv", "stage3_simulation.json"):
        assert (tmp_path / name).exists()
