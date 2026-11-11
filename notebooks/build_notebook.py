"""Create and execute notebooks/walkthrough.ipynb.

    python notebooks/build_notebook.py

The notebook runs the same code as run_pipeline.py (into a scratch folder, so the committed
result files are not touched) and then walks through each stage with tables, circuits and figures.
"""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CELLS = [
    ("md", """# Quantum campaign targeting: walkthrough

This notebook runs the whole pipeline once and then looks at each stage: the Stage 1 classifiers, the Stage 2 QUBO and QAOA, the Stage 3 circuit, and the error propagation. It uses the same code as `python run_pipeline.py`. The report with the full discussion is `README.md`."""),
    ("code", """import shutil, sys, warnings
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "run_pipeline.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 80)
import run_pipeline
from campaign import config, qaoa, qubo, stage3, hardware
from campaign.models import feature_map_circuit"""),
    ("md", """## Run the pipeline

One call. Results go to a scratch folder (`notebook_run/`, deleted at the end) so that the committed files stay as they are; the last section compares the two."""),
    ("code", """out = Path("notebook_run")
run = run_pipeline.main(out)
s1, kernel, s2, s3, prop = run["stage1"], run["kernel"], run["stage2"], run["stage3"], run["propagation"]"""),
    ("md", """## Stage 1: subscription prediction

Cross-validation on the 300 training customers only (5 folds, 3 repeats, the same folds for every model). The submitted model is the hybrid: the 50/50 average of logistic regression and the quantum kernel classifier."""),
    ("code", """columns = ["model", "params", "roc_auc", "pr_auc", "precision_at_10", "recall_at_10", "brier", "log_loss", "ece", "calibration_slope", "submitted"]
s1.cv_table[s1.cv_table["finalist"]][columns].round(4)"""),
    ("code", """display(Image(filename=str(out / "results" / "figures" / "stage1_cross_validation.png")))"""),
    ("md", """The quantum part of Stage 1 is the feature map below (6 qubits). A kernel entry is the squared overlap of two such states. The next table re-estimates a few entries by running the overlap circuit with shots."""),
    ("code", """quantum = s1.model.quantum_
print(feature_map_circuit(quantum.n_qubits, quantum.reps).draw(fold=140))"""),
    ("code", """kernel["pairs"].round(4)"""),
    ("md", """Test predictions, ranked high to low (ties to the lower `customer_id`). The first ten go to Stage 2."""),
    ("code", """s1.predictions.head(12)"""),
    ("md", """## Stage 2: constrained selection

Candidate i sits on qubit i. `expected_net_value = predicted_probability * expected_deposit_value - contact_cost`."""),
    ("code", """s2["candidates"].round(4)"""),
    ("code", """exact = s2["exact"]
print("brute force optimum :", exact["ids"])
print("objective           :", round(exact["value"], 6), "| cost", exact["cost"], "| capacity", exact["capacity"])
print("groups of six       :", exact["n_subsets_of_size_k"], "| feasible:", exact["n_feasible"])
print("largest cost / capacity any group of six can need:", exact["max_cost_any_k_subset"], "/", exact["max_capacity_any_k_subset"],
      "(limits", s2["problem"].budget, "/", s2["problem"].capacity_limit, ")")"""),
    ("md", """**QUBO and Ising.** The unbalanced-penalty QUBO needs 10 qubits. Its ground state over all 1,024 bitstrings is checked against brute force, and the Ising energies are checked against the QUBO energies."""),
    ("code", """encoding = s2["encoding"]
tuned = {k: v for k, v in encoding["tuned"].items() if k != "table"}
print("penalty weights:", {k: round(v, 5) if isinstance(v, float) else v for k, v in tuned.items()})
print("ground state check:", encoding["unbalanced_check"])
print("max |Ising - QUBO| over all states:", encoding["ising_max_abs_difference"])
print("slack-variable encoding:", encoding["slack"].n, "qubits, ground state valid:", encoding["slack_check"]["valid"])"""),
    ("md", """**QAOA circuit** (one layer shown): Hadamards, the cost layer (RZ and RZZ), the mixer (RX)."""),
    ("code", """ham = qaoa.hamiltonian(encoding["unbalanced"])
print(qaoa.qaoa_circuit(ham, 1).draw(fold=150))"""),
    ("md", """**Results by depth**, for both QAOA variants, ideal and on the device noise model."""),
    ("code", """pd.read_csv(out / "results" / "stage2_qaoa_depth_study.csv").round(4)"""),
    ("code", """display(Image(filename=str(out / "results" / "figures" / "stage2_qaoa.png")))"""),
    ("md", """**Decoding.** Qiskit prints qubit 0 as the rightmost character. The most frequent measured bitstrings at p = 3:"""),
    ("code", """pd.DataFrame(s2["study"][config.QAOA_LAYERS]["ideal_sample"]["top_bitstrings"])[
    ["bitstring_qiskit", "count", "n_selected", "cost", "capacity", "feasible", "value", "is_exact_optimum"]].round(4)"""),
    ("code", """selection = s2["selection"]
print("selected bitstring (Qiskit key):", selection["bitstring_qiskit"])
print("decoded, qubit 0 first        :", qubo.key_to_bits(selection["bitstring_qiskit"]).tolist())
print("customers                     :", selection["ids"])
print("objective", round(selection["value"], 6), "| gap to brute force", selection["optimality_gap"],
      "| constraint-preserving QAOA agrees:", selection["constraint_preserving_qaoa_agrees"])"""),
    ("md", """## Stage 3: campaign outcome simulation

One qubit per selected customer, in Stage 1 rank order, each rotated so that P(1) is that customer's probability."""),
    ("code", """print(stage3.campaign_circuit(s3["probabilities"]).draw(fold=120))
pd.DataFrame({"qubit": range(6), "customer_id": s3["ids"], "probability": s3["probabilities"], "ry_angle": s3["angles"],
              "measured_p1_ideal": s3["headline"]["marginals"], "measured_p1_noisy": s3["noisy"]["marginals"]}).round(5)"""),
    ("code", """rows = [("exact (Poisson-binomial)", s3["exact"], 0.0),
        ("statevector", s3["statevector"]["estimate"], s3["statevector"]["error"]),
        (f"noiseless, {s3['headline']['shots']:,} shots (submitted)", s3["headline"]["estimate"], s3["headline"]["error"]),
        (f"noiseless, {s3['ideal_shots']['shots']:,} shots", s3["ideal_shots"]["estimate"], s3["ideal_shots"]["error"]),
        ("noisy, unmitigated", s3["noisy"]["estimate"], s3["noisy"]["error"]),
        ("noisy, readout mitigated", s3["mitigated"]["estimate"], s3["mitigated"]["error"]),
        ("counter circuit, noiseless", s3["counter"]["estimate"], s3["counter"]["error"]),
        ("Monte Carlo", s3["monte_carlo"]["estimate"], s3["monte_carlo"]["error"])]
pd.DataFrame(rows, columns=["estimate of P(at least 3)", "value", "error against exact"]).round(6)"""),
    ("code", """display(Image(filename=str(out / "results" / "figures" / "stage3_simulation.png")))"""),
    ("md", """## Error propagation

The Stage 1 model is refitted on bootstrap resamples of the training rows, and each set of probabilities is pushed through Stages 2 and 3."""),
    ("code", """print("top-10 overlap with the submitted candidates (mean):", prop["top10"]["mean_overlap_with_submitted_top10"])
print("same six still optimal (candidates fixed):", prop["stage2"]["fixed_candidates_same_selection_frequency"])
print("P(at least 3), same six customers:", {k: round(v, 4) for k, v in prop["stage3"]["fixed_six"].items()})
print("submitted value:", round(prop["stage3"]["submitted_exact"], 4), "| delta-method sd:", round(prop["stage3"]["delta_method_sd"], 4))
display(Image(filename=str(out / "results" / "figures" / "error_propagation.png")))"""),
    ("md", """## Check against the committed result files

The run above must reproduce the files at the repository root exactly."""),
    ("code", """names = ["stage1_predictions.csv", "stage2_selection.csv", "stage2_selection.json", "stage3_simulation.csv", "stage3_simulation.json"]
for name in names:
    same = (out / name).read_bytes() == (ROOT / name).read_bytes()
    print(f"{name:28s} identical to the committed file: {same}")
    assert same
shutil.rmtree(out)"""),
]


def build() -> Path:
    notebook = new_notebook()
    notebook.cells = [new_markdown_cell(text) if kind == "md" else new_code_cell(text) for kind, text in CELLS]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    path = HERE / "walkthrough.ipynb"
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    print("written:", build())
