"""Figures for the report. One accent colour per meaning, used the same way in every chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

QUANTUM = "#3b5bdb"     # anything produced by a quantum circuit
CLASSICAL = "#868e96"   # classical references
NOISY = "#e8590c"       # device noise model
EXACT = "#212529"       # exact values
MODEL_COLOURS = {"logistic": CLASSICAL, "boosting": "#adb5bd", "rbf_kernel": "#ced4da",
                 "quantum_kernel": "#91a7ff", "hybrid": QUANTUM}
MODEL_LABELS = {"logistic": "Logistic regression", "boosting": "Gradient boosting", "rbf_kernel": "RBF kernel SVM",
                "quantum_kernel": "Quantum kernel SVM", "hybrid": "Hybrid (submitted)"}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "figure.constrained_layout.use": True,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def stage1(s1, path):
    table = s1.cv_table[s1.cv_table["finalist"]].set_index("model")
    order = ["logistic", "boosting", "rbf_kernel", "quantum_kernel", "hybrid"]
    labels = [MODEL_LABELS[m] for m in order]
    base_rate = s1.selection["training_positives"] / s1.selection["training_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), width_ratios=[1.25, 1, 1.1])
    for ax, metric, title in ((axes[0], "pr_auc", "PR-AUC (higher is better)"), (axes[1], "brier", "Brier score (lower is better)")):
        values = np.array([table.loc[m, metric] for m in order])
        errors = np.array([table.loc[m, f"{metric}_se"] for m in order])
        ax.barh(labels, values, xerr=errors, color=[MODEL_COLOURS[m] for m in order],
                error_kw={"ecolor": EXACT, "capsize": 3, "linewidth": 1})
        span = (values + errors).max()
        for i, v in enumerate(values):
            ax.text(v + errors[i] + span * 0.015, i, f"{v:.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(0.09 if metric == "brier" else 0, span * 1.12 if metric == "pr_auc" else span * 1.012)
        ax.grid(axis="y", visible=False)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].axvline(base_rate, color=EXACT, linestyle=":", linewidth=1)
    axes[0].set_xlabel(f"dotted line: subscriber share in training, {base_rate:.3f}")
    axes[1].set_xlabel("axis starts at 0.09")
    ax = axes[2]
    for model in ("logistic", "quantum_kernel", "hybrid"):
        rows = s1.reliability[model]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows], marker="o", markersize=4,
                color=MODEL_COLOURS[model], label=MODEL_LABELS[model], linewidth=1.6)
    top = max(r["mean_predicted"] for rows in s1.reliability.values() for r in rows) * 1.08
    ax.plot([0, top], [0, top], color=EXACT, linestyle=":", linewidth=1)
    ax.set_xlabel("mean predicted probability, 5 equal-count bins")
    ax.set_ylabel("observed subscription rate")
    ax.set_title("Reliability, out-of-fold")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Stage 1: cross-validation on the 300 training customers (5 folds x 3 repeats, mean and standard error)", fontsize=11)
    _save(fig, path)


def stage2(s2, path):
    uniform = s2["uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    series = (
        (s2["study"], "exact_distribution", "penalty QAOA, ideal", QUANTUM, "-", "o"),
        (s2["study"], "noisy_sample", "penalty QAOA, device noise", NOISY, "-", "o"),
        (s2["mixer_study"], "exact_distribution", "constraint-preserving QAOA, ideal", QUANTUM, "--", "s"),
        (s2["mixer_study"], "noisy_sample", "constraint-preserving QAOA, device noise", NOISY, "--", "s"),
    )
    for ax, key, title, reference in ((axes[0], "p_optimal", "Probability of measuring the optimum", uniform["p_optimal"]),
                                      (axes[1], "feasible_fraction", "Share of feasible bitstrings", uniform["feasible_fraction"])):
        for study, name, label, colour, style, marker in series:
            depths = sorted(study)
            ax.plot(depths, [study[d][name][key] for d in depths], marker=marker, markersize=4, color=colour,
                    linestyle=style, label=label, linewidth=1.6)
        ax.axhline(reference, color=EXACT, linestyle=":", linewidth=1, label="uniform random bitstrings")
        ax.set_xlabel("QAOA layers p")
        ax.set_xticks(sorted(s2["study"]))
        ax.set_title(title)
    axes[0].set_yscale("log")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=5, fontsize=8.5)
    ax = axes[2]
    for study, label, colour, offset in ((s2["study"], "penalty QAOA", CLASSICAL, -0.2),
                                         (s2["mixer_study"], "constraint-preserving QAOA", "#343a40", 0.2)):
        depths = sorted(study)
        gates = [study[d]["transpile"]["two_qubit_gates_transpiled"] for d in depths]
        ax.bar(np.array(depths) + offset, gates, 0.4, color=colour, label=label)
        for d, g in zip(depths, gates):
            ax.text(d + offset, g, str(g), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("QAOA layers p")
    ax.set_xticks(sorted(s2["study"]))
    ax.set_title("Two-qubit gates after transpilation")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Stage 2: two QAOA variants on the 10-qubit selection problem, ideal against the heavy-hex device noise model", fontsize=11)
    _save(fig, path)


def stage3(s3, path):
    from .stage3 import decode
    n = len(s3["ids"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ax = axes[0]
    k = np.arange(n + 1)

    def count_distribution(counts):
        outcomes, shots = decode(counts, n)
        return np.bincount(outcomes.sum(axis=1), weights=shots, minlength=n + 1) / shots.sum()

    width = 0.27
    ax.bar(k - width, s3["pmf"], width, color=EXACT, label="exact")
    ax.bar(k, count_distribution(s3["counts_headline"]), width, color=QUANTUM, label="circuit, ideal")
    ax.bar(k + width, count_distribution(s3["counts_noisy"]), width, color=NOISY, label="circuit, noisy")
    ax.axvline(s3["threshold"] - 0.5, color=EXACT, linestyle=":", linewidth=1)
    ax.text(s3["threshold"] - 0.4, max(s3["pmf"]) * 0.95, f"success: {s3['threshold']} or more", fontsize=8)
    ax.set_xlabel("number of the six customers who subscribe")
    ax.set_title("Distribution of subscribers")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    shots = np.array([e["shots"] for e in s3["shot_sweep"]], dtype=float)
    ax.loglog(shots, [e["abs_error"] for e in s3["shot_sweep"]], marker="o", markersize=4, color=QUANTUM, label="|estimate - exact|")
    exact = s3["exact"]
    ax.loglog(shots, np.sqrt(exact * (1 - exact) / shots), color=EXACT, linestyle=":", label="one standard error")
    ax.set_xlabel("shots")
    ax.set_title("Sampling error against shots")
    ax.legend(fontsize=8)

    ax = axes[2]
    names = ["Statevector", "Ideal shots (submitted)", "Noisy", "Noisy, readout mitigated", "Counter circuit, ideal", "Monte Carlo"]
    entries = [s3["statevector"], s3["headline"], s3["noisy"], s3["mitigated"], s3["counter"], s3["monte_carlo"]]
    colours = [QUANTUM, QUANTUM, NOISY, NOISY, QUANTUM, CLASSICAL]
    ax.barh(names, [e["error"] for e in entries], color=colours,
            xerr=[e.get("standard_error", 0.0) for e in entries], error_kw={"ecolor": EXACT, "capsize": 3, "linewidth": 1})
    ax.axvline(0, color=EXACT, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("estimate minus exact (bar: one standard error)")
    ax.set_title("Error of each estimate")
    ax.grid(axis="y", visible=False)
    fig.suptitle(f"Stage 3: P(at least {s3['threshold']} of {n} subscribe), exact value {exact:.4f}", fontsize=11)
    _save(fig, path)


def propagation(prop, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ax = axes[0]
    inclusion = prop["top10"]["inclusion_frequency"]
    ax.barh(list(inclusion), list(inclusion.values()), color=CLASSICAL)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of refits with the customer in the top 10")
    ax.set_title("Do the ten candidates stay candidates?")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ids = list(prop["stage3"]["probability_submitted"])
    submitted = np.array([prop["stage3"]["probability_submitted"][c] for c in ids])
    low = np.array([prop["stage3"]["probability_p05"][c] for c in ids])
    high = np.array([prop["stage3"]["probability_p95"][c] for c in ids])
    ax.errorbar(submitted, np.arange(len(ids)), xerr=[submitted - np.minimum(low, submitted), np.maximum(high, submitted) - submitted],
                fmt="o", color=QUANTUM, ecolor=CLASSICAL, capsize=3, markersize=5)
    ax.set_yticks(np.arange(len(ids)), ids)
    ax.invert_yaxis()
    ax.set_xlabel("dot: submitted, bar: 5th to 95th percentile")
    ax.set_title("Probabilities of the six selected customers")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    ax.hist(prop["samples"]["p_fixed_six"], bins=24, color=CLASSICAL)
    ax.axvline(prop["stage3"]["submitted_exact"], color=QUANTUM, linewidth=2)
    ax.text(prop["stage3"]["submitted_exact"], ax.get_ylim()[1] * 0.92, " submitted", color=QUANTUM, fontsize=9)
    ax.set_xlabel("value under each refit, same six customers")
    ax.set_title("P(at least 3 of the six subscribe)")
    fig.suptitle(f"Error propagation: {prop['rounds']} bootstrap refits of the stage 1 model", fontsize=11)
    _save(fig, path)
