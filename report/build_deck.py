"""Build presentation.pdf from the result files.

    python report/build_deck.py

Twelve 16:9 slides. Every number is read from the same result files as the README.
"""
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_report  # noqa: E402

INK = "#212529"
MUTED = "#5c636a"
QUANTUM = "#3b5bdb"
CLASSICAL = "#868e96"
NOISY = "#e8590c"
PANEL = "#f1f3f5"
W, H = 13.333, 7.5
REPO = "github.com/sidharth-n/quantum-campaign-targeting"


class Deck:
    def __init__(self, path):
        self.pdf = PdfPages(path, metadata={
            "Title": "Quantum campaign targeting: who to call, and how likely the campaign is to work",
            "Author": "Sidharth",
            "Subject": "The Atomiq Hackathon, Qiskit Fall Fest 2026 at IEM",
        })
        self.page = 0

    def slide(self, title, kicker=None):
        self.page += 1
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 0.18, H, color=QUANTUM, lw=0))
        if kicker:
            ax.text(0.7, H - 0.62, kicker.upper(), fontsize=11, color=QUANTUM, weight="bold", va="center")
        ax.text(0.7, H - 1.15, title, fontsize=25, color=INK, weight="bold", va="center")
        ax.text(0.7, 0.32, REPO, fontsize=9, color=MUTED, va="center")
        ax.text(W - 0.5, 0.32, str(self.page), fontsize=9, color=MUTED, va="center", ha="right")
        return fig, ax

    def save(self, fig):
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self):
        self.pdf.close()


def bullets(ax, items, x, y, width=60, size=14, gap=0.2, colour=INK):
    """Wrapped bullet points; returns the y position after the last one."""
    line = size * 1.42 / 72
    for item in items:
        lines = textwrap.wrap(item, width)
        ax.text(x, y, "•", fontsize=size, color=QUANTUM, va="top", weight="bold")
        ax.text(x + 0.27, y, "\n".join(lines), fontsize=size, color=colour, va="top", linespacing=1.32)
        y -= line * len(lines) + gap
    return y


def panel(ax, x, y, w, h, colour=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", color=colour, lw=0))


def image(fig, path, y_bottom, y_top, x_left=0.7, x_right=W - 0.7):
    """Place a picture inside the given band (inches), keeping its aspect ratio, centred."""
    picture = mpimg.imread(path)
    aspect = picture.shape[1] / picture.shape[0]
    height = y_top - y_bottom
    width = min(x_right - x_left, height * aspect)
    height = width / aspect
    left = x_left + (x_right - x_left - width) / 2
    ax = fig.add_axes([left / W, (y_top - height) / H, width / W, height / H])
    ax.imshow(picture)
    ax.axis("off")


def figure_slide(deck, title, kicker, points, picture):
    """Bullets on top, one figure filling the space below them."""
    fig, ax = deck.slide(title, kicker)
    end = bullets(ax, points, 0.7, H - 1.75, width=128, size=12, gap=0.1)
    image(fig, picture, 0.6, end - 0.05)
    deck.save(fig)


def stat(ax, x, y, w, value, label, colour=QUANTUM, size=30):
    panel(ax, x, y, w, 1.5)
    ax.text(x + 0.25, y + 0.98, value, fontsize=size, color=colour, weight="bold", va="center")
    ax.text(x + 0.25, y + 0.36, "\n".join(textwrap.wrap(label, int(w * 9.5))), fontsize=10.5, color=MUTED, va="center", linespacing=1.25)


def build(path=ROOT / "presentation.pdf"):
    r = build_report.load(ROOT)
    d = build_report.derived(r)
    s2, s3, prop, stress, sel, kernel = r["s2"], r["s3"], r["prop"], r["stress"], r["sel"], r["kernel"]
    q, mixer, uniform = s2["qaoa"], s2["constraint_preserving_qaoa"], s2["qaoa"]["uniform_random_reference"]
    figures = ROOT / "results" / "figures"
    ext = build_report.extension_values(r)
    deck = Deck(path)

    # 1 ------------------------------------------------------------------ title
    fig, ax = deck.slide("Who to call, and how likely the campaign is to work", "Quantum campaign targeting")
    ax.text(0.7, H - 1.85, "A hybrid quantum-classical pipeline for a bank term-deposit campaign", fontsize=15, color=MUTED, va="center")
    ax.text(0.7, H - 2.3, "The Atomiq Hackathon, Qiskit Fall Fest 2026 at IEM   |   Sidharth", fontsize=12, color=MUTED, va="center")
    panel(ax, 0.7, 2.75, 11.9, 1.5)
    ax.text(0.95, 3.85, "SIX CUSTOMERS TO CONTACT", fontsize=10.5, color=MUTED, weight="bold", va="center")
    ax.text(0.95, 3.25, d["ids"], fontsize=24, color=INK, weight="bold", va="center")
    stat(ax, 0.7, 0.9, 3.8, f"{s3['quantum_estimate']:.4f}", f"P(at least 3 subscribe), quantum estimate, {s3['estimates']['noiseless_shots_submitted']['shots']:,} shots")
    stat(ax, 4.75, 0.9, 3.8, f"{s3['exact_probability']:.4f}", "exact value from the same probabilities", INK)
    stat(ax, 8.8, 0.9, 3.8, f"{s2['optimality_gap']:.1f}", f"gap to the brute-force optimum (objective {s2['objective_value']:.2f})", INK)
    deck.save(fig)

    # 2 ------------------------------------------------------------------ pipeline
    fig, ax = deck.slide("One pipeline, three linked stages", "Architecture")
    boxes = [
        ("Stage 1  Predict", "300 labelled customers\n100 test probabilities\ntop 10 move on",
         "6-qubit ZZ feature map,\nfidelity quantum kernel", "preprocessing, SVM, Platt scaling,\nlogistic regression, 50/50 average"),
        ("Stage 2  Select", "exactly 6 of 10\nmax expected net value\nbudget and capacity limits",
         "QAOA on a 10-qubit QUBO,\ntwo variants, sampled on Aer", "QUBO and Ising map, COBYLA,\nbrute-force reference, decoding"),
        ("Stage 3  Simulate", "P(at least 3 of 6 subscribe)\none qubit per customer",
         "six RY rotations, measured;\ncoherent counter circuit", "exact Poisson-binomial,\nMonte Carlo, readout inversion"),
    ]
    for i, (name, what, quantum, classical) in enumerate(boxes):
        x = 0.7 + i * 4.1
        panel(ax, x, 1.45, 3.75, 4.35)
        ax.text(x + 0.25, 5.4, name, fontsize=17, color=INK, weight="bold", va="center")
        ax.text(x + 0.25, 4.95, what, fontsize=11.5, color=MUTED, va="top", linespacing=1.35)
        ax.text(x + 0.25, 3.6, "QUANTUM", fontsize=9.5, color=QUANTUM, weight="bold", va="center")
        ax.text(x + 0.25, 3.35, quantum, fontsize=11.5, color=INK, va="top", linespacing=1.35)
        ax.text(x + 0.25, 2.45, "CLASSICAL", fontsize=9.5, color=CLASSICAL, weight="bold", va="center")
        ax.text(x + 0.25, 2.2, classical, fontsize=11.5, color=INK, va="top", linespacing=1.35)
        if i < 2:
            ax.annotate("", xy=(x + 4.08, 3.6), xytext=(x + 3.78, 3.6), arrowprops={"arrowstyle": "-|>", "color": QUANTUM, "lw": 2})
    ax.text(0.7, 0.95, "\n".join(textwrap.wrap(
        "The same customer ids and the same Stage 1 probabilities run through all three result files. "
        "A traceability check reads the files back and stops the run if anything differs.", 135)),
        fontsize=12, color=INK, va="center", linespacing=1.3)
    deck.save(fig)

    # 3 ------------------------------------------------------------------ stage 1
    cv = d["cv"]
    figure_slide(deck, "Stage 1: a hybrid classifier, validated on training data only", "Predict", [
        f"{sel['selection']['training_rows']} training customers, {sel['selection']['training_positives']} subscribers. "
        "5-fold cross-validation, 3 repeats, same folds for every model. The test rows are predicted once.",
        f"Quantum kernel classifier: 6 qubits, ZZ feature map, SVM, Platt scaling. Alone it reaches PR-AUC {cv['quantum_kernel']['pr_auc']:.3f}, "
        f"logistic regression {cv['logistic']['pr_auc']:.3f}, a classical RBF kernel on the same inputs {cv['rbf_kernel']['pr_auc']:.3f}.",
        f"Submitted: the fixed 50/50 hybrid, by design. PR-AUC {cv['hybrid']['pr_auc']:.3f}, Brier {cv['hybrid']['brier']:.4f}, "
        f"log loss {cv['hybrid']['log_loss']:.4f}, calibration slope {cv['hybrid']['calibration_slope']:.2f}.",
        "No quantum gain here, and I do not claim one. The differences between models are within the fold-to-fold noise.",
    ], figures / "stage1_cross_validation.png")

    # 4 ------------------------------------------------------------------ stage 2 formulation
    fig, ax = deck.slide("Stage 2: six of ten, written as a QUBO", "Select")
    panel(ax, 0.7, 3.2, 6.6, 2.55)
    formula = ("maximise    sum v_i x_i        v_i = p_i * deposit_i - cost_i\n"
               "subject to  sum x_i = 6\n"
               "            sum cost_i x_i     <= 25\n"
               "            sum capacity_i x_i <= 22\n\n"
               "E(x) = - sum v_i x_i + A (sum x_i - 6)^2\n"
               "       + [ -l1 g(x) + l2 g(x)^2 ]  for budget and capacity,\n"
               "       g(x) = limit - sum w_i x_i")
    ax.text(0.95, 5.55, formula, fontsize=11, color=INK, va="top", family="monospace", linespacing=1.45)
    bullets(ax, [
        "Unbalanced penalties: 10 qubits, no auxiliary qubits. Weights chosen by a stated rule and checked on all 1,024 bitstrings.",
        f"Slack-variable version built and verified too: {s2['qubo']['slack_variant']['qubits']} qubits, exact, "
        f"{s2['qubo']['slack_variant']['zz_couplings']} couplings against {s2['qubo']['zz_couplings']}.",
        "Ising map x = (1 - z) / 2, energies equal on every state.",
    ], 7.65, 5.6, width=50, size=12.5, gap=0.16)
    panel(ax, 0.7, 0.85, 11.9, 2.05, "#fff4e6")
    ax.text(0.95, 2.55, "THE LIMITS NEVER BIND FOR THESE TEN CANDIDATES", fontsize=10.5, color=NOISY, weight="bold", va="center")
    ax.text(0.95, 2.2, "\n".join(textwrap.wrap(
        f"The costliest group of six costs {s2['constraint_activity']['max_cost_of_any_selection']:.0f} (limit "
        f"{s2['constraint_activity']['budget']:.0f}) and the most capacity any group needs is "
        f"{s2['constraint_activity']['max_capacity_of_any_selection']:.0f} (limit {s2['constraint_activity']['capacity']:.0f}). "
        f"All {s2['constraint_activity']['selections_of_size_k']} groups are feasible. I kept the full constraint handling and added a stress "
        f"instance with tightened limits ({stress['feasible_selections']} feasible groups, a different optimum) to test it where it matters.", 120)),
        fontsize=12.5, color=INK, va="top", linespacing=1.35)
    deck.save(fig)

    # 5 ------------------------------------------------------------------ stage 2 results
    figure_slide(deck, "Stage 2: what QAOA adds, measured per shot", "Select", [
        f"Penalty QAOA, p = 3: {q['exact_output_distribution']['p_optimal']:.1%} of the probability on the optimum, "
        f"{d['amp_penalty']:.0f} times a random bitstring and {ext['ideal_over_valid']:.1f} times a random valid selection of six. Constraint-preserving QAOA (Dicke state + XY mixer): "
        f"{mixer['exact_output_distribution']['p_optimal']:.1%}, every bitstring has exactly six ones.",
        f"The best feasible measured bitstring is the brute-force optimum, gap {s2['optimality_gap']:.1f}. Both variants agree.",
        f"Honest reference: with {q['shots']} shots, random bitstrings contain the optimum {uniform['p_optimum_seen_in_shots']:.2%} of the time. "
        "Finding it is not the achievement at 10 qubits.",
        f"Device noise model, heavy-hex: {q['transpilation']['two_qubit_gates_transpiled']} two-qubit gates at p = 3 and the optimum "
        f"probability falls to {q['noisy_sample']['p_optimal']:.4f}.",
    ], figures / "stage2_qaoa.png")

    # 6 ------------------------------------------------------------------ stage 3
    figure_slide(deck, "Stage 3: six qubits, one question", "Simulate", [
        "Qubit i is the i-th selected customer in Stage 1 rank order. RY(2 asin(sqrt(p_i))) makes P(1) equal that customer's probability.",
        "Success: at least three qubits read 1. Bit order is checked through the per-customer marginals, not only the count.",
        f"Submitted quantum estimate {s3['quantum_estimate']:.6f} against exact {s3['exact_probability']:.6f} "
        f"(error {s3['errors']['quantum_minus_exact']:+.6f}, one standard error {s3['quantum_estimate_standard_error']:.6f}).",
        f"A coherent counter circuit (adder + comparator, {s3['estimates']['counter_circuit']['qubits']} qubits) gives the same answer on one flag qubit, "
        "the starting point for amplitude estimation. It does not survive the noise model.",
    ], figures / "stage3_simulation.png")

    # 7 ------------------------------------------------------------------ errors
    fig, ax = deck.slide("Where the error comes from", "Noise and error propagation")
    stat(ax, 0.7, 4.45, 3.8, f"{d['shot_se']:.4f}", "shot noise of the submitted Stage 3 estimate", QUANTUM, 26)
    stat(ax, 4.75, 4.45, 3.8, f"{d['noise_bias']:.4f}", "readout bias on the Stage 3 circuit (device noise model)", NOISY, 26)
    stat(ax, 8.8, 4.45, 3.8, f"{prop['stage3']['fixed_six']['sd']:.3f}", f"Stage 1 model uncertainty, {prop['rounds']} bootstrap refits", INK, 26)
    image(fig, figures / "error_propagation.png", 0.6, 3.55)
    ax.text(0.7, 4.0, "\n".join(textwrap.wrap(
        f"Stage 1 dominates by a factor of about {d['model_over_shot']:.0f}. On average {prop['top10']['mean_overlap_with_submitted_top10']:.1f} of the ten "
        f"candidates stay in the top 10 of a refit, and the same six customers get a lower probability in {d['submitted_percentile']:.0%} of the refits. The answer is very sensitive to the training sample.", 140)),
        fontsize=12, color=INK, va="center", linespacing=1.3)
    deck.save(fig)

    # 8 ------------------------------------------------------------------ rigor
    fig, ax = deck.slide("Every quantum result has a classical check next to it", "Classical vs quantum")
    panel(ax, 0.7, 0.85, 5.8, 4.95)
    ax.text(0.95, 5.45, "CHECKED", fontsize=10.5, color=QUANTUM, weight="bold", va="center")
    bullets(ax, [
        "Quantum kernel against logistic regression, boosting and an RBF kernel on the same inputs and folds",
        f"Kernel entries from the overlap circuit against exact values (mean difference {kernel['mean_abs_error_ideal']:.4f})",
        "QUBO ground state and QAOA selection against brute force",
        "QAOA sampling against uniform random bitstrings",
        "Stage 3 circuit against the exact formula, enumeration and Monte Carlo",
        "Training emulators against Qiskit state vectors of the real circuits",
    ], 0.95, 5.1, width=52, size=11.5, gap=0.13)
    panel(ax, 6.8, 0.85, 5.8, 4.95)
    ax.text(7.05, 5.45, "NOT CLAIMED", fontsize=10.5, color=NOISY, weight="bold", va="center")
    bullets(ax, [
        "No quantum advantage, in accuracy or speed, at any stage. Brute force needs 1,024 evaluations.",
        "QAOA finding the optimum is not evidence of quality at this size. The per-shot probability is.",
        "The Stage 3 circuit samples a distribution a laptop samples just as well. Its error falls like Monte Carlo's.",
        "No real hardware. The noise model is a stated assumption, not a device calibration.",
    ], 7.05, 5.1, width=52, size=11.5, gap=0.13)
    deck.save(fig)

    # 9 ------------------------------------------------------------------ beyond the brief
    corr, ae, hw = r["corr"], r["ae"], r["hw"]
    fig, ax = deck.slide("Three questions the brief does not ask", "Beyond the brief")
    stat(ax, 0.7, 4.2, 3.8, f"{ext['loss_first']:.0%} to {ext['loss_last']:.0%}",
         f"model's chance that the campaign loses money when the six customers are correlated (factor parameter 0 to {ext['rho_max']:.1f})", NOISY)
    stat(ax, 4.75, 4.2, 3.8, f"{ext['ae_ratio']:.1f}x",
         f"smaller error with amplitude estimation at {ext['ae_cost']:,} state preparations (ideal simulator only)")
    stat(ax, 8.8, 4.2, 3.8, f"{ext['t3']['p_optimal']:.2%}",
         f"of shots return the optimum at p = 3 on the IBM Torino snapshot (random valid selection {ext['random_valid']:.2%}, ideal {ext['ideal_p3']:.2%})")
    points = [
        f"Not independent? A one-factor correlation model on an entangled {corr['circuit']['logical_qubits']}-qubit circuit keeps every Stage 1 probability. "
        f"P(at least 3) barely moves ({ext['p_first']:.3f} to {ext['p_last']:.3f}) and the expected net value not at all, while the model's loss probability doubles (payoff = expected deposit value).",
        f"Fewer circuit runs? On an ideal simulator the error of amplitude estimation falls with slope {ext['ae_slope']:.2f} (direct sampling {ext['ae_direct_slope']:.2f}). "
        f"On my noise model it fails: error {ext['ae_noisy_rmse']:.2f} against {ext['ae_noisy_direct']:.3f}. One Grover step already needs {ext['ae_one_step_gates']} two-qubit gates.",
        f"Real machines? On IBM Torino and Brisbane calibration data the Stage 3 estimate is within its standard error after readout mitigation. "
        f"For QAOA, Torino is {ext['torino_over_stated']:.1f} times better than my stated noise model, and Brisbane matches it.",
    ]
    if hw is not None:
        points.append(
            f"Run on real hardware: IBM {hw['backend']} gives {hw['stage3']['mitigated']['estimate']:.3f} for Stage 3 after readout mitigation "
            f"(exact {hw['stage3']['exact']:.3f}), job {hw['job_id']}."
        )
    bullets(ax, points, 0.7, 3.75, width=118, size=13, gap=0.16)
    deck.save(fig)

    # 10 ----------------------------------------------------------------- beyond the brief, figure
    figure_slide(deck, "Same answer, a wider picture around it", "Beyond the brief", [
        "Every panel is checked against exact maths, rebuilt by python run_extensions.py, and none of it enters the three stage files.",
    ], figures / "beyond_the_brief.png")

    # 11 ----------------------------------------------------------------- business
    fig, ax = deck.slide("What the bank should do", "Business impact")
    bullets(ax, [
        f"Contact {d['ids']}. Expected net value {s2['objective_value']:.2f}, cost {s2['budget_used']:.0f} of {s2['constraints']['budget']:.0f}, "
        f"capacity {s2['capacity_used']:.0f} of {s2['constraints']['capacity']:.0f}.",
        f"Do not plan on three subscriptions. The model says {s3['exact_probability']:.0%}, with {d['expected_subscribers']:.1f} subscribers expected, "
        f"and refits of the model on resampled data usually give the same six less: {prop['stage3']['fixed_six']['p05']:.0%} to {prop['stage3']['fixed_six']['p95']:.0%}.",
        f"The limits are not the bottleneck, the exactly-six rule is: {d['budget_left']:.0f} units of budget and {d['capacity_left']:.0f} of capacity stay unused. "
        f"Without that rule the best campaign inside the same limits contacts {d['free_size']} of the ten for {d['free_value']:.2f}.",
        f"The cheapest improvement is more labelled data. The whole pipeline rests on {sel['selection']['training_positives']} past subscribers, "
        "and that uncertainty is far larger than anything the quantum steps add.",
        "At 10 candidates a laptop solves the selection exactly. What the quantum part gives today is a formulation verified at small size, not a speed-up.",
    ], 0.7, H - 1.95, width=98, size=15.5, gap=0.3)
    deck.save(fig)

    # 12 ----------------------------------------------------------------- reproduce
    fig, ax = deck.slide("Reproduce it", "Code and reproducibility")
    panel(ax, 0.7, 3.85, 11.9, 1.9)
    ax.text(0.95, 5.5, "pip install -r requirements.txt\n"
            "python run_pipeline.py     # every result file and figure, a few minutes on a laptop\n"
            "python run_extensions.py   # the beyond-the-brief studies, under two minutes\n"
            "python -m pytest           # the test suite, including a byte-for-byte rerun",
            fontsize=13, color=INK, va="top", family="monospace", linespacing=1.6)
    bullets(ax, [
        "Python 3.12, Qiskit 2.6.2, Qiskit Aer 0.18.2, scikit-learn 1.9.1. Versions pinned, every seed in one config file.",
        "Tests cover leakage, the tie-break rule, file schemas, id and probability traceability, QUBO against brute force, "
        "Ising against QUBO, emulators against Qiskit, bit order, the Stage 3 maths and readout mitigation.",
        "The README is filled in from the result files by a script, so the report cannot drift from the data.",
    ], 0.7, 3.4, width=104, size=13.5, gap=0.2)
    ax.text(0.7, 0.95, REPO, fontsize=16, color=QUANTUM, weight="bold", va="center")
    deck.save(fig)

    deck.close()
    return path


if __name__ == "__main__":
    print("written:", build())
