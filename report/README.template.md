# Quantum campaign targeting: who to call, and how likely the campaign is to work

My entry for The Atomiq Hackathon (Qiskit Fall Fest 2026 at IEM). A bank wants to run a term-deposit campaign. I predict who is likely to subscribe, use a quantum optimiser to pick six customers to contact, and use a quantum circuit to estimate the chance that at least three of them subscribe.

**Final answer**

| | |
|---|---|
| Six customers to contact | **{d[ids]}** |
| Probability that at least 3 of them subscribe | **{s3[quantum_estimate]:.4f}** (quantum estimate, {s3[estimates][noiseless_shots_submitted][shots]:,} shots). Exact value from the same probabilities: {s3[exact_probability]:.4f} |
| Expected net value of the selection | {s2[objective_value]:.3f}, equal to the brute-force optimum (gap {s2[optimality_gap]:.1f}) |
| Budget and capacity used | {s2[budget_used]:.0f} of {s2[constraints][budget]:.0f}, and {s2[capacity_used]:.0f} of {s2[constraints][capacity]:.0f} |

Everything is rebuilt by one command, `python run_pipeline.py`, in two to four minutes on a laptop. Two runs give byte-identical result files, and a test checks that.

**Beyond the brief.** The brief asks for one circuit and one number. I also asked three questions it does not ask and checked each answer against exact maths (section 11). None of them changes the submitted answer above.

| Question | What I found |
|---|---|
| What if the six customers are not independent? | I load a one-factor correlation model on an entangled {corr[circuit][logical_qubits]}-qubit circuit that keeps every customer's Stage 1 probability. At factor parameter rho = {x[rho_max]:.1f} (mean correlation between two customers' outcomes {x[pair_last]:.2f}) the chance of at least three subscriptions barely moves ({x[p_first]:.3f} to {x[p_last]:.3f}), while the model's chance that the campaign loses money goes from {x[loss_first]:.0%} to {x[loss_last]:.0%} (taking each customer's expected deposit value as the payoff when they subscribe). The expected net value that Stage 2 optimises does not move at all. |
| Can the probability be read with fewer circuit runs? | On an ideal simulator, yes. With amplitude estimation the error falls with slope {x[ae_slope]:.2f} against the number of state preparations (direct sampling, measured the same way: {x[ae_direct_slope]:.2f}), and at {x[ae_cost]:,} preparations it is {x[ae_ratio]:.1f} times smaller. On my noise model the same estimator fails (error {x[ae_noisy_rmse]:.2f} against {x[ae_noisy_direct]:.3f} for direct sampling), so the submitted number uses direct sampling. |
| What happens on the calibration of real IBM machines? | On snapshots of IBM's Torino and Brisbane processors the Stage 3 estimate is within its standard error of the exact value after readout mitigation. QAOA at p = 3 returns the optimal selection in {x[t3][p_optimal]:.2%} of shots on Torino and {x[b3][p_optimal]:.2%} on Brisbane. The ideal circuit gives {x[ideal_p3]:.2%}, a random bitstring {x[random_bitstring]:.2%} and a random valid selection {x[random_valid]:.2%}. |{t[hardware_summary]}

Four things I found that I think matter more than the headline numbers:

1. **The budget and capacity limits never bind for my ten candidates.** The most expensive possible group of six costs {s2[constraint_activity][max_cost_of_any_selection]:.0f} (limit {s2[constraint_activity][budget]:.0f}) and the most capacity-hungry needs {s2[constraint_activity][max_capacity_of_any_selection]:.0f} (limit {s2[constraint_activity][capacity]:.0f}), so all {s2[constraint_activity][selections_of_size_k]} groups of six are feasible. I still built and verified the full constraint handling, and I added a stress test with tightened limits to show it works when the limits do bind.
2. **QAOA finds the optimum, but so would random guessing at this size.** With {s2[qaoa][shots]} shots, uniformly random bitstrings contain the optimum with probability {s2[qaoa][uniform_random_reference][p_optimum_seen_in_shots]:.2%}. What QAOA adds is measured per shot: it raises the chance of the optimal bitstring from {s2[qaoa][uniform_random_reference][p_optimal]:.4f} for a random bitstring to {s2[qaoa][exact_output_distribution][p_optimal]:.4f} ({d[amp_penalty]:.0f} times). Against the fairer reference, a random valid selection of six ({x[random_valid]:.4f}), that is {x[ideal_over_valid]:.1f} times. I make no claim of quantum advantage anywhere in this project.
3. **On the device noise model I assume, the QAOA signal is almost gone.** The p = 3 circuit needs {s2[qaoa][transpilation][two_qubit_gates_transpiled]} two-qubit gates on heavy-hex hardware, and its output is close to uniform noise (optimum probability {s2[qaoa][noisy_sample][p_optimal]:.4f}, against {s2[qaoa][uniform_random_reference][p_optimal]:.4f} for random bitstrings). On the calibration snapshot of IBM's Torino processor the same circuit does better ({x[t3][p_optimal]:.4f}, section 11.3).
4. **Stage 1 uncertainty dominates everything else.** The shot noise of the Stage 3 estimate is about {d[shot_se]:.4f}. The spread caused by refitting the Stage 1 model on resampled training data is about {d[model_sd]:.2f}, roughly {d[model_over_shot]:.0f} times larger. The same six customers get a lower probability in {d[submitted_percentile]:.0%} of those refits, so the submitted number is very sensitive to the training sample.

## Contents

1. [Pipeline architecture](#1-pipeline-architecture)
2. [Stage 1: subscription prediction](#2-stage-1-subscription-prediction)
3. [Mathematical formulation of Stage 2](#3-mathematical-formulation-of-stage-2)
4. [Stage 2 results](#4-stage-2-results)
5. [Stage 3: campaign outcome simulation](#5-stage-3-campaign-outcome-simulation)
6. [Circuit design and transpilation](#6-circuit-design-and-transpilation)
7. [Classical vs quantum rigor](#7-classical-vs-quantum-rigor)
8. [Error and noise analysis](#8-error-and-noise-analysis)
9. [Setup and execution](#9-setup-and-execution)
10. [Business impact](#10-business-impact)
11. [Beyond the brief](#11-beyond-the-brief)
12. [Limitations](#12-limitations)
13. [References](#13-references)

## 1. Pipeline architecture

```
organisers' workbook (data/bank_campaign_participant_release.xlsx)
        |
        v
STAGE 1  300 labelled customers -> cross-validation -> final fit -> 100 test probabilities
         quantum:   6-qubit ZZ feature map, fidelity kernel            (Qiskit)
         classical: preprocessing, SVM on the kernel, Platt scaling, logistic regression
         output:    stage1_predictions.csv  (rank, top-10 flag)
        |
        |  top 10 customers, same probabilities, same ids
        v
STAGE 2  choose exactly 6 of 10: maximise sum(p * deposit_value - cost), budget and capacity limits
         classical: brute force over all 1,024 bitstrings (the reference)
         quantum:   QUBO -> Ising -> QAOA, 10 qubits, sampled on Aer   (Qiskit, Qiskit Aer)
         output:    stage2_selection.csv / .json
        |
        |  6 selected customers, same probabilities, same ids
        v
STAGE 3  P(at least 3 of 6 subscribe)
         quantum:   one RY qubit per customer, measured                (Qiskit, Qiskit Aer)
         classical: exact Poisson-binomial value, Monte Carlo
         output:    stage3_simulation.csv / .json
        |
        v
ERROR PROPAGATION  200 bootstrap refits of Stage 1, pushed through Stages 2 and 3
```

The stages are linked through `customer_id` and through one set of probabilities. Stage 1 rounds its probabilities once to 12 decimals, and those exact values are what the CSV holds and what Stages 2 and 3 compute with. A test (`tests/test_result_files.py`) reads the three files back and checks that the ids and probabilities are identical in all of them, and that the Stage 2 objective and the Stage 3 exact value can be recomputed from the Stage 1 file and the workbook alone.

What is quantum and what is classical:

| Step | Quantum | Classical |
|---|---|---|
| Stage 1 features | | row-wise feature engineering, scaling, one-hot encoding, PLS or PCA to 6 numbers |
| Stage 1 kernel | 6-qubit ZZ feature map, kernel entry = squared overlap of two feature-map states | |
| Stage 1 classifier | | SVM on the kernel matrix, Platt scaling, logistic regression, the 50/50 average |
| Stage 2 formulation | | QUBO, penalty weights, Ising map, brute-force reference |
| Stage 2 optimiser | QAOA circuits (two variants), sampled bitstrings | COBYLA outer loop, decoding and feasibility check |
| Stage 3 | six RY rotations, measurement, counter circuit | exact formula, Monte Carlo, readout-error inversion |

Repository layout:

```
run_pipeline.py            one command that rebuilds every result
run_extensions.py          the studies of section 11 (reads the stage files, never writes them)
build_report.py            fills this README from the result files
src/campaign/
  config.py                every seed, grid, shot count and noise number
  data.py                  loads and validates the workbook
  features.py  models.py   feature engineering, classifiers, quantum kernel
  stage1.py                cross-validation, final fit, ranking
  qubo.py                  brute force, two QUBO encodings, Ising map, penalty rule
  qaoa.py                  QAOA circuits, exact emulators for training, Dicke state, CVaR
  stage2.py                the Stage 2 study and the selection
  stage3.py                Stage 3 circuits, exact values, readout mitigation
  hardware.py              heavy-hex coupling map, noise model, transpilation statistics
  propagation.py           bootstrap error propagation
  correlated.py            correlated customers: copula model, entangled circuit, money at risk
  amplitude.py             amplitude estimation of the Stage 3 probability
  ibm_backends.py          Stage 2 and 3 circuits on IBM device snapshots
  real_device.py           circuits and post-processing for a real IBM processor
  outputs.py  figures.py   result files and figures
scripts/run_ibm_hardware.py   optional run on a real IBM processor (needs an API key)
tests/                     pytest suite
notebooks/walkthrough.ipynb   executed notebook that walks through the three stages
stage1_predictions.csv  stage2_selection.csv/.json  stage3_simulation.csv/.json   required result files
results/                   cross-validation table, QAOA runs and counts, transpilation, seeds, environment, figures
presentation.pdf           the deck
```

## 2. Stage 1: subscription prediction

**Data.** {sel[selection][training_rows]} labelled customers, {sel[selection][training_positives]} of them subscribers ({d[base_rate]:.1%}), and 100 unlabelled test customers. Fifteen features (age, job, marital status, education, default, balance, housing loan, personal loan, contact type, day, month, number of contacts, days since last contact, previous contacts, previous outcome).

**Leakage control.** Feature engineering is row-wise only (signed log of balance, logs of the count columns, a "contacted before" flag from `pdays = -1`). Everything that learns from data (scaler, one-hot levels, PCA or PLS, the SVM, Platt scaling) is fitted inside scikit-learn estimators on training rows only, and inside cross-validation on the training part of each fold only. The test sheet has no labels. The submitted model is chosen and fitted before it sees a test row, and predicts the test rows once. Later analyses (the comparison models, the bootstrap refits) also predict the test features, without labels and without any feedback into the choice of model or settings. Tests check that the fitted scaler holds exactly the training statistics, that a prediction does not change when the other test rows are shuffled or corrupted, and that refitting on the 300 training rows reproduces the submitted file.

**Models.**

- *Classical baseline:* L2 logistic regression without class weighting (calibration is measured in cross-validation). {d[n_logistic]} values of C.
- *Classical reference:* gradient boosting with fixed conservative settings.
- *Quantum:* a fidelity quantum kernel classifier. The features are reduced to 6 numbers, standardised, multiplied by a bandwidth and used as rotation angles of Qiskit's `zz_feature_map` on 6 qubits with linear entanglement. The kernel entry for two customers is the squared overlap of their two circuit states. The kernel matrix goes into an SVM, and Platt scaling on out-of-fold SVM scores turns the scores into probabilities (the whole chain, including the reduction, is refitted inside every calibration fold, so a score never comes from a model that saw that row or its label). Grid: reduction by PCA or PLS, 1 or 2 feature-map repetitions, three bandwidths, three values of C ({d[n_quantum]} settings).
- *Classical control:* the same pipeline with a Gaussian (RBF) kernel in place of the circuit, same inputs, same grid ({d[n_rbf]} settings). It isolates what the quantum feature map adds.
- *Hybrid (submitted):* the fixed 50/50 average of the tuned logistic regression and the tuned quantum kernel classifier. The weight is not tuned.

**How the submitted model is chosen.** Within each family the setting with the highest mean cross-validated PR-AUC wins (ties within 0.005 go to the lower Brier score). The submitted model is the hybrid, by design: the challenge asks for a quantum or hybrid classifier and treats the classical model as the baseline. The score did not make this choice. On the tuning metric (PR-AUC) logistic regression is slightly ahead, and the two models are level within the fold scatter, as the tables below show.

**Validation.** Repeated stratified 5-fold cross-validation, 3 repeats, seed 42, same folds for every model, training rows only. Mean over the 15 folds, plus or minus the standard deviation of the fold scores divided by the square root of 15:

{t[cv]}

P@10 and R@10 are computed inside each held-out fold of 60 customers. ECE and the calibration slope use the pooled out-of-fold predictions (slope 1 is ideal, below 1 means the model is overconfident). The plus-or-minus values are a rough guide to the fold-to-fold scatter, not confidence intervals: the repeated folds share customers, and the settings were tuned on the same folds, so the scores of the tuned models are also slightly optimistic.

Hybrid against the classical baseline on the same folds (mean paired difference):

{t[paired]}

**How I read this.**

- The dataset carries little signal: {sel[selection][training_positives]} subscribers in {sel[selection][training_rows]} rows. Every model lands between {d[roc_min]:.2f} and {d[roc_max]:.2f} ROC-AUC with a fold scatter of {d[roc_se_min]:.2f} to {d[roc_se_max]:.2f}, so most differences between models are inside the noise.
- The quantum kernel classifier alone ranks worse than logistic regression (PR-AUC {d[cv][quantum_kernel][pr_auc]:.3f} against {d[cv][logistic][pr_auc]:.3f}), and the classical RBF control on the same six inputs reaches {d[cv][rbf_kernel][pr_auc]:.3f}. The quantum feature map adds nothing measurable here, and I do not claim it does.
- The tuned quantum kernel setting is {d[qk_setting]}. The choice of reduction makes no measurable difference (best PR-AUC {d[pca_best_pr]:.3f} with PCA, {d[pls_best_pr]:.3f} with PLS, which is fitted on training labels inside each fold). A small bandwidth is the regime where this kernel behaves most like a smooth classical kernel.
- The hybrid and logistic regression are level: the paired table shows no difference larger than about two of its standard errors, in either direction. On PR-AUC alone logistic regression is ahead ({d[cv][logistic][pr_auc]:.3f} against {d[cv][hybrid][pr_auc]:.3f}); on log loss and on the calibration measures the hybrid is ahead (ECE {d[cv][hybrid][ece]:.3f} against {d[cv][logistic][ece]:.3f}, slope {d[cv][hybrid][calibration_slope]:.2f} against {d[cv][logistic][calibration_slope]:.2f}), because averaging two different models pulls the extreme predictions in.
- A calibration slope of {d[cv][hybrid][calibration_slope]:.2f} still means the highest probabilities are somewhat too high. That matters for Stage 3 and I come back to it in the error analysis.

![Stage 1 cross-validation](results/figures/stage1_cross_validation.png)

**Test predictions.** The hybrid is fitted once on all {sel[selection][training_rows]} training rows and predicts the 100 test customers once. Probabilities range from {d[p_min]:.3f} to {d[p_max]:.3f} with mean {d[p_mean]:.3f}. Customers are ranked high to low, ties go to the lower `customer_id`, and the first ten are the Stage 2 candidates. File: `stage1_predictions.csv`. The test probabilities of the other four models are in `results/stage1_test_predictions_all_models.csv` so the comparison with the classical baseline can be scored on the hidden labels too; nothing was selected from that file.

**Is the kernel really a quantum circuit result?** The classifier uses exact kernel values from a state-vector calculation. To check them against the circuit, I ran the overlap circuit (feature map for one customer, inverse feature map for another, measure all qubits, count all-zero outcomes) for {d[kernel_pairs]} random pairs of training customers with {kernel[shots]} shots each. Mean absolute difference from the exact entry: {kernel[mean_abs_error_ideal]:.4f} without noise, {kernel[mean_abs_error_noisy]:.4f} on the device noise model, where the entries come out {kernel[mean_signed_error_noisy]:+.3f} too low on average because noise leaks probability out of the all-zero outcome. Table: `results/stage1_kernel_circuit_check.csv`.

## 3. Mathematical formulation of Stage 2

**Problem.** Candidate i is the customer with Stage 1 rank i + 1, and qubit i carries its decision bit x_i (1 = contact).

```
maximise    sum_i v_i x_i              v_i = p_i * expected_deposit_value_i - contact_cost_i
subject to  sum_i x_i = K              K = 6
            sum_i c_i x_i <= C         C = 25   (contact_cost)
            sum_i h_i x_i <= H         H = 22   (capacity_units)
            x_i = 0 or 1,  i = 0..9
```

K, C and H are read from the `Stage2_Constraints` sheet. The probabilities p_i are the Stage 1 values, unrounded beyond the 12 decimals in the file.

**Exact reference.** `qubo.brute_force` evaluates all 1,024 bitstrings, keeps the feasible ones and returns the best. A test compares it with an independent `itertools.combinations` enumeration.

**QUBO, encoding 1: unbalanced penalisation (10 qubits, no auxiliary qubits).** This is the encoding the QAOA runs on.

```
E(x) = - sum_i v_i x_i
       + A (sum_i x_i - K)^2
       + [ -lambda_1 g_C(x) + lambda_2 g_C(x)^2 ]     g_C(x) = C - sum_i c_i x_i
       + [ -lambda_1 g_H(x) + lambda_2 g_H(x)^2 ]     g_H(x) = H - sum_i h_i x_i
```

The first penalty is the usual quadratic penalty for the equality constraint. The other two are the unbalanced penalty of Montanez-Barrera et al. for inequalities: it is smallest at slack g* = lambda_1 / (2 lambda_2), grows quadratically when the limit is exceeded (g < 0), and needs no slack qubits. The price is that it is an approximation: it also shifts the energies of feasible states, so the weights have to be validated rather than assumed.

**Penalty rule.** Weights are set in units of v_max = max |v_i| = {s2[qubo][penalties][v_max]:.3f}. On a grid of A in (0.2 .. 2.0) v_max, lambda_2 in (0.001 .. 1.0) v_max and preferred slack g* in (0 .. 3), a setting is valid when the QUBO ground state over all 1,024 bitstrings is the brute-force optimum. Among valid settings I keep the one with the largest relative gap (energy gap between the optimum and the best other selection, divided by the full energy span), because QAOA only sees the cost function up to scale and a larger relative gap means the optimum stands out more. {s2[qubo][penalties][n_valid]} of {s2[qubo][penalties][n_grid]} settings are valid. Chosen: A = {s2[qubo][penalties][A_over_vmax]} v_max = {s2[qubo][penalties][A]:.3f}, lambda_2 = {s2[qubo][penalties][lambda2_over_vmax]} v_max = {s2[qubo][penalties][lambda_2]:.4f}, lambda_1 = {s2[qubo][penalties][lambda_1]:.4f}. Relative gap {s2[qubo][penalties][relative_gap]:.4f}. The full grid is in `results/stage2_penalty_grid.csv`.

Two honest notes on this rule. First, it uses the brute-force answer, which is only possible because the instance is small; on a large instance the weights would have to come from bounds or from smaller instances of the same family. Second, the two limits cannot bind here (next section), so on this instance the inequality penalties only have to stay out of the way. On the stress instance, where the limits do bind, the same rule picks lambda_2 = {stress[unbalanced_penalties][lambda2_over_vmax]} v_max with preferred slack {stress[unbalanced_penalties][preferred_slack]}. If no setting on the grid is valid, the tuning returns the setting that leaves the fewest selections below the optimum and flags it as not valid. For the organisers' instance the pipeline then stops rather than run QAOA on an invalid encoding; for the stress instance it carries on and reports the flag.

**QUBO, encoding 2: slack variables (exact, {s2[qubo][slack_variant][qubits]} qubits).** Built and verified as the comparison.

```
E(x, s) = - sum_i v_i x_i + A (sum_i x_i - K)^2
          + mu (sum_i c_i x_i + sum_k a_k s_k - C)^2
          + mu (sum_i h_i x_i + sum_k b_k t_k - H)^2
```

Each inequality becomes an equality with an integer slack written in binary on auxiliary qubits. The slack only has to cover the range a group of exactly K customers can need, U = limit - (sum of the K smallest weights), and the last binary coefficient is truncated so the slack can represent 0..U and nothing more. That gives {s2[qubo][slack_variant][auxiliary_qubits]} auxiliary qubits ({d[slack_labels]}) on top of the 10 decision qubits. With A = mu = 2 v_max, enumeration over all 2^{s2[qubo][slack_variant][qubits]} states confirms that the ground state decodes to the brute-force optimum, and that for every feasible selection the best slack assignment has energy exactly minus its net value (largest deviation {s2[qubo][slack_variant][feasible_energy_max_abs_error]:.0e}).

| | Unbalanced | Slack |
|---|---|---|
| Qubits | 10 | {s2[qubo][slack_variant][qubits]} |
| ZZ couplings per QAOA layer | {s2[qubo][zz_couplings]} | {s2[qubo][slack_variant][zz_couplings]} |
| Exact for feasible states | no, validated by enumeration | yes |
| Relative gap | {s2[qubo][penalties][relative_gap]:.4f} | {s2[qubo][slack_variant][ground_state_check][relative_gap]:.5f} |

The slack encoding is exact, but it costs {s2[qubo][slack_variant][auxiliary_qubits]} more qubits and about three times the couplings, and its relative gap is far smaller because the squared slack terms stretch the energy range. That is why the QAOA runs on the unbalanced encoding.

**Ising map.** Substituting x_i = (1 - z_i) / 2 with z_i = +1 for bit 0 and -1 for bit 1:

```
E = sum_i h_i z_i + sum_(i<j) J_ij z_i z_j + const
h_i   = - Q_ii / 2 - (1/4) sum_(j != i) Q_ij
J_ij  = Q_ij / 4
const = offset + (1/2) sum_i Q_ii + (1/4) sum_(i<j) Q_ij
```

where Q is the upper-triangular QUBO matrix (diagonal = linear terms). The cost Hamiltonian is H_C = sum_i h_i Z_i + sum_(i<j) J_ij Z_i Z_j, divided by its largest coefficient so the angles are of order one. A test checks that the Ising energies equal the QUBO energies on all 1,024 states (largest difference in the submitted run: {s2[qubo][ising_vs_qubo_max_abs_difference]:.0e}). The values of h, J and the constant are in `stage2_selection.json`.

## 4. Stage 2 results

**Candidates and selection.**

{t[candidates]}

**Do the limits bind?** No. The six most expensive candidates together cost {s2[constraint_activity][max_cost_of_any_selection]:.0f}, under the budget of {s2[constraint_activity][budget]:.0f}, and the six that need most capacity need {s2[constraint_activity][max_capacity_of_any_selection]:.0f}, under the limit of {s2[constraint_activity][capacity]:.0f}. All {s2[constraint_activity][feasible_selections]} of the {s2[constraint_activity][selections_of_size_k]} groups of six are feasible, so the optimum is simply the six largest expected net values. (Contact costs are at most 4 in the whole dataset, so six customers can never cost more than 24: the budget of 25 cannot bind for any top 10.) I report this because it changes how the Stage 2 numbers should be read: the cardinality constraint is the only one doing work on this instance.

**Brute force.** Optimum {d[bf_ids]}, objective {s2[brute_force][objective_value]:.4f}, cost {s2[brute_force][budget_used]:.0f}, capacity {s2[brute_force][capacity_used]:.0f}. The second-best selection is {d[value_gap]:.3f} lower.

**QAOA, two variants.**

1. *Penalty QAOA (gives the submitted selection).* Start in the uniform superposition, alternate the cost layer exp(-i gamma H_C) and the X mixer exp(-i beta sum X). All three constraints are in the cost function through the unbalanced QUBO above.
2. *Constraint-preserving QAOA (cross-check).* Start in the Dicke state, the equal superposition of all {s2[brute_force][selections_of_size_k]} bitstrings with exactly six ones, and mix with XY rotations exp(-i beta (XX + YY) / 2) on neighbouring qubits, which only move a 1 from one qubit to the next. The number of ones never changes, so "exactly six" holds for every outcome of the ideal circuit and needs no penalty. A limit that cannot bind is dropped from the cost function, since it is implied by the cardinality constraint; a limit that can bind keeps its unbalanced penalty.

Training: the circuit parameters are optimised with COBYLA ({seeds[shots][stage2]} shots are not used for training; the objective is evaluated on the exact output distribution). The objective is CVaR with alpha = 0.25, the mean energy of the best quarter of the distribution. Each depth starts from one linear-ramp schedule, seven seeded random points and one point interpolated from the best parameters of the depth below, and the run with the lowest final objective is kept. The trained circuit is then sampled on Qiskit Aer with {s2[qaoa][shots]} shots.

Optimised parameters at p = 3, penalty QAOA: gamma = ({d[gamma]}), beta = ({d[beta]}). Constraint-preserving QAOA: gamma = ({d[mixer_gamma]}), beta = ({d[mixer_beta]}).

**Decoding.** Qiskit prints qubit 0 as the rightmost character of a count key. `qubo.key_to_bits` reverses the key, so position i of the decoded array is qubit i, which is candidate i. Every measured bitstring is then checked: exactly six ones, cost within budget, capacity within limit. The feasible bitstring with the highest expected net value is the selection (ties: more counts, then the smaller basis index). The selection is taken from the noiseless p = 3 sample of the penalty QAOA.

Most frequent measured bitstrings (penalty QAOA, p = 3, noiseless, {s2[qaoa][shots]} shots):

{t[bitstrings]}

The selection is `{s2[selection_bitstring][qiskit_key]}` (qubit 0 on the right), measured {s2[selection_bitstring][times_measured]} times. Read from the right: qubits {d[set_qubits]} are set, which are {d[ids]}. It equals the brute-force optimum, gap {s2[optimality_gap]:.1f}. The constraint-preserving QAOA decodes to the same six customers.

**How good is the sampler, per shot?**

{t[depth]}

The "ideal" columns come from the exact output distribution of the trained circuit; the noiseless {s2[qaoa][shots]}-shot samples agree with them within sampling error (`results/stage2_qaoa_depth_study.csv`, which also has the circuit depths). The "noisy" columns are {s2[qaoa][shots]}-shot samples on the device noise model. "P(optimum given feasible)" keeps only the feasible bitstrings, which is what the decoding step does. Reference for uniformly random bitstrings: P(optimum) = {s2[qaoa][uniform_random_reference][p_optimal]:.4f}, feasible share = {s2[qaoa][uniform_random_reference][feasible_fraction]:.3f}, P(optimum given feasible) = {s2[qaoa][uniform_random_reference][p_optimal_given_feasible]:.4f}.

- At p = 3 the penalty QAOA puts {s2[qaoa][exact_output_distribution][p_optimal]:.1%} of its probability on the optimum, {d[amp_penalty]:.0f} times the uniform rate, and {s2[qaoa][exact_output_distribution][feasible_fraction]:.0%} on feasible bitstrings. The constraint-preserving QAOA puts {s2[constraint_preserving_qaoa][exact_output_distribution][p_optimal]:.1%} on the optimum ({d[amp_mixer]:.0f} times the uniform rate, and {s2[constraint_preserving_qaoa][exact_output_distribution][p_optimal]:.4f} against {d[subspace_uniform]:.4f} for a uniform draw among the 210 groups of six) and 100% on feasible bitstrings by construction.
- The mean net value of the feasible samples of the penalty QAOA is {s2[qaoa][exact_output_distribution][mean_value_of_feasible_over_optimum]:.0%} of the optimum, against {s2[qaoa][uniform_random_reference][mean_value_of_feasible_over_optimum]:.0%} for a random feasible selection.
- On the noise model both variants fall back towards the uniform reference at every depth. Section 8 goes through why.

![Stage 2 QAOA](results/figures/stage2_qaoa.png)

**Stress instance: limits that bind.** To exercise the constraint handling I tightened both limits to budget {stress[budget]:.0f} and capacity {stress[capacity]:.0f}. The rule, fixed in `config.py`: {stress[rule]}. Only {stress[feasible_selections]} of the 210 groups of six remain feasible, and the optimum changes to {d[stress_ids]} (net value {stress[brute_force][value]:.3f}, cost {stress[brute_force][cost]:.0f}, capacity {stress[brute_force][capacity]:.0f}).

| Stress instance, p = 3, ideal | P(optimum) | Feasible share | P(optimum given feasible) | Best measured feasible bitstring is the optimum |
|---|---|---|---|---|
| Uniform random bitstrings | {stress[uniform_random_reference][p_optimal]:.4f} | {stress[uniform_random_reference][feasible_fraction]:.4f} | {stress[uniform_random_reference][p_optimal_given_feasible]:.4f} | |
| Penalty QAOA | {stress[qaoa_exact_distribution][p_optimal]:.4f} | {stress[qaoa_exact_distribution][feasible_fraction]:.4f} | {stress[qaoa_exact_distribution][p_optimal_given_feasible]:.4f} | {d[stress_penalty_found]} |
| Constraint-preserving QAOA | {stress[constraint_preserving_exact_distribution][p_optimal]:.4f} | {stress[constraint_preserving_exact_distribution][feasible_fraction]:.4f} | {stress[constraint_preserving_exact_distribution][p_optimal_given_feasible]:.4f} | {d[stress_mixer_found]} |

Is the brute-force optimum the ground state of each encoding on the stress instance? Unbalanced, 10 qubits: {d[stress_unbalanced_valid]}. Slack, {stress[slack_qubits]} qubits: {d[stress_slack_valid]}. Unbalanced inside the exactly-six subspace (the cost function of the constraint-preserving QAOA): {d[stress_mixer_valid]}. The penalty QAOA reaches {d[stress_amp_penalty]:.1f} times the uniform P(optimum) here and the constraint-preserving QAOA {d[stress_amp_mixer]:.0f} times: with binding limits, taking the cardinality constraint out of the cost function leaves more of the energy range for the objective and the two inequality penalties. Details: `results/stage2_stress_instance.json`.

## 5. Stage 3: campaign outcome simulation

**Circuit.** One qubit per selected customer. Qubit i is the i-th selected customer in Stage 1 rank order (highest probability first) and gets RY(theta_i) with theta_i = 2 asin(sqrt(p_i)), so P(qubit i reads 1) = sin^2(theta_i / 2) = p_i. No entangling gates, because the customers are assumed independent. A shot is a success when at least three qubits read 1.

{t[qubits]}

**Bit order.** Count keys are reversed before use, so position i is qubit i. The success test only counts ones and would survive a reversed order, so I also check the per-qubit marginals: each measured P(1) must match its own customer's probability, which fails if the order is wrong. The table above shows they match, and `tests/test_stage3.py` runs the same check with six clearly different probabilities and confirms that the reversed order does not fit.

**Results.**

{t[stage3]}

The submitted quantum estimate is the noiseless {s3[estimates][noiseless_shots_submitted][shots]:,}-shot run: **{s3[quantum_estimate]:.6f}**, against the exact **{s3[exact_probability]:.6f}**. The error {s3[errors][quantum_minus_exact]:+.6f} is {d[quantum_z]:.1f} standard errors (one standard error is {s3[quantum_estimate_standard_error]:.6f}). I submit a measured estimate rather than the state-vector value because the task asks for measured outcomes to be counted; the state-vector row shows that the circuit itself reproduces the exact value. The shot count and the seed were fixed in `config.py` before the run.

The exact value is the Poisson-binomial tail computed by dynamic programming over the six customers, and it is cross-checked by summing all 64 outcomes. Distribution of the number of subscribers (0 to 6): {d[pmf]}. Expected number of subscribers: {d[expected_subscribers]:.2f}.

**Counter circuit.** The six-RY circuit does the counting classically, after measurement. As a second design I built a circuit that counts coherently: an adder (`WeightedAdder`) writes the number of ones into a 3-qubit register, a comparator (`IntegerComparator`) sets one flag qubit when the sum is at least 3, and only the flag is measured. P(flag = 1) is the answer; its state-vector value agrees with the exact one to {s3[estimates][counter_circuit][statevector_abs_error]:.0e}. This is the form of circuit that amplitude estimation would start from at scale. In theory amplitude estimation needs quadratically fewer state preparations than sampling. Section 11.2 measures that for these six customers, with a simpler oracle written from the truth table. It uses {s3[estimates][counter_circuit][qubits]} qubits and {s3[transpilation][counter_circuit][two_qubit_gates_transpiled]} two-qubit gates after transpilation, and on the noise model it returns {s3[estimates][counter_circuit][noisy_estimate]:.3f}, which is useless. Amplitude estimation would only make this circuit deeper. Section 11.2 runs it on the lighter six-qubit oracle and finds that it fails on the same noise model.

![Stage 3](results/figures/stage3_simulation.png)

## 6. Circuit design and transpilation

**Backends.** Everything runs on simulators; no real quantum hardware was used.

- Ideal: `qiskit_aer.AerSimulator(method="statevector")`, and `qiskit.quantum_info.Statevector` for exact distributions.
- Noisy: `AerSimulator` with an explicit noise model on a heavy-hex coupling map (`CouplingMap.from_heavy_hex(3)`, 19 physical qubits) and the IBM Eagle basis gates ecr, rz, sx, x. The noise numbers are stated assumptions in the range of public Eagle calibrations, not a calibration of a specific device: depolarising error {seeds[noise_model][one_qubit_depolarizing]} per sx and x gate, {seeds[noise_model][two_qubit_depolarizing]} per ecr gate, readout errors P(read 1 | 0) = {seeds[noise_model][readout_p1_given_0]} and P(read 0 | 1) = {seeds[noise_model][readout_p0_given_1]}. rz is virtual and error free. The model leaves out T1/T2 decay during idle time, crosstalk and drift, all of which add error on real devices.
- Transpilation: `qiskit.transpile`, optimisation level 3. For each circuit I try transpiler seeds 0 to 9 and keep the result that touches the fewest physical qubits, then has the fewest two-qubit gates (then lower depth, then lower seed). The seed kept is in the last column.
- Noisy simulation method: an exact density matrix for circuits up to 11 physical qubits, state-vector trajectories (one noisy run per shot) for wider ones.

{t[transpile]}

How to read the columns. "Original" is the circuit as written (H, RZ, RZZ, RX for the penalty QAOA; the constraint-preserving circuit and the counter circuit contain multi-qubit blocks, listed separately, that the transpiler breaks down). "Physical qubits used" counts the device qubits touched by at least one gate. "SWAPs" is the number of SWAP gates the router inserts, counted by transpiling once more with `swap` kept as a basis gate at the same seed. In the final circuit each SWAP is up to three ECR gates, some of which the optimiser merges with neighbouring gates, so the SWAP count and the ECR count are reported side by side rather than derived from each other.

**Design notes.**

- *Kernel circuit (Stage 1).* Linear entanglement was chosen so the circuit fits a line of qubits on heavy-hex: no SWAPs.
- *Penalty QAOA (Stage 2).* The cardinality penalty couples every pair of qubits, so each layer has 45 RZZ gates on a device where each qubit has two or three neighbours. Routing is what makes it expensive: {s2[qaoa][transpilation][swap_count]} SWAPs and {s2[qaoa][transpilation][two_qubit_gates_transpiled]} ECR gates at p = 3, depth {s2[qaoa][transpilation][depth_transpiled]} against {s2[qaoa][transpilation][depth_original]} as written.
- *Constraint-preserving QAOA (Stage 2).* On this instance the cost layer is single-qubit only and the XY mixer acts on neighbours, so the layers are cheap ({s2[constraint_preserving_qaoa][zz_couplings]} ZZ couplings). The cost has moved into the Dicke state preparation (Bartschi and Eidenbenz), which dominates the gate count.
- *Campaign circuit (Stage 3).* Six single-qubit rotations, depth {s3[transpilation][campaign_circuit][depth_transpiled]} after transpilation (each RY becomes a short sequence of rz and sx gates in the device basis), no two-qubit gates. It is the only circuit in this project that survives the noise model essentially intact.

## 7. Classical vs quantum rigor

Every quantum result has a classical reference next to it, computed independently.

| Quantum result | Classical reference | Outcome |
|---|---|---|
| Quantum kernel classifier, hybrid | Logistic regression, gradient boosting, RBF kernel SVM on the same inputs, same folds | No quantum gain. The quantum kernel alone is behind logistic regression and the RBF control on PR-AUC; the hybrid is level with logistic regression within the fold scatter. |
| Kernel entries from the overlap circuit | Exact state-vector kernel | Mean absolute difference {kernel[mean_abs_error_ideal]:.4f} at {kernel[shots]} shots |
| QUBO ground state (both encodings) | Brute force over all bitstrings | Identical selection, on the real and the stress instance |
| QAOA selection | Brute-force optimum | Identical, gap {s2[optimality_gap]:.1f} |
| QAOA sampling quality | Uniform random bitstrings | {d[amp_penalty]:.0f} times (penalty) and {d[amp_mixer]:.0f} times (constraint-preserving) the uniform P(optimum) at p = 3, ideal; {d[amp_penalty_noisy]:.1f} and {d[amp_mixer_noisy]:.1f} times on the noise model |
| Stage 3 circuit estimate | Exact Poisson-binomial, enumeration, Monte Carlo | Error {s3[errors][quantum_minus_exact]:+.6f} (quantum) and {s3[errors][monte_carlo_minus_exact]:+.6f} (Monte Carlo, {s3[estimates][monte_carlo][trials]:,} draws) |

The emulators are not a separate source of truth. Training QAOA on Qiskit state vectors directly would be slow, so the feature-map states and the QAOA states are computed by short NumPy routines that apply the same gates. Tests compare each of them with `qiskit.quantum_info.Statevector` on the actual Qiskit circuits, amplitude by amplitude for the feature map and the Dicke state, and up to a global phase for QAOA.

What I do not claim:

- No quantum advantage, in accuracy or in speed, at any stage. The problem has 10 binary variables; brute force needs 1,024 evaluations and takes milliseconds.
- QAOA "finding the optimum" is not evidence of quality by itself. With {s2[qaoa][shots]} shots on a 10-qubit problem, random bitstrings find it too. The per-shot probabilities above are the fair measure.
- The Stage 3 circuit samples a distribution that a classical computer samples just as easily. Its sampling error falls as one over the square root of the number of shots, exactly like Monte Carlo, and the table in the next section shows that.

## 8. Error and noise analysis

**Stage 3: ideal, finite shots, noisy, mitigated.** The four rows of the Stage 3 table again, with what explains each error:

| Setting | Estimate | Error | What causes the error |
|---|---|---|---|
| Ideal (state vector) | {s3[estimates][statevector][estimate]:.6f} | {s3[estimates][statevector][error]:+.1e} | floating point only |
| Noiseless, {s3[estimates][noiseless_shots][shots]:,} shots | {s3[estimates][noiseless_shots][estimate]:.6f} | {s3[estimates][noiseless_shots][error]:+.6f} | sampling: one standard error is {s3[estimates][noiseless_shots][standard_error]:.4f} |
| Noisy, unmitigated, {s3[estimates][noisy_unmitigated][shots]:,} shots | {s3[estimates][noisy_unmitigated][estimate]:.6f} | {s3[estimates][noisy_unmitigated][error]:+.6f} | sampling plus a readout bias predicted at {s3[estimates][noisy_unmitigated][expected_bias]:+.6f} |
| Noisy, readout mitigated | {s3[estimates][noisy_readout_mitigated][estimate]:.6f} | {s3[estimates][noisy_readout_mitigated][error]:+.6f} | sampling of the run and of the calibration: one standard error is {s3[estimates][noisy_readout_mitigated][standard_error]:.4f} |

Shot noise, measured with an independent seed per shot count:

{t[sweep]}

The errors follow the one-over-square-root law within the scatter one expects from single runs.

*What the noise does to this circuit.* The circuit has no two-qubit gates, so readout error is what matters. Readout turns each customer's probability p into p (1 - e10) + (1 - p) e01, where e01 = {seeds[noise_model][readout_p1_given_0]} and e10 = {seeds[noise_model][readout_p0_given_1]}. Customers with p below e01 / (e01 + e10) = {d[crossover]:.3f} read slightly too high and customers above it slightly too low. My six probabilities run from {d[p_lo]:.2f} to {d[p_hi]:.2f}, on both sides of that point, so the shifts partly cancel. Putting the shifted probabilities into the exact formula predicts a bias of {s3[estimates][noisy_unmitigated][expected_bias]:+.4f}. A small bias is a property of these probabilities and this noise model, not a sign that noise is harmless in general.

*Mitigation.* Tensored readout mitigation: two calibration circuits (all qubits in 0, all qubits in 1) give a 2 x 2 confusion matrix per qubit, and the inverse of their tensor product is applied to the measured distribution. The result is a quasi-distribution whose entries can be slightly negative (the smallest here is {s3[estimates][noisy_readout_mitigated][most_negative_quasi_probability]:.4f}). Mitigation is designed to remove the readout bias on average, under the assumption that readout errors are independent per qubit, which holds for this noise model by construction. It does not remove sampling error, and it adds the sampling error of the calibration: the standard error of the mitigated estimate, from a parametric bootstrap of the run and the calibration, is {s3[estimates][noisy_readout_mitigated][standard_error]:.4f}, against {s3[estimates][noisy_unmitigated][standard_error]:.4f} for the unmitigated one. Both errors in the table are within about one of those standard errors, so this single run cannot show the benefit of mitigation by itself (here the mitigated estimate even lands further from the exact value than the unmitigated one, in the other direction). A test checks the method exactly: it pushes the true distribution through known per-qubit confusion matrices and recovers it to 1e-12.

**Stage 2 under noise.** At p = 3 the penalty QAOA uses {s2[qaoa][transpilation][two_qubit_gates_transpiled]} ECR gates. As a rough guide, (1 - {seeds[noise_model][two_qubit_depolarizing]})^{s2[qaoa][transpilation][two_qubit_gates_transpiled]} is about {d[survival]:.1%}: if gate errors were independent events, only about {d[survival]:.0%} of shots would pass through the circuit without any two-qubit error. This is a heuristic for how much signal can survive, not a measured probability of a wrong bitstring. The measured optimum probability falls from {s2[qaoa][noiseless_sample][p_optimal]:.4f} to {s2[qaoa][noisy_sample][p_optimal]:.4f} and the feasible share from {s2[qaoa][noiseless_sample][feasible_fraction]:.3f} to {s2[qaoa][noisy_sample][feasible_fraction]:.3f}, against {s2[qaoa][uniform_random_reference][p_optimal]:.4f} and {s2[qaoa][uniform_random_reference][feasible_fraction]:.3f} for uniform noise. The best feasible bitstring in the noisy sample is still the optimum, for the reason given above: at 10 qubits almost any sampler with {s2[qaoa][shots]} shots sees it. The depth table shows the trade-off: more layers help the ideal circuit and cost more two-qubit gates, and on the noise model no depth stays far above the uniform reference. The constraint-preserving QAOA loses its guarantee under noise: its feasible share at p = 3 drops from 100% to {d[mixer_feasible_noisy]:.0%}, because gate errors change the number of ones. Post-selecting on feasibility, which the decoding does anyway, recovers part of the signal (the "given feasible" columns). I did not apply error mitigation to the QAOA samples. Readout mitigation would correct a small part of the error budget, and the gate errors that dominate would need methods such as zero-noise extrapolation, which estimate expectation values rather than repair individual bitstrings.

**Stage 1 kernel under noise.** The overlap circuit has {kernel[transpile][two_qubit_gates_transpiled]} two-qubit gates. Noise lowers the estimated kernel entries by {kernel[mean_signed_error_noisy]:+.3f} on average (mean absolute error {kernel[mean_abs_error_noisy]:.4f}, against {kernel[mean_abs_error_ideal]:.4f} from shots alone).

**Error propagation across the stages.** The Stage 3 answer inherits three kinds of error. I measured each.

| Source | Size, in units of P(at least 3) | Type | How it was measured |
|---|---|---|---|
| Shot noise of the submitted estimate | {d[shot_se]:.4f} | standard error | binomial formula, {s3[estimates][noiseless_shots_submitted][shots]:,} shots |
| Device noise on the Stage 3 circuit | {d[noise_bias]:.4f} | predicted bias (absolute value) | exact formula with readout-shifted probabilities |
| Stage 1 model uncertainty | {prop[stage3][fixed_six][sd]:.3f} | standard deviation across refits | {prop[rounds]} bootstrap refits, same six customers |

For the last row I refit the submitted Stage 1 model on {prop[rounds]} bootstrap resamples of the {sel[selection][training_rows]} training rows (stratified by label) and pushed each set of 100 test probabilities through the same ranking, the exact Stage 2 solver and the exact Stage 3 formula. No test label is involved.

*Stage 1 to the top 10.* On average {prop[top10][mean_overlap_with_submitted_top10]:.1f} of my ten candidates are still in the top 10 of a refit.

{t[top10]}

*Stage 1 to Stage 2.* Keeping the ten candidates fixed and changing only their probabilities, the submitted six are still the exact optimum in {d[same_selection_count]} of {prop[rounds]} refits ({prop[stage2][fixed_candidates_same_selection_frequency]:.1%}). Measured with a refit's own probabilities, keeping my six instead of that refit's optimum gives up {prop[stage2][fixed_candidates_regret][p50]:.2f} units of net value at the median (5th to 95th percentile: {prop[stage2][fixed_candidates_regret][p05]:.2f} to {prop[stage2][fixed_candidates_regret][p95]:.2f}), on an objective of {s2[objective_value]:.2f}. This sensitivity analysis uses the exact brute-force solver, so all of the movement comes from the Stage 1 inputs. (The submitted QAOA selection matched the brute-force optimum on the submitted instance.)

*Stage 1 to Stage 3.* With the six customers fixed:

{t[six]}

P(at least 3) across the refits has median {prop[stage3][fixed_six][p50]:.3f}, 5th to 95th percentile {prop[stage3][fixed_six][p05]:.3f} to {prop[stage3][fixed_six][p95]:.3f}, standard deviation {prop[stage3][fixed_six][sd]:.3f}. A first-order calculation from the sensitivities in the last column and the covariance of the six probabilities gives a standard deviation of {prop[stage3][delta_method_sd]:.3f}; the two agree in order of magnitude, and they need not agree better, because the spread is wide for a linear approximation. The sensitivities are exact: dP/dp_i is the probability that exactly two of the other five subscribe.

The same six customers come out below the submitted value {s3[exact_probability]:.3f} in {d[submitted_percentile]:.0%} of the refits (median {prop[stage3][fixed_six][p50]:.3f}). When each refit picks its own best six, the median is {prop[stage3][full_rerun][p50]:.3f}, also below the submitted value. What this establishes is that the answer is very sensitive to the training sample and to the selection step. It is not a calibrated interval for the true probability, and it does not by itself prove which way the submitted number is off. Two mechanisms would both push in this direction, and I did not separate them: customers chosen because one fit rated them highly tend to rate lower on another fit (the winner's curse), and a bootstrap resample holds fewer distinct customers than the full training set, which may make the refitted models more cautious. The cross-validated calibration slope of {d[cv][hybrid][calibration_slope]:.2f} points the same way for the highest probabilities.

*What if Stage 1 had been a different model?* The same Stages 2 and 3, solved exactly, with each model's test probabilities. Only test features are used, no labels, and nothing was chosen from this table.

{t[models]}

The classical baseline alone would select {d[logistic_shared]} of the same six customers and give P(at least 3) = {d[logistic_p3]:.3f}. The final answer depends far more on the Stage 1 model than on anything in Stages 2 and 3.

![Error propagation](results/figures/error_propagation.png)

## 9. Setup and execution

Python 3.12. Tested on macOS (Apple silicon); nothing is platform specific.

```bash
git clone https://github.com/sidharth-n/quantum-campaign-targeting.git
cd quantum-campaign-targeting
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py           # rebuilds every result file and figure
python run_extensions.py         # rebuilds the studies of section 11 (one to two minutes)
python -m pytest                 # all tests, including one full rerun of the pipeline
python -m pytest -m "not slow"   # the same without the full rerun
```

Timings on my laptop (Apple silicon, 10 cores): the pipeline takes two to four minutes, the extension studies one to two minutes, the full test suite four to six minutes, the fast tests under a minute. The code needs no account, token or network access after the install and uses nothing specific to my machine, so the same three commands should work in a cloud environment such as Google Colab or a GitHub Codespace; I have not run it there.

`run_pipeline.py` writes the five required files to the repository root and the detail files to `results/`, reads the three stage files back and stops if an id or probability differs between them (`results/traceability.csv`), and then refreshes this README from the result files (`build_report.py`). `python run_pipeline.py --out some/dir` writes everything to another folder and leaves the repository untouched. The notebook `notebooks/walkthrough.ipynb` is committed with its outputs and can be rerun with Jupyter.

**Versions.** Python {env[python]}, qiskit {env[packages][qiskit]}, qiskit-aer {env[packages][qiskit-aer]}, scikit-learn {env[packages][scikit-learn]}, numpy {env[packages][numpy]}, scipy {env[packages][scipy]}, pandas {env[packages][pandas]}. Pinned in `requirements.txt`, recorded in `results/environment.json`. The IBM device snapshots of section 11.3 come from qiskit-ibm-runtime {x[runtime_version]}, also pinned; it is used offline, with no account.

**Seeds.** All in `src/campaign/config.py`, recorded in `results/seeds.json`.

{t[seeds]}

**Shots.** Stage 2: {seeds[shots][stage2]} per circuit. Stage 3: {seeds[shots][stage3_submitted_estimate]:,} for the submitted estimate, {seeds[shots][stage3]:,} for the noiseless, noisy and calibration runs, {seeds[shots][stage3_counter_noisy]:,} for the noisy counter circuit. Kernel check: {seeds[shots][kernel_check]} per pair. Monte Carlo: {seeds[monte_carlo_trials]:,} draws. Bootstrap: {seeds[bootstrap_rounds]} refits.

**Reproducibility.** `tests/test_reproducibility.py` runs the whole pipeline into a temporary folder and compares every result file with the committed one, byte for byte (only `results/environment.json` is excluded, since it records the machine). `tests/test_extensions.py` does the same for `run_extensions.py` and also checks that the five stage files are not modified by it. I ran it on one machine, both in my working copy and from a fresh clone of this repository with a new virtual environment built from `requirements.txt` alone. Across platforms the last digits of floating-point results can differ, which is why the Stage 1 probabilities are rounded once to 12 decimals before anything else uses them.

**What the tests cover.** No test-set leakage (scaler statistics, one-hot levels, predictions independent of the other test rows, cross-validation splits, calibration scores that never see their own row or label); the tie-break rule; the schema of the three result files; ids and probabilities identical across the files; Stage 2 validity and the objective recomputed from scratch; brute force against an independent enumeration; both QUBO encodings against brute force on the organisers' instance, the stress instance and random instances; the stress rule keeping its two properties; Ising energies against QUBO energies on all states; emulators against Qiskit state vectors; Dicke state; Qiskit bit order; Stage 3 exact value three ways; qubit-to-customer order; readout mitigation on a known case; the counter circuit; the traceability check refusing corrupted files; this README matching the result files; byte-identical reruns. For section 11: the correlated circuit against the exact mixture on every outcome, its marginals, its control-qubit order and the money statistics on a hand-computed case; the Grover identity for every power, the likelihood estimator and the cost count; the rebuilt QAOA circuits against the Stage 2 values; calibration circuits on the same physical qubits; the hardware post-processing on simulated counts; the extension files rerun byte for byte with the stage files untouched.

## 10. Business impact

**Recommendation.** Contact {d[ids]}. This is the best group of six among my ten candidates: expected net value {s2[objective_value]:.2f}, contact cost {s2[budget_used]:.0f} of a budget of {s2[constraints][budget]:.0f}, capacity {s2[capacity_used]:.0f} of {s2[constraints][capacity]:.0f}.

**What to expect.** Do not plan on three subscriptions. My model gives a {s3[exact_probability]:.0%} chance of reaching three or more, so falling short is the more likely outcome ({d[p_fewer_than_3]:.0%}). It expects {d[expected_subscribers]:.1f} subscribers out of six, and the single most likely count is {d[mode]} ({d[p_mode]:.0%}). The error analysis adds a caution: when the model is refitted on resampled training data, the same six customers usually get a lower chance (median {prop[stage3][fixed_six][p50]:.0%}).

**What the bank can do with this.**

- *The limits are not the bottleneck, the "exactly six" rule is.* The selection leaves {d[budget_left]:.0f} units of budget and {d[capacity_left]:.0f} units of capacity unused, and {d[n_positive]} of the ten candidates have a positive expected net value (the range is {d[min_value]:.2f} to {d[max_value]:.2f}). If the campaign size were a choice and not a rule, the best campaign inside the same two limits would contact {d[free_size]} of these ten ({d[free_ids]}) for an expected net value of {d[free_value]:.2f}, at cost {d[free_cost]:.0f} and capacity {d[free_capacity]:.0f}.
- *Targeting works, modestly.* In cross-validation the highest fifth of predictions subscribed at {d[top_observed]:.0%}, against {d[base_rate]:.0%} overall ({d[top_ratio]:.1f} times the base rate). That is real value per call, but it is not a precise ranking: on average {prop[top10][mean_overlap_with_submitted_top10]:.1f} of the top 10 stay in the top 10 when the model is refitted.
- *More labelled data would help more than anything else.* The whole pipeline rests on {sel[selection][training_positives]} past subscribers. Stage 1 uncertainty is about {d[model_over_shot]:.0f} times larger than the Stage 3 shot noise and far larger than the device noise on the Stage 3 circuit. Recording the outcome of this campaign and retraining is the cheapest improvement available.
- *Do not plan on the single number.* Across the refits the same six customers give {prop[stage3][fixed_six][p05]:.0%} to {prop[stage3][fixed_six][p95]:.0%} (5th to 95th percentile), mostly below the submitted {s3[exact_probability]:.0%}. That spread is not a calibrated forecast, but it is a fair warning that a different sample of past customers would have produced a noticeably different promise for the same six people.

**Where quantum stands for this use case.** At 10 candidates a laptop solves the selection exactly in milliseconds, and the quantum optimiser is slower and, on today's noise levels, close to random. What the project does establish is a verified formulation at small size. The QUBO, the Ising map and the decoding extend to larger candidate pools as they are; the exhaustive checks I relied on here (brute force, enumeration of every bitstring, exact emulation of the circuit) do not, so a larger instance would need other evidence that the encoding and the optimiser are right. Whether QAOA is competitive there is an open research question, and this project is not evidence either way.

## 11. Beyond the brief

The brief fixes one circuit and one number. This section asks three questions it does not ask. None of them touches the submitted answer: `run_extensions.py` reads the committed stage files and writes its results to `results/extensions/`, one figure to `results/figures/`, and refreshes this README. A test checks that the five stage files are unchanged by a run and that a rerun reproduces the extension files byte for byte.

![Beyond the brief](results/figures/beyond_the_brief.png)

### 11.1 What if the six customers are not independent?

Stage 3 assumes that customers respond independently. That is why its circuit has no entangling gate, and why a laptop can write the answer down. Real campaigns are not like that: the month, interest rates or a competing offer move everyone at once. Banks handle the same problem in credit portfolios with a one-factor Gaussian copula, and Woerner and Egger (2019) showed how to load that model on qubits. I did this for the six selected customers.

- **Model.** A market factor Z takes {x[scenarios]} equally likely values z_k = Phi^-1((k + 1/2) / {x[scenarios]}). Given z_k, customer i subscribes with probability p_i(z_k) = Phi((c_i + sqrt(rho) z_k) / sqrt(1 - rho)), and customers are independent given the factor. I solve c_i so that the average of p_i(z_k) over the scenarios is exactly the Stage 1 probability p_i. Every customer keeps their Stage 1 probability. Only the dependence between customers changes, and rho = 0 gives back the Stage 3 distribution exactly.
- **Circuit.** {corr[factor_qubits]} factor qubits in a uniform superposition (Hadamard gates), then one uniformly controlled RY per customer, controlled by the factor register, with angle 2 asin sqrt(p_i(z_k)) when the register holds k. That is {corr[circuit][logical_qubits]} qubits. The factor register is entangled with every customer qubit, and that is what correlates the customers. Only the customer qubits are measured, so the decoding is the same as in Stage 3. On the heavy-hex device model the circuit has depth {corr[circuit][depth_transpiled]} and {corr[circuit][two_qubit_gates_transpiled]} ECR gates.
- **Checks.** The state vector of the circuit equals the exact model, a mixture of product distributions, on all 64 outcomes (largest difference {x[max_statevector_error]:.0e}). At rho = 0 the exact value equals the Stage 3 value. The subscriber count is then a mixture of Poisson-binomial distributions. The sampled marginals stay on the Stage 1 probabilities at every rho (largest deviation {x[max_marginal_deviation]:.4f}). A test on the joint distribution of both registers pins the order of the control qubits.

Net value of an outcome is the expected deposit value of the customers who subscribe minus the contact cost of all six, the same money terms Stage 2 uses. The workbook gives one expected deposit value per customer and no distribution of payoffs, so I take that value as the amount received when the customer subscribes. Every money figure below is a result of the model under that assumption, not an observed business probability. It is computed from the same measured bitstrings. "Loses money" means the net value is below zero.

{t[correlated]}

What this says:

- **The headline number is the least sensitive to the assumption.** P(at least {corr[threshold]}) moves from {x[p_first]:.3f} to {x[p_last]:.3f} across the whole range. It goes up, which looks like good news and is not. The expected number of subscribers stays at {x[expected_subscribers]:.2f}, below three. Correlation only pushes probability to both ends of the count.
- **The downside is the most sensitive.** The chance that none or only one of the six subscribes goes from {x[low_first]:.0%} to {x[low_last]:.0%}, and the model's chance that the campaign loses money goes from {x[loss_first]:.0%} to {x[loss_last]:.0%}, {x[loss_ratio]:.1f} times as likely. In the worst quarter of outcomes the mean net value falls from {x[es_first]:.1f} to {x[es_last]:.1f}.
- **Stage 2 cannot see any of this.** The expected net value is {x[expected_net]:.2f} at every rho, because an expectation does not depend on correlation. The selection objective and the submitted probability would both report "no change" while the risk doubles.
- **Rho is the factor parameter, not the correlation you would observe.** The second column gives the correlation between two customers' outcomes, which is about half of rho here.
- **I do not know rho.** The workbook holds one snapshot and no repeated campaigns, so the correlation cannot be estimated from it. The table is a range of what-ifs. The practical reading is that a bank that believes its customers move together should judge this campaign by the loss probability, not by the chance of three subscriptions.

### 11.2 Can the probability be read with fewer circuit runs?

Sampling the Stage 3 circuit N times gives an error that falls like 1 / sqrt(N). Amplitude estimation reaches an error close to 1 / N in the number of times the state is prepared. I built it for the six customers and measured the scaling, first on an ideal simulator and then on the noise model.

- **Circuit.** A is the Stage 3 state preparation. S_chi flips the sign of every outcome with at least {ae[threshold]} ones, S_0 flips the sign of the all-zero state, and Q = A S_0 A^-1 S_chi is the Grover operator. After k applications of Q the probability of a "good" outcome is sin^2((2k + 1) phi), where sin^2(phi) is the probability I want. The state vector confirms this for every power used (largest difference {x[ae_identity]:.0e}).
- **Estimator.** Maximum-likelihood amplitude estimation (Suzuki et al. 2020): sample Q^k A for k = 0, 1, 2, 4, ... and take the phi that maximises the joint likelihood. It needs no phase estimation register and no controlled Q, only the {ae[qubits]} customer qubits.
- **Measurement.** {x[ae_shots_per_power]} shots per circuit, each schedule repeated {x[ae_repeats]} times, root mean square error against the exact value. Direct sampling is measured the same way, with the same number of repetitions.
- **What is counted.** Cost is the number of applications of A or its inverse: one shot of Q^k A uses 2k + 1. This is the query count in which the quadratic speed-up is stated. It leaves out the two reflections inside Q, which hold almost all of the physical gates, so it is not a run-time comparison.

{t[amplitude]}

Columns three to five are the ideal simulator. For this instance a straight-line fit through the {x[ae_points]} points on a log-log plot gives slope {x[ae_slope]:.2f} for amplitude estimation and {x[ae_direct_slope]:.2f} for direct sampling (theory: -0.50). With {x[ae_cost]:,} applications the error is {x[ae_rmse]:.5f}, {x[ae_ratio]:.1f} times smaller than direct sampling on the same budget, which would need about {x[ae_direct_shots]:,.0f} shots to match it. At the smallest budget amplitude estimation is the worse of the two.

The last two columns repeat the whole study on the stated noise model, with the same schedules and the same estimator. There amplitude estimation fails: its error is between {x[ae_noisy_rmse_min]:.2f} and {x[ae_noisy_rmse_max]:.2f} at every budget, almost all of it bias ({x[ae_noisy_bias]:+.2f} at the largest), while direct sampling under the same noise reaches {x[ae_noisy_direct]:.4f}. The reason is in the Grover circuits themselves:

{t[amplitude_noisy]}

One Grover step needs {x[ae_one_step_gates]} two-qubit gates on heavy-hex, and the amplified probability reads {x[ae_one_step_noisy]:.3f} where it should read {x[ae_one_step_ideal]:.3f}. For powers 4 to {x[ae_deep_power]} the measured probability stays between {x[ae_deep_low]:.3f} and {x[ae_deep_high]:.3f} whatever the ideal value is, close to the {x[ae_uniform_bits_readout]:.3f} that uniformly random bits give after the stated readout error. For this circuit the estimator is fitting a sine curve to numbers that carry almost no signal. So on this noise model direct sampling is the right choice, and the submitted estimate uses it. I did not test other noise models or error mitigation for these circuits.

Two more limits. S_chi is written from the truth table of 64 outcomes; the scalable form is the adder and comparator of the counter circuit in section 5. And at six qubits the exact answer is one line of classical code, so this measures how the algorithm scales on this instance. It is not a quantum advantage.

### 11.3 What happens on the calibration of real IBM machines?

The noise model in section 8 has four stated numbers. Here the same circuits run on the calibration snapshots of two real IBM processors that ship with qiskit-ibm-runtime {x[runtime_version]}: coupling map, native gates, gate and readout errors, T1 and T2 of the device, turned into a noise model with thermal relaxation by `AerSimulator.from_backend`. Every circuit touches at most ten physical qubits, so the simulator evolves the exact density matrix of the noisy circuit and only the shot noise is statistical.

{t[ibm_devices]}

Stage 3 circuit, {x[ibm_shots]:,} shots, with the two calibration circuits placed on the same physical qubits:

{t[ibm_stage3]}

Readout error pushes the raw estimate up by about 0.004 on both devices. The tensored readout mitigation brings both back to within their standard error of the exact value.

Stage 2, the trained QAOA circuits, {x[ibm_qaoa_shots]:,} shots each. The stated noise model of the main pipeline is rerun with the same shot count, so the three noise sources are compared like for like (section 4 reports its {s2[qaoa][shots]}-shot run). There are two references. A random bitstring is the optimum with probability {x[random_bitstring]:.4f}. A random valid selection, six of the ten at random, is the optimum with probability {x[random_valid]:.4f}, because all {x[valid_selections]} such selections are feasible here. The second one is the reference a sensible classical sampler would set. The ± values are one standard error.

{t[ibm_stage2]}

What this says:

- **Torino is clearly the best of the three.** At p = 3 it returns the optimum in {x[t3][p_optimal]:.2%} of shots, {x[torino_over_stated]:.1f} times the stated noise model ({x[m3][p_optimal]:.2%}). Its median two-qubit error is {x[error_ratio_devices]:.0%} of Brisbane's.
- **My stated noise model sits next to Brisbane.** At p = 3 they give {x[m3][p_optimal]:.2%} and {x[b3][p_optimal]:.2%}, {x[stated_minus_brisbane_z]:.1f} combined standard errors apart. For this circuit at p = 3, the four numbers I assumed behave like the Brisbane snapshot and are pessimistic next to the Torino snapshot. Two snapshots and one circuit do not say more than that about either processor family.
- **Against a random valid selection, only Torino at p = 3 comes out ahead per shot** ({x[t3][p_optimal]:.4f} against {x[random_valid]:.4f}). Brisbane and the stated model stay below it at every depth. All are far below the ideal {x[ideal_p3]:.4f}.
- **Most noisy shots are not valid selections.** Only {x[right_size_low]:.0%} to {x[right_size_high]:.0%} of them select exactly six customers (the ideal p = 3 circuit: {s2[qaoa][exact_output_distribution][feasible_fraction]:.0%}; random bits: {x[random_right_size]:.0%}). Among the shots that do, every noise source beats the random valid selection at p = 3 ({x[t3][p_optimal_given_right_size]:.4f}, {x[b3][p_optimal_given_right_size]:.4f} and {x[m3][p_optimal_given_right_size]:.4f} for Torino, Brisbane and the stated model).

The SWAP column is a routing diagnostic: a second compilation from the same starting qubits and the same seed that keeps SWAP as a gate. The circuit that is run has those SWAPs merged into native gates, so the two-qubit gate count is not derived from it. A snapshot is a past calibration, not the device today. The simulation has no crosstalk or drift, and each number is one seeded run.

### 11.4 On a real quantum processor

{t[hardware]}

## 12. Limitations

- **Small data.** 300 training rows with 35 subscribers. Model comparisons are within noise. If the test set resembles the training set, it holds roughly a dozen subscribers, so test metrics will be noisy too.
- **Model selection is optimistic.** Hyperparameters were chosen with the same cross-validation folds that produce the reported scores. The scores are for comparing models, not unbiased estimates of test performance.
- **The hybrid was a design choice.** On PR-AUC alone, logistic regression is ahead of the hybrid ({d[cv][logistic][pr_auc]:.3f} against {d[cv][hybrid][pr_auc]:.3f}). The difference is inside the fold-to-fold scatter, but I did not let the score make the choice.
- **Calibration at the top.** The calibration slope is {d[cv][hybrid][calibration_slope]:.2f}, below 1, so the largest probabilities, which are the ones Stages 2 and 3 use, are probably somewhat too high.
- **Independence.** The submitted Stage 3 number assumes customers respond independently, as the task states. Section 11.1 shows what a common cause does to it. The correlation itself cannot be estimated from this workbook.
- **Penalty weights use the answer.** The unbalanced-penalty weights are validated against brute force. That is honest for a 10-qubit instance and not available at scale.
- **QAOA is trained on the exact distribution.** A real device would have to estimate the objective from shots, which adds noise to the optimisation itself. I did not study that.
{t[noise_limitation]}
- **The limits do not bind.** The constraint handling is therefore tested on the stress instance, not on the organisers' instance.
- **Public source data.** The workbook names its source as `bank.csv`, a public dataset, so the hidden test labels could in principle be looked up. I did not download or use that file. The pipeline reads the organisers' workbook and nothing else, and a judge can check that in `src/campaign/data.py`.
- **Extensions are what-ifs.** Section 11 uses an assumed range for the correlation, counts state preparations and not gates for amplitude estimation, and uses past calibration snapshots in one seeded simulation each. None of it enters the submitted files.
- **One machine.** Byte-identical reruns were checked on one machine only (working copy and fresh clone).

## 13. References

- S. Moro, P. Cortez, P. Rita. A data-driven approach to predict the success of bank telemarketing. Decision Support Systems 62 (2014). The bank marketing dataset.
- V. Havlicek et al. Supervised learning with quantum-enhanced feature spaces. Nature 567 (2019). Quantum kernels and the ZZ feature map.
- R. Shaydulin, S. M. Wild. Importance of kernel bandwidth in quantum machine learning. Physical Review A 106 (2022).
- J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods (1999).
- E. Farhi, J. Goldstone, S. Gutmann. A quantum approximate optimization algorithm. arXiv:1411.4028 (2014).
- A. Lucas. Ising formulations of many NP problems. Frontiers in Physics 2 (2014).
- P. Barkoutsos et al. Improving variational quantum optimization using CVaR. Quantum 4 (2020).
- L. Zhou et al. Quantum approximate optimization algorithm: performance, mechanism, and implementation on near-term devices. Physical Review X 10 (2020). The interpolated start.
- A. Montanez-Barrera et al. Unbalanced penalization: a new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms. Quantum Science and Technology 9 (2024).
- S. Hadfield et al. From the quantum approximate optimization algorithm to a quantum alternating operator ansatz. Algorithms 12 (2019).
- A. Bartschi, S. Eidenbenz. Deterministic preparation of Dicke states. FCT 2019, arXiv:1904.07358.
- S. Woerner, D. J. Egger. Quantum risk analysis. npj Quantum Information 5 (2019). Loading a correlated portfolio on qubits.
- D. J. Egger et al. Credit risk analysis using quantum computers. IEEE Transactions on Computers 70 (2021). The one-factor Gaussian copula circuit.
- G. Brassard, P. Hoyer, M. Mosca, A. Tapp. Quantum amplitude amplification and estimation. Contemporary Mathematics 305 (2002).
- Y. Suzuki et al. Amplitude estimation without phase estimation. Quantum Information Processing 19 (2020). The maximum-likelihood estimator.
- Qiskit, Qiskit Aer and qiskit-ibm-runtime documentation, docs.quantum.ibm.com.

Licence: MIT.
