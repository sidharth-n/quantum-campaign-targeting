"""Paths, seeds and run settings.

Every random source in the pipeline reads its seed from this file, and every tuning grid is
declared here before any model is fitted. Nothing in this file depends on the data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "bank_campaign_participant_release.xlsx"

# --------------------------------------------------------------------------- seeds
SEED = 42                 # CV splits, PCA, logistic regression, boosting, Platt calibration folds
QAOA_SEED = 2026          # start points of the QAOA outer loop
SIM_SEED = 1234           # Aer simulator seed (seed_simulator) for every sampled circuit
TRANSPILE_SEEDS = tuple(range(10))   # transpiler seeds tried; fewest physical qubits, then fewest 2-qubit gates
MC_SEED = 777             # classical Monte Carlo benchmark in stage 3
BOOTSTRAP_SEED = 99       # bootstrap refits for the error propagation study

# --------------------------------------------------------------------------- stage 1
CV_SPLITS = 5             # stratified K-fold on the 300 training customers only
CV_REPEATS = 3            # repeated with a different shuffle each time
PLATT_FOLDS = 5           # inner folds that give out-of-fold SVM scores for Platt scaling

KERNEL_QUBITS = 6         # qubits in the quantum kernel feature map (= reduced input dimensions)
LOGISTIC_GRID = [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)]
KERNEL_GRID = [
    {"reducer": reducer, "reps": reps, "bandwidth": bw, "C": c}
    for reducer in ("pca", "pls")
    for reps in (1, 2)
    for bw in (0.1, 0.25, 0.5)
    for c in (0.1, 1.0, 10.0)
]
RBF_GRID = [   # classical kernel control: same inputs, Gaussian kernel instead of the circuit
    {"reducer": reducer, "bandwidth": bw, "C": c}
    for reducer in ("pca", "pls")
    for bw in (0.1, 0.25, 0.5)
    for c in (0.1, 1.0, 10.0)
]
HYBRID_WEIGHT = 0.5       # fixed, never tuned
SUBMITTED_MODEL = "hybrid"   # design decision, not a score-based choice (see stage1.py)
PROBABILITY_DECIMALS = 12 # stage 1 probabilities are rounded once to this many decimals
PR_AUC_TIE = 0.005        # PR-AUC values closer than this count as tied; lower Brier then wins
KERNEL_CHECK_PAIRS = 12   # kernel entries re-estimated by sampling the overlap circuit
KERNEL_CHECK_SHOTS = 8192

# --------------------------------------------------------------------------- stage 2
QAOA_LAYERS = 3           # QAOA depth p used for the selection
QAOA_LAYER_SWEEP = (1, 2, 3, 4, 5)   # depths trained for the depth-against-noise study
XY_LAYER_SWEEP = (1, 2, 3)           # depths of the constraint-preserving (Dicke + XY mixer) variant
QAOA_RESTARTS = 8         # seeded start points per depth (1 linear ramp + 7 random), plus one
                          # start interpolated from the best parameters of the previous depth
QAOA_MAXITER = 600        # COBYLA iterations per start
QAOA_CVAR_ALPHA = 0.25    # CVaR fraction used as the training objective
SHOTS_STAGE2 = 8192
# penalty grids, in units of v_max = max |expected net value| (see qubo.tune_unbalanced)
CARDINALITY_GRID = (0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)   # A
UNBALANCED_L2_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)    # lambda_2
UNBALANCED_SLACK_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)            # preferred slack g* = lambda_1 / (2 lambda_2)
SLACK_PENALTY = 2.0       # mu for the slack-variable QUBO
SLACK_CARDINALITY = 2.0   # A for the slack-variable QUBO
# stress instance: each limit targets the smaller of (a) this percentile of what the groups of
# exactly K candidates need and (b) one unit less than what the optimum uses. stage2.stress_limits
# takes the closest pair of limits that leaves something feasible and excludes the optimum.
STRESS_PERCENTILE = 25

# --------------------------------------------------------------------------- stage 3
SHOTS_STAGE3 = 32768                     # shot count for ideal, noisy and mitigated runs
SHOTS_STAGE3_HEADLINE = 2 ** 20          # shot count of the submitted quantum estimate
SHOT_SWEEP = tuple(2 ** k for k in range(10, 21, 2))
SHOTS_COUNTER_NOISY = 1024               # noisy run of the (deep) counter circuit
MC_TRIALS = 1_000_000
MITIGATION_BOOTSTRAP = 500               # bootstrap rounds for the standard error of the mitigated estimate (seed MC_SEED + 1)

# --------------------------------------------------------------------------- noise model
# An explicit, stated noise model in the range of public IBM Eagle calibrations. It is an
# assumption for simulation, not a calibration of any specific device.
NOISE = {
    "one_qubit_depolarizing": 3e-4,      # on sx and x (rz is virtual and error free)
    "two_qubit_depolarizing": 8e-3,      # on ecr
    "readout_p1_given_0": 0.010,         # P(read 1 | prepared 0)
    "readout_p0_given_1": 0.025,         # P(read 0 | prepared 1)
}
BASIS_GATES = ["ecr", "rz", "sx", "x"]
DENSITY_MATRIX_MAX_QUBITS = 11   # noisy runs: exact density matrix up to this width, trajectories above
HEAVY_HEX_DISTANCE = 3    # CouplingMap.from_heavy_hex(3): 19 physical qubits

# --------------------------------------------------------------------------- error propagation
BOOTSTRAP_ROUNDS = 200

# --------------------------------------------------------------------------- beyond the brief (run_extensions.py)
# These settings feed the extension studies only. They never touch the three stage files.
EXT_SEED = 4242                               # Aer seed (seed_simulator) for every sampled extension circuit
EXT_RESULTS = ROOT / "results" / "extensions"

# correlated customers: one-factor Gaussian copula loaded on a small "market" register
CORRELATION_GRID = (0.0, 0.1, 0.2, 0.3, 0.5)  # common-factor correlation rho
FACTOR_QUBITS = 3                             # 2^3 equally likely market scenarios
SHOTS_CORRELATED = 32768
REVENUE_TAIL = 0.25                           # tail level for value at risk and expected shortfall (the worst quarter;
                                              # P(nobody subscribes) is above 5%, so a 5% tail would sit on the floor)

# amplitude estimation of P(at least threshold)
AE_SCHEDULES = (1, 2, 3, 4, 5, 6)             # schedule j uses Grover powers 0, 1, 2, 4, ..., 2^(j-1)
AE_SHOTS_PER_POWER = 64
AE_REPEATS = 300                              # independent repetitions per schedule (blocks of one seeded run per power)

# IBM device snapshots (calibration data shipped with qiskit-ibm-runtime, simulated with Aer)
IBM_SNAPSHOTS = ("FakeTorino", "FakeBrisbane")
SHOTS_IBM = 32768                             # stage 3 circuit and its calibration circuits
SHOTS_IBM_QAOA = 2 ** 17                      # QAOA circuits: exact density matrix, so shots are cheap
IBM_QAOA_LAYERS = (1, 2, 3)

# real device run (scripts/run_ibm_hardware.py; optional, needs an IBM Quantum account)
HARDWARE_RHO = 0.3                            # correlation of the correlated-customers circuit sent to the device
