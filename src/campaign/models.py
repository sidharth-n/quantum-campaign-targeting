"""Stage 1 classifiers: classical baselines, a quantum kernel classifier and a fixed hybrid.

Every model is a scikit-learn estimator whose fit() sees training rows only.

Quantum part
------------
The quantum kernel is the fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2, where |phi(x)> is
the state prepared by Qiskit's ZZ feature map on `n_qubits` qubits. The classical data reach
the circuit as rotation angles: preprocessing -> reduction to n_qubits components -> standardise
-> multiply by a bandwidth. The reduction is PCA (unsupervised) or PLS (supervised: directions
of largest covariance with the label, fitted on the training rows of the current fit only). A
small bandwidth keeps the kernel from concentrating (all off-diagonal entries going to the same
value), which is the known failure mode of this feature map.

For speed the feature-map states are computed with a short NumPy routine (`feature_map_states`)
that applies the same gates as the Qiskit circuit. `tests/test_quantum_kernel.py` checks that it
matches `qiskit.quantum_info.Statevector` on the real circuit amplitude by amplitude, and
`kernel_circuit` builds the overlap circuit U(x')^dagger U(x) whose all-zeros probability is the
kernel entry, so entries can also be estimated from shots.

Classical part
--------------
The kernel matrix goes into a classical SVM. Probabilities come from Platt scaling fitted on
out-of-fold SVM scores of the training rows; the whole chain is refitted inside every
calibration fold (see KernelSVC).

Control
-------
`KernelSVC(kernel="rbf")` is the same pipeline with a Gaussian kernel exp(-|a - b|^2) on the same
angles. It isolates what the quantum feature map contributes beyond the classical steps around it.
"""
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from scipy.linalg import hadamard
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import HYBRID_WEIGHT, KERNEL_QUBITS, PLATT_FOLDS, SEED
from .features import engineer, make_preprocessor

ENTANGLEMENT = "linear"


# --------------------------------------------------------------------------- feature map
def feature_map_circuit(n_qubits: int, reps: int) -> QuantumCircuit:
    """The parameterised ZZ feature map (parameters x[0] .. x[n-1])."""
    return zz_feature_map(n_qubits, reps=reps, entanglement=ENTANGLEMENT)


def feature_map_states(angles: np.ndarray, reps: int) -> np.ndarray:
    """Statevectors of the ZZ feature map for each row of `angles`, in Qiskit's basis order.

    One repetition is H on every qubit, P(2 x_i) on qubit i, and for each neighbouring pair
    (i, i+1) the block CX, P(2 (pi - x_i)(pi - x_{i+1})), CX. The pair block multiplies a basis
    state by exp(i * 2 (pi - x_i)(pi - x_{i+1})) when bits i and i+1 differ, so every repetition
    is a Hadamard transform followed by a diagonal phase. Basis index b = sum_i b_i 2^i
    (qubit 0 is the least significant bit, as in Qiskit).
    """
    angles = np.atleast_2d(np.asarray(angles, dtype=float))
    n = angles.shape[1]
    dim = 2 ** n
    bits = (np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1          # (dim, n)
    phase = 2.0 * angles @ bits.T                                           # single-qubit P gates
    for i in range(n - 1):
        differ = bits[:, i] ^ bits[:, i + 1]
        phase += np.outer(2.0 * (np.pi - angles[:, i]) * (np.pi - angles[:, i + 1]), differ)
    diagonal = np.exp(1j * phase)                                           # (rows, dim)
    walsh = hadamard(dim) / np.sqrt(dim)
    states = np.zeros((angles.shape[0], dim), dtype=complex)
    states[:, 0] = 1.0
    for _ in range(reps):
        states = (states @ walsh) * diagonal
    return states


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    return np.abs(states_a.conj() @ states_b.T) ** 2


def kernel_circuit(x_a: np.ndarray, x_b: np.ndarray, reps: int) -> QuantumCircuit:
    """Overlap circuit: P(all zeros) = |<phi(x_b)|phi(x_a)>|^2."""
    n = len(x_a)
    fmap = feature_map_circuit(n, reps)
    params = list(fmap.parameters)
    forward = fmap.assign_parameters(dict(zip(params, map(float, x_a))))
    backward = fmap.assign_parameters(dict(zip(params, map(float, x_b)))).inverse()
    circuit = QuantumCircuit(n)
    circuit.compose(forward, inplace=True)
    circuit.compose(backward, inplace=True)
    circuit.measure_all()
    return circuit


# --------------------------------------------------------------------------- estimators
class _ProbabilityModel(ClassifierMixin, BaseEstimator):
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class Classical(_ProbabilityModel):
    """A classical scikit-learn pipeline behind the same row-wise feature engineering."""

    def __init__(self, pipe=None):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe_ = clone(self.pipe).fit(engineer(X), y)
        self.classes_ = self.pipe_.classes_
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(engineer(X))


def logistic(C=0.1) -> Classical:
    """Classical baseline: L2 logistic regression, no class weighting (keeps probabilities calibrated)."""
    return Classical(make_pipeline(make_preprocessor(), LogisticRegression(C=C, max_iter=5000, random_state=SEED)))


def boosting() -> Classical:
    """Non-linear classical reference with fixed, conservative settings (not tuned)."""
    return Classical(
        make_pipeline(
            make_preprocessor(),
            HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10,
                l2_regularization=1.0, random_state=SEED,
            ),
        )
    )


def rbf_kernel(angles_a: np.ndarray, angles_b: np.ndarray) -> np.ndarray:
    squared = ((angles_a[:, None, :] - angles_b[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-squared)


class _KernelChain:
    """One complete fit: preprocessing -> reduction -> scaling -> kernel -> SVM. Every step sees
    only the rows it is given."""

    def __init__(self, kernel, reducer, n_qubits, reps, bandwidth, C):
        self.kernel, self.reducer, self.n_qubits, self.reps = kernel, reducer, n_qubits, reps
        self.bandwidth, self.C = bandwidth, C

    def fit(self, X, y):
        E = engineer(X)
        self.pre = make_preprocessor().fit(E)
        Z = self.pre.transform(E)
        if self.reducer == "pca":
            self.reduce = PCA(n_components=self.n_qubits, svd_solver="full").fit(Z)
        elif self.reducer == "pls":
            self.reduce = PLSRegression(n_components=self.n_qubits, scale=False).fit(Z, np.asarray(y, float))
        else:
            raise ValueError(f"unknown reducer {self.reducer!r}")
        self.scale = StandardScaler().fit(self.reduce.transform(Z))
        self.train_angles = self.angles(X)
        self.svc = SVC(kernel="precomputed", C=self.C).fit(self.gram(self.train_angles, self.train_angles), y)
        return self

    def angles(self, X):
        """Rotation angles fed to the feature map."""
        return self.bandwidth * self.scale.transform(self.reduce.transform(self.pre.transform(engineer(X))))

    def gram(self, angles_a, angles_b):
        if self.kernel == "quantum":
            return fidelity_kernel(feature_map_states(angles_a, self.reps), feature_map_states(angles_b, self.reps))
        if self.kernel == "rbf":
            return rbf_kernel(angles_a, angles_b)
        raise ValueError(f"unknown kernel {self.kernel!r}")

    def decision(self, X):
        return self.svc.decision_function(self.gram(self.angles(X), self.train_angles))


class KernelSVC(_ProbabilityModel):
    """Kernel -> SVM -> Platt scaling. kernel="quantum" is the fidelity kernel, "rbf" the control.

    Platt scaling needs SVM scores for rows the SVM has not seen. They come from PLATT_FOLDS
    inner folds, and the whole chain (scaler, one-hot levels, PCA or PLS, kernel, SVM) is refitted
    on the other folds each time, so no step that produced a row's score has seen that row or
    its label. The sigmoid is a logistic regression of the labels on those out-of-fold scores.
    The final SVM is then fitted on all rows given to fit().
    """

    def __init__(self, kernel="quantum", reducer="pls", n_qubits=KERNEL_QUBITS, reps=2, bandwidth=0.25, C=1.0):
        self.kernel = kernel
        self.reducer = reducer
        self.n_qubits = n_qubits
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C

    def _chain(self):
        return _KernelChain(self.kernel, self.reducer, self.n_qubits, self.reps, self.bandwidth, self.C)

    def calibration_folds(self, X, y):
        """Identical rows always share a fold, so a customer that appears twice (as in a bootstrap
        resample) can never sit on both sides of a calibration split."""
        groups = pd.util.hash_pandas_object(pd.DataFrame(X).reset_index(drop=True), index=False).to_numpy()
        splitter = StratifiedGroupKFold(PLATT_FOLDS, shuffle=True, random_state=SEED)
        return list(splitter.split(np.zeros(len(y)), y, groups))

    def out_of_fold_scores(self, X, y, folds):
        scores = np.full(len(y), np.nan)
        for fit_rows, held_rows in folds:
            scores[held_rows] = self._chain().fit(X.iloc[fit_rows], y[fit_rows]).decision(X.iloc[held_rows])
        return scores

    def fit(self, X, y):
        y = np.asarray(y)
        scores = self.out_of_fold_scores(X, y, self.calibration_folds(X, y))
        self.platt_ = LogisticRegression(C=1e6, max_iter=5000).fit(scores.reshape(-1, 1), y)
        self.chain_ = self._chain().fit(X, y)
        self.train_angles_ = self.chain_.train_angles
        self.classes_ = self.platt_.classes_
        return self

    def angles(self, X):
        return self.chain_.angles(X)

    def predict_proba(self, X):
        return self.platt_.predict_proba(self.chain_.decision(X).reshape(-1, 1))


def quantum_kernel_svm(**params) -> KernelSVC:
    return KernelSVC(kernel="quantum", **params)


def rbf_kernel_svm(**params) -> KernelSVC:
    return KernelSVC(kernel="rbf", **params)


class Hybrid(_ProbabilityModel):
    """Fixed-weight average of the classical baseline and the quantum kernel classifier."""

    def __init__(self, classical=None, quantum=None, weight=HYBRID_WEIGHT):
        self.classical = classical
        self.quantum = quantum
        self.weight = weight

    def fit(self, X, y):
        self.classical_ = clone(self.classical).fit(X, y)
        self.quantum_ = clone(self.quantum).fit(X, y)
        self.classes_ = self.classical_.classes_
        return self

    def predict_proba(self, X):
        p = (self.weight * self.classical_.predict_proba(X)[:, 1]
             + (1.0 - self.weight) * self.quantum_.predict_proba(X)[:, 1])
        return np.column_stack([1.0 - p, p])
