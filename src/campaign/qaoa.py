"""QAOA for the stage 2 QUBO: circuit, exact emulator for training, and sampling on Aer.

Cost Hamiltonian (after the Ising map, divided by `scale` so angles are of order one):

    H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

One QAOA layer is exp(-i beta sum_i X_i) exp(-i gamma H_C), applied to |+>^n. In gates:
RZ(2 gamma h_i) on qubit i, RZZ(2 gamma J_ij) on each pair, then RX(2 beta) on every qubit.

Training uses `emulate`, a NumPy routine that applies the same unitaries to the full state
vector (the cost layer is diagonal, the mixer is one 2x2 rotation per qubit). It is checked
against `qiskit.quantum_info.Statevector` on the real circuit in tests/test_qaoa.py. The trained
circuit is then run on Aer with a finite number of shots, and that sample gives the selection.

The training objective is CVaR_alpha (Barkoutsos et al., 2020): the mean energy of the best
alpha fraction of the output distribution, which rewards putting weight on good bitstrings
rather than lowering the average.
"""
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

from .qubo import Qubo, to_ising


@dataclass
class Hamiltonian:
    h: np.ndarray            # scaled fields
    J: np.ndarray            # scaled couplings (upper triangular)
    scale: float             # original = scaled * scale
    constant: float          # original constant term
    diagonal: np.ndarray     # scaled H_C on every basis state (no constant)
    energies: np.ndarray     # original QUBO energy of every basis state

    @property
    def n(self) -> int:
        return len(self.h)


def hamiltonian(qubo: Qubo) -> Hamiltonian:
    h, J, constant = to_ising(qubo)
    scale = float(max(np.abs(h).max(), np.abs(J).max()))
    energies = qubo.energies()
    return Hamiltonian(h / scale, J / scale, scale, constant, (energies - constant) / scale, energies)


# --------------------------------------------------------------------------- circuit
def qaoa_circuit(ham: Hamiltonian, layers: int, measure: bool = True) -> QuantumCircuit:
    """Parameterised QAOA circuit. Parameters: gamma[0..p-1] then beta[0..p-1]."""
    n = ham.n
    gamma = ParameterVector("gamma", layers)
    beta = ParameterVector("beta", layers)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(layers):
        for i in range(n):
            if ham.h[i] != 0.0:
                qc.rz(2.0 * gamma[layer] * float(ham.h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if ham.J[i, j] != 0.0:
                    qc.rzz(2.0 * gamma[layer] * float(ham.J[i, j]), i, j)
        for i in range(n):
            qc.rx(2.0 * beta[layer], i)
    if measure:
        qc.measure_all()
    return qc


def bind(circuit: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
    """Bind [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}] by parameter name, not by position."""
    layers = len(params) // 2
    values = {}
    for parameter in circuit.parameters:
        index = int(parameter.name.split("[")[1].rstrip("]"))
        values[parameter] = float(params[index] if parameter.name.startswith("gamma") else params[layers + index])
    return circuit.assign_parameters(values)


# --------------------------------------------------------------------------- emulator
def emulate(diagonal: np.ndarray, params: np.ndarray) -> np.ndarray:
    """State after the QAOA circuit, basis index b = sum_i x_i 2^i (Qiskit's order)."""
    dim = len(diagonal)
    n = int(round(np.log2(dim)))
    layers = len(params) // 2
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    for layer in range(layers):
        gamma, beta = params[layer], params[layers + layer]
        state = state * np.exp(-1j * gamma * diagonal)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(n):
            view = state.reshape(-1, 2, 2 ** q)          # middle axis is bit q
            state = (c * view + s * view[:, ::-1, :]).reshape(dim)
    return state


def cvar(probabilities: np.ndarray, energies: np.ndarray, alpha: float, order: np.ndarray = None) -> float:
    """Mean energy of the lowest-energy alpha fraction of the distribution (alpha = 1: plain mean)."""
    order = np.argsort(energies, kind="stable") if order is None else order
    p = probabilities[order]
    before = np.cumsum(p) - p
    weight = np.clip(alpha - before, 0.0, p)
    return float(weight @ energies[order] / alpha)


# --------------------------------------------------------------------------- training
def start_points(layers: int, restarts: int, seed: int) -> list:
    """One linear-ramp start (annealing-like schedule) followed by seeded random starts."""
    steps = (np.arange(layers) + 0.5) / layers
    points = [np.concatenate([0.75 * steps, 0.75 * (1.0 - steps)])]
    rng = np.random.default_rng(seed)
    for _ in range(restarts - 1):
        points.append(np.concatenate([rng.uniform(0.0, 1.0, layers), rng.uniform(0.0, np.pi / 2, layers)]))
    return points


def interpolate(params: np.ndarray, layers: int) -> np.ndarray:
    """Stretch a trained schedule of depth p - 1 to depth p (the INTERP start of Zhou et al., 2020)."""
    previous = len(params) // 2
    if previous == 1:
        return np.concatenate([np.repeat(params[:1], layers), np.repeat(params[1:], layers)])
    old = (np.arange(previous) + 0.5) / previous
    new = (np.arange(layers) + 0.5) / layers
    return np.concatenate([np.interp(new, old, params[:previous]), np.interp(new, old, params[previous:])])


def train(ham: Hamiltonian, layers: int, alpha: float, restarts: int, maxiter: int, seed: int,
          previous: np.ndarray = None, emulator=None) -> dict:
    """COBYLA from each start point on the exact CVaR objective. The lowest final objective wins
    (ties: earlier start). `previous` adds a start interpolated from the depth below. `emulator`
    maps parameters to the output state (default: the penalty QAOA of `emulate`)."""
    order = np.argsort(ham.diagonal, kind="stable")
    if emulator is None:
        def emulator(params):
            return emulate(ham.diagonal, params)

    def objective(params):
        probabilities = np.abs(emulator(params)) ** 2
        return cvar(probabilities, ham.diagonal, alpha, order)

    starts = start_points(layers, restarts, seed)
    if previous is not None:
        starts.append(interpolate(np.asarray(previous, dtype=float), layers))
    runs = []
    for index, start in enumerate(starts):
        result = minimize(objective, start, method="COBYLA", options={"rhobeg": 0.2, "maxiter": maxiter, "tol": 1e-6})
        runs.append({
            "start_index": index,
            "start_kind": "linear ramp" if index == 0 else ("interpolated" if index == restarts else "random"),
            "start": [float(v) for v in start],
            "params": [float(v) for v in result.x],
            "objective": float(result.fun),
            "evaluations": int(result.nfev),
        })
    best = min(runs, key=lambda r: (r["objective"], r["start_index"]))
    return {"params": np.array(best["params"]), "objective": best["objective"],
            "best_start": best["start_index"], "best_start_kind": best["start_kind"], "runs": runs}


def exact_distribution(ham: Hamiltonian, params: np.ndarray) -> np.ndarray:
    return np.abs(emulate(ham.diagonal, params)) ** 2


# =========================================================================== constraint-preserving variant
# Quantum alternating operator ansatz (Hadfield et al., 2019): start in a uniform superposition
# of all bitstrings with exactly K ones (a Dicke state) and mix with XY rotations, which only
# swap a 1 and a 0 between two qubits. The number of ones is conserved, so "exactly K customers"
# holds for every measured bitstring of the ideal circuit and needs no penalty term.

def dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """Deterministic Dicke state preparation (Bartschi and Eidenbenz, 2019), without ancillas.

    For k > n/2 the state with n - k ones is prepared and every qubit is flipped, which needs
    fewer gates and gives the same state.
    """
    if k > n // 2:
        qc = dicke_circuit(n, n - k)
        qc.x(range(n))
        return qc
    from qiskit.circuit.library import RYGate

    def split_and_cyclic_shift(circuit, m, l_max):
        # acts on qubits m - l_max - 1 .. m - 1
        circuit.cx(m - 2, m - 1)
        circuit.cry(2.0 * np.arccos(np.sqrt(1.0 / m)), m - 1, m - 2)
        circuit.cx(m - 2, m - 1)
        for l in range(2, l_max + 1):
            circuit.cx(m - l - 1, m - 1)
            circuit.append(RYGate(2.0 * np.arccos(np.sqrt(l / m))).control(2), [m - 1, m - l, m - l - 1])
            circuit.cx(m - l - 1, m - 1)

    qc = QuantumCircuit(n)
    if k == 0:
        return qc
    qc.x(range(n - k, n))
    for m in range(n, k, -1):
        split_and_cyclic_shift(qc, m, k)
    for m in range(k, 1, -1):
        split_and_cyclic_shift(qc, m, m - 1)
    return qc


def dicke_state(n: int, k: int) -> np.ndarray:
    """Equal superposition of all basis states with exactly k ones."""
    ones = np.array([bin(b).count("1") for b in range(2 ** n)])
    state = (ones == k).astype(complex)
    return state / np.linalg.norm(state)


def xy_pairs(n: int) -> list:
    """Nearest-neighbour pairs on a line: even bonds first, then odd bonds (two layers of gates)."""
    return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]


def xy_circuit(ham: Hamiltonian, layers: int, k: int, measure: bool = True) -> QuantumCircuit:
    """Dicke state, then per layer the cost phase and XY rotations exp(-i beta (XX + YY) / 2)."""
    from qiskit.circuit.library import XXPlusYYGate

    n = ham.n
    gamma = ParameterVector("gamma", layers)
    beta = ParameterVector("beta", layers)
    qc = QuantumCircuit(n)
    qc.compose(dicke_circuit(n, k), inplace=True)
    for layer in range(layers):
        for i in range(n):
            if ham.h[i] != 0.0:
                qc.rz(2.0 * gamma[layer] * float(ham.h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if ham.J[i, j] != 0.0:
                    qc.rzz(2.0 * gamma[layer] * float(ham.J[i, j]), i, j)
        for i, j in xy_pairs(n):
            qc.append(XXPlusYYGate(2.0 * beta[layer]), [i, j])
    if measure:
        qc.measure_all()
    return qc


def emulate_xy(diagonal: np.ndarray, params: np.ndarray, k: int) -> np.ndarray:
    """State after the constraint-preserving circuit (same basis order as `emulate`)."""
    dim = len(diagonal)
    n = int(round(np.log2(dim)))
    layers = len(params) // 2
    index = np.arange(dim)
    swaps = []
    for i, j in xy_pairs(n):
        a = index[((index >> i) & 1 == 1) & ((index >> j) & 1 == 0)]
        swaps.append((a, a ^ (1 << i) ^ (1 << j)))
    state = dicke_state(n, k)
    for layer in range(layers):
        gamma, beta = params[layer], params[layers + layer]
        state = state * np.exp(-1j * gamma * diagonal)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for a, b in swaps:
            amp_a, amp_b = state[a].copy(), state[b].copy()
            state[a] = c * amp_a + s * amp_b
            state[b] = s * amp_a + c * amp_b
    return state
