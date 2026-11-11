"""Stage 2 as a binary optimisation problem: exact solution, QUBO encodings and the Ising map.

Decision variable x_i = 1 when candidate i is contacted. Candidate i is the customer with
stage 1 rank i + 1, and in every circuit qubit i carries x_i.

    maximise    sum_i v_i x_i            v_i = p_i * expected_deposit_value_i - contact_cost_i
    subject to  sum_i x_i = K
                sum_i contact_cost_i x_i   <= budget
                sum_i capacity_units_i x_i <= capacity

Two QUBO encodings (both written as minimisation):

  unbalanced  no auxiliary qubits. Each inequality g(x) = limit - sum_i w_i x_i >= 0 adds
              -lambda_1 g(x) + lambda_2 g(x)^2 (Montanez-Barrera et al., 2023). The penalty is
              smallest at slack g* = lambda_1 / (2 lambda_2) and grows quadratically for violations.
              It is an approximation: it also changes the energies of feasible states, so the
              weights must be validated.
  slack       exact. Each inequality becomes the equality sum_i w_i x_i + s = limit with an integer
              slack s in [0, U] written in binary on auxiliary qubits, penalised by mu (...)^2.

Both share the cardinality penalty A (sum_i x_i - K)^2.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Problem:
    ids: tuple            # customer ids in stage 1 rank order (index = qubit)
    probability: np.ndarray
    deposit: np.ndarray
    cost: np.ndarray
    capacity: np.ndarray
    value: np.ndarray     # expected net value
    k: int
    budget: float
    capacity_limit: float

    @property
    def n(self) -> int:
        return len(self.ids)

    def with_limits(self, budget: float, capacity_limit: float) -> "Problem":
        return Problem(self.ids, self.probability, self.deposit, self.cost, self.capacity,
                       self.value, self.k, budget, capacity_limit)


def make_problem(candidates: pd.DataFrame, k: int, budget: float, capacity_limit: float) -> Problem:
    """`candidates` is in stage 1 rank order with the stage 1 probability and the stage 2 inputs."""
    p = candidates["predicted_probability"].to_numpy(float)
    deposit = candidates["expected_deposit_value"].to_numpy(float)
    cost = candidates["contact_cost"].to_numpy(float)
    return Problem(
        ids=tuple(candidates["customer_id"]),
        probability=p,
        deposit=deposit,
        cost=cost,
        capacity=candidates["capacity_units"].to_numpy(float),
        value=p * deposit - cost,
        k=int(k),
        budget=float(budget),
        capacity_limit=float(capacity_limit),
    )


# --------------------------------------------------------------------------- bit conventions
def all_bitstrings(n: int) -> np.ndarray:
    """Row b holds x_0 .. x_{n-1} with x_i = bit i of the integer b (qubit 0 = least significant)."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def bits_to_index(x) -> int:
    return int(sum(int(bit) << i for i, bit in enumerate(x)))


def qiskit_key(x) -> str:
    """Qiskit prints qubit 0 as the rightmost character."""
    return "".join(str(int(bit)) for bit in reversed(list(x)))


def key_to_bits(key: str) -> np.ndarray:
    """Inverse of qiskit_key: element i of the result is the value measured on qubit i."""
    return np.array([int(ch) for ch in reversed(key.replace(" ", ""))], dtype=np.int8)


# --------------------------------------------------------------------------- exact solution
def evaluate(problem: Problem, X: np.ndarray) -> dict:
    """Totals and feasibility for each row of X (decision bits of the n candidates)."""
    X = np.atleast_2d(X)[:, : problem.n].astype(float)
    count = X.sum(axis=1)
    cost = X @ problem.cost
    capacity = X @ problem.capacity
    feasible = (count == problem.k) & (cost <= problem.budget + 1e-9) & (capacity <= problem.capacity_limit + 1e-9)
    return {"count": count, "cost": cost, "capacity": capacity, "value": X @ problem.value, "feasible": feasible}


def brute_force(problem: Problem) -> dict:
    """Exact optimum by checking every bitstring. Ties go to the smaller basis index."""
    X = all_bitstrings(problem.n)
    ev = evaluate(problem, X)
    right_size = ev["count"] == problem.k
    feasible = ev["feasible"]
    if not feasible.any():
        raise ValueError("no feasible selection exists")
    masked = np.where(feasible, ev["value"], -np.inf)
    best = int(np.argmax(masked))                      # argmax returns the first maximum
    order = np.argsort(-masked, kind="stable")
    x = X[best]
    return {
        "x": x,
        "index": best,
        "value": float(ev["value"][best]),
        "cost": float(ev["cost"][best]),
        "capacity": float(ev["capacity"][best]),
        "ids": [problem.ids[i] for i in np.flatnonzero(x)],
        "n_subsets_of_size_k": int(right_size.sum()),
        "n_feasible": int(feasible.sum()),
        "feasible_values_sorted": masked[order][: int(feasible.sum())],
        "feasible_order": order[: int(feasible.sum())],
        "second_best_value": float(masked[order][1]) if feasible.sum() > 1 else float("nan"),
        "max_cost_any_k_subset": float(ev["cost"][right_size].max()),
        "max_capacity_any_k_subset": float(ev["capacity"][right_size].max()),
    }


# --------------------------------------------------------------------------- QUBO container
@dataclass
class Qubo:
    """E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + offset, with Q upper triangular."""
    Q: np.ndarray
    offset: float
    n_decision: int
    labels: tuple          # one label per qubit

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energies(self) -> np.ndarray:
        """Energy of every basis state, indexed like all_bitstrings."""
        X = all_bitstrings(self.n).astype(float)
        return np.einsum("bi,ij,bj->b", X, self.Q, X, optimize=True) + self.offset

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)


class _Builder:
    def __init__(self, n: int):
        self.Q = np.zeros((n, n))
        self.offset = 0.0

    def linear(self, coefficients, weight: float = 1.0):
        """weight * sum_i a_i x_i"""
        for i, a in enumerate(coefficients):
            self.Q[i, i] += weight * a

    def squared(self, coefficients, constant: float, weight: float):
        """weight * (sum_i a_i x_i + constant)^2, using x_i^2 = x_i."""
        a = np.asarray(coefficients, dtype=float)
        for i in range(len(a)):
            self.Q[i, i] += weight * (a[i] ** 2 + 2.0 * constant * a[i])
            for j in range(i + 1, len(a)):
                self.Q[i, j] += weight * 2.0 * a[i] * a[j]
        self.offset += weight * constant ** 2

    def constant(self, value: float):
        self.offset += value


def _pad(values, n):
    out = np.zeros(n)
    out[: len(values)] = values
    return out


def can_bind(problem: Problem) -> tuple:
    """(budget, capacity): can any selection of exactly K candidates exceed the limit?
    If the K largest weights already fit, the constraint is implied by the cardinality constraint."""
    def largest(weights):
        return float(np.sort(weights)[-problem.k:].sum())
    return largest(problem.cost) > problem.budget, largest(problem.capacity) > problem.capacity_limit


def unbalanced_qubo(problem: Problem, A: float, lambda_1: float, lambda_2: float, active=(True, True)) -> Qubo:
    """n qubits, no auxiliaries. A, lambda_1, lambda_2 are absolute weights. `active` switches the
    budget and capacity penalties on or off; A = 0 leaves the cardinality constraint to the circuit."""
    n = problem.n
    b = _Builder(n)
    b.linear(-problem.value)
    if A:
        b.squared(np.ones(n), -problem.k, A)
    for (weights, limit), on in zip(((problem.cost, problem.budget), (problem.capacity, problem.capacity_limit)), active):
        if not on:
            continue
        # g(x) = limit - sum_i w_i x_i ; add -lambda_1 g + lambda_2 g^2
        b.linear(weights, lambda_1)
        b.constant(-lambda_1 * limit)
        b.squared(-weights, limit, lambda_2)
    return Qubo(b.Q, b.offset, n, tuple(f"x{i}" for i in range(n)))


def slack_coefficients(upper: int) -> list:
    """Binary expansion that represents exactly the integers 0 .. upper (last coefficient truncated)."""
    upper = int(upper)
    if upper <= 0:
        return []
    m = int(np.floor(np.log2(upper))) + 1
    coefficients = [2 ** k for k in range(m - 1)]
    coefficients.append(upper - sum(coefficients))
    return coefficients


def slack_qubo(problem: Problem, A: float, mu: float) -> Qubo:
    """Exact encoding with binary slack qubits appended after the n decision qubits.

    The slack range is U = limit - (sum of the K smallest weights): the largest slack that a
    selection of exactly K customers can need.
    """
    n = problem.n
    blocks = []
    for name, weights, limit in (("b", problem.cost, problem.budget), ("h", problem.capacity, problem.capacity_limit)):
        upper = limit - np.sort(weights)[: problem.k].sum()
        if upper < 0:
            raise ValueError(f"constraint {name} cannot be met by any selection of {problem.k}")
        blocks.append((name, weights, limit, slack_coefficients(int(round(upper)))))
    total = n + sum(len(c) for *_, c in blocks)
    b = _Builder(total)
    b.linear(_pad(-problem.value, total))
    b.squared(_pad(np.ones(n), total), -problem.k, A)
    labels = [f"x{i}" for i in range(n)]
    start = n
    for name, weights, limit, coefficients in blocks:
        row = _pad(weights, total)
        row[start: start + len(coefficients)] = coefficients
        b.squared(row, -limit, mu)
        labels += [f"s_{name}{k}" for k in range(len(coefficients))]
        start += len(coefficients)
    return Qubo(b.Q, b.offset, n, tuple(labels))


# --------------------------------------------------------------------------- Ising map
def to_ising(qubo: Qubo):
    """Substitute x_i = (1 - z_i) / 2. Returns (h, J, constant) with
    E = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + constant and z_i = +1 for bit 0, -1 for bit 1."""
    Q = qubo.Q
    n = qubo.n
    upper = np.triu(Q, 1)
    pair_sum = upper + upper.T
    h = -np.diag(Q) / 2.0 - pair_sum.sum(axis=1) / 4.0
    J = upper / 4.0
    constant = qubo.offset + np.diag(Q).sum() / 2.0 + upper.sum() / 4.0
    return h, J, float(constant)


def ising_energies(h: np.ndarray, J: np.ndarray, constant: float) -> np.ndarray:
    Z = 1.0 - 2.0 * all_bitstrings(len(h)).astype(float)
    return Z @ h + np.einsum("bi,ij,bj->b", Z, J, Z, optimize=True) + constant


# --------------------------------------------------------------------------- validation and tuning
def check_ground_state(problem: Problem, qubo: Qubo, exact: dict, energies: np.ndarray = None,
                       within_cardinality: bool = False) -> dict:
    """Does the QUBO minimum decode to the brute-force optimum, and how well separated is it?
    With within_cardinality=True only states with exactly K ones are considered (the subspace a
    constraint-preserving circuit stays in)."""
    E = qubo.energies() if energies is None else energies
    index = np.arange(len(E))
    if within_cardinality:
        keep = all_bitstrings(qubo.n).sum(axis=1) == problem.k
        E, index = E[keep], index[keep]
    ground_position = int(np.argmin(E))
    ground = int(index[ground_position])
    decision = ground & (2 ** problem.n - 1)           # decision bits are the low n bits
    # lowest energy whose decision bits differ from the optimum
    other = E[(index & (2 ** problem.n - 1)) != exact["index"]]
    gap = float(other.min() - E[ground_position])
    span = float(E.max() - E[ground_position])
    optimum_energy = E[(index & (2 ** problem.n - 1)) == exact["index"]].min()
    return {
        "valid": bool(decision == exact["index"] and gap > 1e-9),
        "ground_index": ground,
        "ground_energy": float(E[ground_position]),
        "gap": gap,
        "relative_gap": gap / span if span > 0 else 0.0,
        "energy_max": float(E.max()),
        # how many other selections sit below the true optimum (0 when the encoding is valid)
        "selections_below_optimum": int(len(np.unique((index & (2 ** problem.n - 1))[E < optimum_energy - 1e-12]))),
    }


def tune_unbalanced(problem: Problem, exact: dict, cardinality_grid, l2_grid, slack_grid,
                    within_cardinality: bool = False, active=(True, True)) -> dict:
    """Penalty rule for the unbalanced QUBO.

    Grid values are in units of v_max = max_i |v_i|. For each (A, lambda_2, g*) with
    lambda_1 = 2 lambda_2 g*: the setting is valid when the QUBO ground state over all 2^n
    bitstrings is the brute-force optimum. Among valid settings the largest relative gap
    (gap to the best other selection divided by the full energy span) wins, because QAOA sees
    the cost function only up to scale. Ties go to the earlier grid entry. If no setting on the
    grid is valid (this can happen: the unbalanced penalty is an approximation), the setting
    with the fewest selections below the optimum is returned with "valid": False.

    within_cardinality=True applies the same rule inside the exactly-K subspace (used with
    cardinality_grid=(0,) for the constraint-preserving circuit); `active` drops the penalty of
    a constraint that cannot bind.
    """
    v_max = float(np.abs(problem.value).max())
    rows = []
    for a in cardinality_grid:
        for l2 in l2_grid:
            for g_star in slack_grid:
                A, lambda_2 = a * v_max, l2 * v_max
                lambda_1 = 2.0 * lambda_2 * g_star
                model = unbalanced_qubo(problem, A, lambda_1, lambda_2, active)
                check = check_ground_state(problem, model, exact, within_cardinality=within_cardinality)
                rows.append({
                    "A_over_vmax": a, "lambda2_over_vmax": l2, "preferred_slack": g_star,
                    "A": A, "lambda_1": lambda_1, "lambda_2": lambda_2,
                    "valid": check["valid"], "gap": check["gap"], "relative_gap": check["relative_gap"],
                    "selections_below_optimum": check["selections_below_optimum"],
                })
    table = pd.DataFrame(rows)
    valid = table[table["valid"]]
    if valid.empty:
        # no setting makes the optimum the ground state: keep the one that leaves the fewest
        # selections below it (ties: earlier grid entry) and say so through "valid": False
        best = table.loc[table["selections_below_optimum"].idxmin()]
    else:
        best = valid.loc[valid["relative_gap"].idxmax()]
    return {
        "valid": bool(best["valid"]),
        "selections_below_optimum": int(best["selections_below_optimum"]),
        "v_max": v_max,
        "A": float(best["A"]), "lambda_1": float(best["lambda_1"]), "lambda_2": float(best["lambda_2"]),
        "A_over_vmax": float(best["A_over_vmax"]), "lambda2_over_vmax": float(best["lambda2_over_vmax"]),
        "preferred_slack": float(best["preferred_slack"]),
        "relative_gap": float(best["relative_gap"]), "gap": float(best["gap"]),
        "n_valid": int(len(valid)), "n_grid": int(len(table)),
        "table": table,
    }
