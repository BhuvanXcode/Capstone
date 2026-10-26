"""QUBO and Ising Hamiltonian formulation for small routing optimization problems.

Formulates an N-customer routing problem as a Quadratic Unconstrained Binary
Optimization (QUBO) problem, and analytically maps it to an Ising Problem
Hamiltonian suitable for genuine gate-level QAOA quantum circuits.

Variable Definition:
    x_{i, p} in {0, 1}
    Indicates whether customer (i + 1) is visited at route position p,
    where i in {0, ..., N - 1} and p in {0, ..., N - 1}.
    Qubit index: q = i * N + p (Total qubits = N^2).
"""

from typing import Dict, List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

try:
    from .data_loader import ProblemInstance
except ImportError:
    from data_loader import ProblemInstance


class RoutingQUBO:
    """Constructs the QUBO matrix and translates it into an Ising Hamiltonian."""

    def __init__(self, instance: ProblemInstance, penalty: float = 40.0):
        """Initialize with problem instance and constraint penalty weight.

        Args:
            instance: ProblemInstance containing customer coordinates and distance matrix.
            penalty: Lagrange multiplier P enforcing customer and position permutation constraints.
        """
        self.instance = instance
        self.n = instance.n_customers
        self.n_qubits = instance.n_qubits
        self.penalty = penalty
        self.dist = instance.distance_matrix

        # Build QUBO matrix and offset
        self.Q, self.offset = self._build_qubo()

        # Derive Ising Hamiltonian coefficients: H_C = sum h_u Z_u + sum J_uv Z_u Z_v + C0 * I
        self.h, self.J, self.C0 = self._qubo_to_ising()

        # Build Qiskit SparsePauliOp representation
        self.hamiltonian = self._build_sparse_pauli_op()

    def var_index(self, customer_idx: int, position_idx: int) -> int:
        """Map (customer_index, route_position) to linear qubit index q."""
        return customer_idx * self.n + position_idx

    def _build_qubo(self) -> Tuple[np.ndarray, float]:
        """Construct the upper-triangular QUBO matrix Q and constant scalar offset.

        Objective:
            Min x^T Q x + offset
            = Distance_Cost(x) + Penalty * [ Row_Constraints(x) + Col_Constraints(x) ]
        """
        n = self.n
        n_vars = self.n_qubits
        Q = np.zeros((n_vars, n_vars), dtype=float)
        offset = 0.0

        # --- 1. ROUTING DISTANCE OBJECTIVE ---
        # Depot (node 0) -> First customer at position 0
        for i in range(n):
            u = self.var_index(i, 0)
            Q[u, u] += self.dist[0, i + 1]

        # Consecutive customers from position p to position p + 1
        for p in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        u = self.var_index(i, p)
                        v = self.var_index(j, p + 1)
                        if u < v:
                            Q[u, v] += self.dist[i + 1, j + 1]
                        else:
                            Q[v, u] += self.dist[i + 1, j + 1]

        # Last customer at position (n - 1) -> Return to Depot (node 0)
        for i in range(n):
            u = self.var_index(i, n - 1)
            Q[u, u] += self.dist[i + 1, 0]

        # --- 2. ROW CONSTRAINTS: Each customer i visited exactly once ---
        # Penalty * sum_i (sum_p x_{i,p} - 1)^2
        # Since x^2 = x: (sum x - 1)^2 = sum x + 2 sum_{p1<p2} x_p1 x_p2 - 2 sum x + 1
        #                              = -sum x + 2 sum_{p1<p2} x_p1 x_p2 + 1
        for i in range(n):
            offset += self.penalty * 1.0
            for p in range(n):
                u = self.var_index(i, p)
                Q[u, u] -= self.penalty * 1.0
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    u = self.var_index(i, p1)
                    v = self.var_index(i, p2)
                    if u > v:
                        u, v = v, u
                    Q[u, v] += self.penalty * 2.0

        # --- 3. COLUMN CONSTRAINTS: Each position p visited by exactly one customer ---
        # Penalty * sum_p (sum_i x_{i,p} - 1)^2
        for p in range(n):
            offset += self.penalty * 1.0
            for i in range(n):
                u = self.var_index(i, p)
                Q[u, u] -= self.penalty * 1.0
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    u = self.var_index(i1, p)
                    v = self.var_index(i2, p)
                    if u > v:
                        u, v = v, u
                    Q[u, v] += self.penalty * 2.0

        return Q, offset

    def _qubo_to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO formulation into Ising spin Hamiltonian coefficients.

        Mapping relation:
            x_u = (I - Z_u) / 2
            where Z_u in {+1, -1} is the Pauli-Z operator for qubit u.
        """
        n_vars = self.n_qubits
        h = np.zeros(n_vars, dtype=float)
        J = np.zeros((n_vars, n_vars), dtype=float)
        C0 = float(self.offset)

        # Linear diagonal terms Q[u, u] * x_u
        for u in range(n_vars):
            C0 += self.Q[u, u] / 2.0
            h[u] -= self.Q[u, u] / 2.0

        # Quadratic cross terms Q[u, v] * x_u * x_v (u < v)
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                coeff = (self.Q[u, v] + self.Q[v, u]) / 4.0
                C0 += coeff
                h[u] -= coeff
                h[v] -= coeff
                J[u, v] += coeff

        return h, J, C0

    def _build_sparse_pauli_op(self) -> SparsePauliOp:
        """Construct Qiskit SparsePauliOp for the Ising Problem Hamiltonian H_C."""
        n_vars = self.n_qubits
        terms = []

        # Constant identity term
        terms.append(("I" * n_vars, self.C0))

        # Single-qubit Pauli-Z terms
        for u in range(n_vars):
            if abs(self.h[u]) > 1e-8:
                # Qiskit Pauli string convention: qubit 0 is rightmost (index n_vars - 1)
                label = ["I"] * n_vars
                label[n_vars - 1 - u] = "Z"
                terms.append(("".join(label), self.h[u]))

        # Two-qubit ZZ interaction terms
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if abs(self.J[u, v]) > 1e-8:
                    label = ["I"] * n_vars
                    label[n_vars - 1 - u] = "Z"
                    label[n_vars - 1 - v] = "Z"
                    terms.append(("".join(label), self.J[u, v]))

        return SparsePauliOp.from_list(terms)

    def evaluate_bitstring_qubo(self, x: np.ndarray) -> float:
        """Compute the scalar QUBO objective energy for a binary vector x."""
        return float(x @ self.Q @ x + self.offset)

    def evaluate_breakdown(self, x: np.ndarray) -> Dict[str, float]:
        """Compute detailed breakdown: distance cost, customer penalty, position penalty."""
        matrix = x.reshape((self.n, self.n))
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)

        cust_penalty = float(self.penalty * sum((s - 1.0) ** 2 for s in row_sums))
        pos_penalty = float(self.penalty * sum((s - 1.0) ** 2 for s in col_sums))

        total_cost = self.evaluate_bitstring_qubo(x)
        pure_dist = total_cost - cust_penalty - pos_penalty

        return {
            "total_qubo_energy": total_cost,
            "routing_distance": pure_dist,
            "customer_penalty": cust_penalty,
            "position_penalty": pos_penalty,
            "is_permutation_feasible": bool(np.all(row_sums == 1) and np.all(col_sums == 1)),
        }
