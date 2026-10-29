"""QUBO and Ising Hamiltonian formulation for routing and subroute optimization.

Formulates routing and subroute re-sequencing problems as Quadratic Unconstrained
Binary Optimization (QUBO) problems and analytically maps them to Ising Problem
Hamiltonians suitable for genuine gate-level QAOA quantum circuits.

Variable Definition:
    x_{i, p} in {0, 1}
    Indicates whether customer i (from the subproblem customer set) is visited at
    relative position p, where i in {0, ..., N - 1} and p in {0, ..., N - 1}.
    Linear qubit index: q = i * N + p (Total qubits = N^2).

Supports:
1. Standard closed depot tours: 0 -> pi(1..N) -> 0.
2. Intermediate subroute windows with fixed endpoints: u -> pi(S) -> w.
3. Optional soft time-window arrival penalties.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from qiskit.quantum_info import SparsePauliOp

try:
    from .data_loader import ProblemInstance
except ImportError:
    from data_loader import ProblemInstance


class RoutingQUBO:
    """Constructs the QUBO matrix and translates it into an Ising Hamiltonian."""

    def __init__(
        self,
        instance: Optional[ProblemInstance] = None,
        penalty: float = 40.0,
        start_node: int = 0,
        end_node: int = 0,
        customer_ids: Optional[List[int]] = None,
        customers_dict: Optional[Dict[int, dict]] = None,
        time_window_penalty: float = 0.0,
        start_departure_time: float = 0.0,
    ):
        """Initialize with problem instance or explicit subroute parameters.

        Args:
            instance: ProblemInstance containing customer coordinates and distance matrix (optional).
            penalty: Lagrange multiplier P enforcing customer and position permutation constraints.
            start_node: Predecessor node ID (depot 0 or route predecessor).
            end_node: Successor node ID (depot 0 or route successor).
            customer_ids: Explicit list of customer IDs to sequence (e.g. [17, 18, 19]).
            customers_dict: Dict mapping node ID to customer record with 'x', 'y', 'due_time', etc.
            time_window_penalty: Soft penalty multiplier for estimated late arrivals.
            start_departure_time: Time at which vehicle departs start_node.
        """
        self.instance = instance
        self.penalty = penalty
        self.start_node = start_node
        self.end_node = end_node
        self.time_window_penalty = time_window_penalty
        self.start_departure_time = start_departure_time

        if instance is not None and customer_ids is None:
            self.customer_ids = list(instance.customer_ids)
            self.n = instance.n_customers
            self.n_qubits = instance.n_qubits
            self.customers_dict = instance.customers
            self.dist_matrix = instance.distance_matrix
        elif customer_ids is not None:
            self.customer_ids = list(customer_ids)
            self.n = len(customer_ids)
            self.n_qubits = self.n * self.n
            self.customers_dict = customers_dict
            self.dist_matrix = None
        else:
            raise ValueError("Either instance or customer_ids must be provided to RoutingQUBO.")

        # Build QUBO matrix and offset
        self.Q, self.offset = self._build_qubo()

        # Derive Ising Hamiltonian coefficients: H_C = sum h_u Z_u + sum J_uv Z_u Z_v + C0 * I
        self.h, self.J, self.C0 = self._qubo_to_ising()

        # Build Qiskit SparsePauliOp representation
        self.hamiltonian = self._build_sparse_pauli_op()

    def get_distance(self, u: int, v: int) -> float:
        """Compute exact Euclidean distance between node IDs u and v."""
        if (self.instance is not None and self.dist_matrix is not None and 
                self.start_node == 0 and self.end_node == 0 and 
                self.customer_ids == self.instance.customer_ids):
            id_to_idx = {0: 0, **{cid: i + 1 for i, cid in enumerate(self.customer_ids)}}
            if u in id_to_idx and v in id_to_idx:
                return float(self.dist_matrix[id_to_idx[u], id_to_idx[v]])

        if self.customers_dict is not None and u in self.customers_dict and v in self.customers_dict:
            cu, cv = self.customers_dict[u], self.customers_dict[v]
            return float(math.hypot(cu["x"] - cv["x"], cu["y"] - cv["y"]))

        if self.instance is not None and u in self.instance.customers and v in self.instance.customers:
            cu, cv = self.instance.customers[u], self.instance.customers[v]
            return float(math.hypot(cu["x"] - cv["x"], cu["y"] - cv["y"]))

        raise KeyError(f"Unable to resolve coordinates for distance between node {u} and {v}")

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
        # Predecessor (start_node) -> First customer at relative position 0
        for i in range(n):
            u_var = self.var_index(i, 0)
            cid = self.customer_ids[i]
            d_start = self.get_distance(self.start_node, cid)
            Q[u_var, u_var] += d_start

        # Consecutive customers from position p to position p + 1
        for p in range(n - 1):
            for i in range(n):
                cid_i = self.customer_ids[i]
                for j in range(n):
                    if i != j:
                        cid_j = self.customer_ids[j]
                        u_var = self.var_index(i, p)
                        v_var = self.var_index(j, p + 1)
                        d_ij = self.get_distance(cid_i, cid_j)
                        if u_var < v_var:
                            Q[u_var, v_var] += d_ij
                        else:
                            Q[v_var, u_var] += d_ij

        # Last customer at position (n - 1) -> Successor (end_node)
        for i in range(n):
            u_var = self.var_index(i, n - 1)
            cid = self.customer_ids[i]
            d_end = self.get_distance(cid, self.end_node)
            Q[u_var, u_var] += d_end

        # --- 2. ROW CONSTRAINTS: Each customer i visited exactly once ---
        for i in range(n):
            offset += self.penalty * 1.0
            for p in range(n):
                u_var = self.var_index(i, p)
                Q[u_var, u_var] -= self.penalty * 1.0
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    u_var = self.var_index(i, p1)
                    v_var = self.var_index(i, p2)
                    if u_var > v_var:
                        u_var, v_var = v_var, u_var
                    Q[u_var, v_var] += self.penalty * 2.0

        # --- 3. COLUMN CONSTRAINTS: Each position p visited by exactly one customer ---
        for p in range(n):
            offset += self.penalty * 1.0
            for i in range(n):
                u_var = self.var_index(i, p)
                Q[u_var, u_var] -= self.penalty * 1.0
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    u_var = self.var_index(i1, p)
                    v_var = self.var_index(i2, p)
                    if u_var > v_var:
                        u_var, v_var = v_var, u_var
                    Q[u_var, v_var] += self.penalty * 2.0

        # --- 4. OPTIONAL SOFT TIME-WINDOW ESTIMATE PENALTIES ---
        if self.time_window_penalty > 0 and self.customers_dict is not None:
            for i in range(n):
                cid = self.customer_ids[i]
                c_info = self.customers_dict.get(cid, {})
                due = c_info.get("due_time", float("inf"))
                for p in range(n):
                    est_arr = self.start_departure_time + self.get_distance(self.start_node, cid) + p * 90.0
                    if est_arr > due:
                        lateness = est_arr - due
                        u_var = self.var_index(i, p)
                        Q[u_var, u_var] += self.time_window_penalty * lateness

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


class SubrouteQUBO(RoutingQUBO):
    """QUBO formulation for re-sequencing an intermediate route window between fixed endpoints u and w."""

    def __init__(
        self,
        start_node: int,
        end_node: int,
        customer_ids: List[int],
        customers_dict: Dict[int, dict],
        penalty: float = 35.0,
        time_window_penalty: float = 0.0,
        start_departure_time: float = 0.0,
    ):
        super().__init__(
            instance=None,
            penalty=penalty,
            start_node=start_node,
            end_node=end_node,
            customer_ids=customer_ids,
            customers_dict=customers_dict,
            time_window_penalty=time_window_penalty,
            start_departure_time=start_departure_time,
        )

