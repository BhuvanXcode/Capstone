"""Decoder and Multi-Level Constraint Validator for QAOA Routing Results.

Decodes measured quantum bitstrings into routing candidates, verifies the model-level
QUBO permutation constraints (visited-once and occupied-once), and executes
independent classical post-validation checks for vehicle capacity and customer time windows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from .data_loader import ProblemInstance
    from .classical_solver import ExactClassicalSolver, RouteEvaluation
    from .qubo_formulation import RoutingQUBO
except ImportError:
    from data_loader import ProblemInstance
    from classical_solver import ExactClassicalSolver, RouteEvaluation
    from qubo_formulation import RoutingQUBO


@dataclass
class DecodedQuantumCandidate:
    """Comprehensive decoding and validation report for a measured bitstring."""
    bitstring: str
    matrix: np.ndarray  # Shape: (N, N) where row i is customer i+1, col p is position p
    shots_count: int
    probability: float
    qubo_energy: float
    routing_distance: Optional[float]
    # Model-Level Constraint Validation
    row_sums: List[int]
    col_sums: List[int]
    is_permutation_feasible: bool
    model_constraint_report: str
    # Route decoding
    route: Optional[List[int]]  # [0, c1, ..., cN, 0] if feasible
    # External VRPTW Constraint Validation (Post-Quantum Classical Checks)
    vrptw_evaluation: Optional[RouteEvaluation]


class CandidateDecoderAndValidator:
    """Decodes bitstrings from Qiskit and performs structural and VRPTW validation."""

    def __init__(self, instance: ProblemInstance, qubo: RoutingQUBO):
        self.instance = instance
        self.qubo = qubo
        self.n = instance.n_customers
        self.n_qubits = instance.n_qubits
        self.customer_ids = instance.customer_ids
        self.classical_solver = ExactClassicalSolver(instance)

    def bitstring_to_matrix(self, bitstring: str) -> np.ndarray:
        """Convert a Qiskit bitstring into an (N x N) decision variable matrix.

        Note on Qiskit Bitstring Endianness:
            Qiskit measurement bitstrings are indexed in little-endian order:
            character s[n_qubits - 1 - u] corresponds to qubit u.
        """
        n_qubits = self.n_qubits
        if len(bitstring) != n_qubits:
            raise ValueError(f"Bitstring length ({len(bitstring)}) != n_qubits ({n_qubits})")

        x = np.array([int(bitstring[n_qubits - 1 - u]) for u in range(n_qubits)], dtype=int)
        return x.reshape((self.n, self.n))

    def decode_and_validate(self, bitstring: str, shots: int = 1, total_shots: int = 1) -> DecodedQuantumCandidate:
        """Decode a single bitstring and evaluate both model constraints and VRPTW constraints."""
        matrix = self.bitstring_to_matrix(bitstring)
        x_flat = np.array([int(bitstring[self.n_qubits - 1 - u]) for u in range(self.n_qubits)], dtype=float)

        qubo_energy = self.qubo.evaluate_bitstring_qubo(x_flat)
        row_sums = [int(s) for s in matrix.sum(axis=1)]
        col_sums = [int(s) for s in matrix.sum(axis=0)]

        is_feasible = all(s == 1 for s in row_sums) and all(s == 1 for s in col_sums)

        # Build model constraint explanation
        issues = []
        if not all(s == 1 for s in row_sums):
            for i, s in enumerate(row_sums):
                cid = self.customer_ids[i]
                if s == 0:
                    issues.append(f"Customer {cid} omitted (visited 0 times)")
                elif s > 1:
                    issues.append(f"Customer {cid} visited {s} times")

        if not all(s == 1 for s in col_sums):
            for p, s in enumerate(col_sums):
                if s == 0:
                    issues.append(f"Route step {p} unassigned (0 customers)")
                elif s > 1:
                    issues.append(f"Route step {p} oversubscribed ({s} customers)")

        if is_feasible:
            report = "FEASIBLE: Valid one-to-one customer permutation (each customer visited once, each position filled)."
            # Extract customer sequence
            perm = []
            for p in range(self.n):
                c_idx = int(np.argmax(matrix[:, p]))
                perm.append(self.customer_ids[c_idx])
            route = [0] + perm + [0]

            id_to_idx = {0: 0, **{cid: i + 1 for i, cid in enumerate(self.customer_ids)}}
            # Compute route distance
            distance = 0.0
            for k in range(len(route) - 1):
                distance += self.instance.distance_matrix[id_to_idx[route[k]], id_to_idx[route[k + 1]]]

            # Execute separate classical VRPTW validation checks
            vrptw_eval = self.classical_solver.evaluate_route(perm)
        else:
            report = f"INFEASIBLE MODEL CONSTRAINTS: {'; '.join(issues)}."
            route = None
            distance = None
            vrptw_eval = None

        return DecodedQuantumCandidate(
            bitstring=bitstring,
            matrix=matrix,
            shots_count=shots,
            probability=shots / total_shots if total_shots > 0 else 0.0,
            qubo_energy=round(qubo_energy, 4),
            routing_distance=round(distance, 4) if distance is not None else None,
            row_sums=row_sums,
            col_sums=col_sums,
            is_permutation_feasible=is_feasible,
            model_constraint_report=report,
            route=route,
            vrptw_evaluation=vrptw_eval,
        )
