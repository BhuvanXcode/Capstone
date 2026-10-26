"""Compact Encoding and Hybrid Decomposition Module for Scaling Quantum VRPTW.

Addresses the scalability barrier of order-based N^2 QUBO routing formulations:
1. Compact Logarithmic Binary Encoding Analysis: Compares N^2 (1-hot) vs N*ceil(log2(N)) (binary position index).
2. Hybrid Quantum-Classical Decomposition (Cluster-First, Route-Second):
   Partitions large-scale fleet instances (e.g. 100-customer Solomon C101) into
   capacity-feasible clusters of 3 to 5 customers, suitable for execution on
   near-term quantum subroutines (QAOA / QUBO).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class EncodingComparison:
    """Analytical comparison between 1-hot (N^2) and compact binary encoding."""
    n_customers: int
    one_hot_qubits: int
    one_hot_ram_human: str
    compact_bits_per_cust: int
    compact_qubits: int
    compact_ram_human: str
    qubit_reduction_percent: float
    constraint_degree_one_hot: int
    constraint_degree_compact: int
    nisq_feasibility_verdict: str


class CompactEncodingAnalyzer:
    """Analyzes and compares alternative quantum qubit representations for routing."""

    @staticmethod
    def compare_encodings(n: int) -> EncodingComparison:
        """Compare 1-hot order encoding against compact binary position encoding."""
        # 1-hot: N qubits per customer -> N^2 qubits
        one_hot_q = n * n
        if one_hot_q <= 60:
            one_hot_ram = (2 ** one_hot_q) * 16.0
            if one_hot_ram >= 1024.0 ** 4:
                oh_ram_human = f"{one_hot_ram / (1024.0**4):,.1f} TB"
            elif one_hot_ram >= 1024.0 ** 3:
                oh_ram_human = f"{one_hot_ram / (1024.0**3):,.2f} GB"
            elif one_hot_ram >= 1024.0 ** 2:
                oh_ram_human = f"{one_hot_ram / (1024.0**2):,.2f} MB"
            else:
                oh_ram_human = f"{one_hot_ram / 1024.0:,.1f} KB"
        else:
            oh_ram_human = "> 10^15 TB"

        # Compact: ceil(log2(N)) bits per customer -> N * ceil(log2(N)) qubits
        bits_per_cust = int(math.ceil(math.log2(n)))
        compact_q = n * bits_per_cust
        if compact_q <= 60:
            compact_ram = (2 ** compact_q) * 16.0
            if compact_ram >= 1024.0 ** 4:
                comp_ram_human = f"{compact_ram / (1024.0**4):,.1f} TB"
            elif compact_ram >= 1024.0 ** 3:
                comp_ram_human = f"{compact_ram / (1024.0**3):,.2f} GB"
            elif compact_ram >= 1024.0 ** 2:
                comp_ram_human = f"{compact_ram / (1024.0**2):,.2f} MB"
            else:
                comp_ram_human = f"{compact_ram / 1024.0:,.1f} KB"
        else:
            comp_ram_human = "> 10^15 TB"

        reduction = (1.0 - compact_q / one_hot_q) * 100.0

        if compact_q <= 20:
            verdict = "Easily simulateable on laptop"
        elif compact_q <= 28:
            verdict = "Simulateable with high memory (100MB - 4GB)"
        else:
            verdict = "Exceeds classical laptop memory"

        return EncodingComparison(
            n_customers=n,
            one_hot_qubits=one_hot_q,
            one_hot_ram_human=oh_ram_human,
            compact_bits_per_cust=bits_per_cust,
            compact_qubits=compact_q,
            compact_ram_human=comp_ram_human,
            qubit_reduction_percent=round(reduction, 1),
            constraint_degree_one_hot=2,  # Standard 2-local Ising (pairwise ZZ)
            constraint_degree_compact=n,   # Higher-order multi-qubit polynomials (PUBO)
            nisq_feasibility_verdict=verdict,
        )

    @classmethod
    def get_comparison_table(cls, customer_list: List[int] = None) -> List[Dict]:
        """Generate structured comparison table across customer range."""
        if customer_list is None:
            customer_list = [3, 4, 5, 6, 8, 10, 20, 100]

        table = []
        for n in customer_list:
            comp = cls.compare_encodings(n)
            table.append({
                "customers": n,
                "one_hot_qubits": comp.one_hot_qubits,
                "one_hot_ram": comp.one_hot_ram_human,
                "compact_qubits": comp.compact_qubits,
                "compact_ram": comp.compact_ram_human,
                "qubit_savings": f"{comp.qubit_reduction_percent:.1f}%",
                "tradeoff": f"2-local quadratic vs {comp.constraint_degree_compact}-order PUBO",
                "verdict": comp.nisq_feasibility_verdict,
            })
        return table


class HybridVRPTWDecomposer:
    """Decomposes large-scale VRPTW instances into quantum-tractable subproblems.

    Implements the industry-standard 'Cluster-First, Route-Second' hybrid paradigm:
    1. Classical Stage: Partitions the full customer fleet into capacity/time-window clusters.
    2. Quantum Subroutine: Each cluster of 3 to 5 customers is solved as a focused QUBO/QAOA tour.
    3. Classical Synthesis: Recombines sub-tours into an end-to-end full fleet operational schedule.
    """

    def __init__(self, full_customers: Dict[int, dict], vehicle_capacity: float = 200.0):
        self.customers = full_customers
        self.capacity = vehicle_capacity
        self.depot = full_customers[0]

    def decompose_by_sweep(self, target_cluster_size: int = 4) -> List[Dict]:
        """Cluster customers into geographically and capacity-aligned sub-routes using polar sweep."""
        depot_x, depot_y = self.depot["x"], self.depot["y"]

        # Calculate polar angle for every customer relative to depot
        customer_list = []
        for cid, info in self.customers.items():
            if cid == 0:
                continue
            dx = info["x"] - depot_x
            dy = info["y"] - depot_y
            angle = math.atan2(dy, dx)
            customer_list.append({
                "id": cid,
                "angle": angle,
                "demand": info["demand"],
                "ready_time": info["ready_time"],
                "due_time": info["due_time"],
            })

        # Sort customers by polar angle (angular sweep heuristic)
        customer_list.sort(key=lambda c: (c["angle"], c["ready_time"]))

        clusters = []
        current_cluster = []
        current_demand = 0.0

        for cust in customer_list:
            cid = cust["id"]
            dem = cust["demand"]

            # If adding this customer violates capacity or exceeds target cluster size
            if (current_demand + dem > self.capacity or len(current_cluster) >= target_cluster_size) and current_cluster:
                clusters.append({
                    "cluster_index": len(clusters) + 1,
                    "customer_ids": [c["id"] for c in current_cluster],
                    "total_demand": current_demand,
                    "size": len(current_cluster),
                    "quantum_subproblem_qubits": len(current_cluster) ** 2,
                    "is_quantum_tractable": len(current_cluster) <= 5,
                })
                current_cluster = []
                current_demand = 0.0

            current_cluster.append(cust)
            current_demand += dem

        if current_cluster:
            clusters.append({
                "cluster_index": len(clusters) + 1,
                "customer_ids": [c["id"] for c in current_cluster],
                "total_demand": current_demand,
                "size": len(current_cluster),
                "quantum_subproblem_qubits": len(current_cluster) ** 2,
                "is_quantum_tractable": len(current_cluster) <= 5,
            })

        return clusters
