"""Resource Pre-Flight Estimator for Quantum Vehicle Routing Simulations.

Calculates theoretical qubit allocations, statevector Hilbert space dimension,
RAM memory footprints, gate count estimates, and feasible subspace ratios
before allocating simulator resources on an ordinary laptop CPU.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple


@dataclass
class ResourceProfile:
    """Computational resource profile for a quantum routing instance."""
    n_customers: int
    n_qubits: int
    hilbert_dimension: int
    ram_bytes: float
    ram_mb: float
    ram_gb: float
    ram_human: str
    pauli_terms_estimate: int
    feasible_routes_count: int
    feasible_fraction: float
    feasible_percent_str: str
    is_safe: bool
    status_label: str
    recommendation: str


class ResourceEstimator:
    """Calculates memory and computational feasibility for quantum routing instances."""

    # Maximum safe statevector RAM allocation on an ordinary laptop (default 2.0 GB)
    DEFAULT_MAX_SAFE_RAM_GB = 2.0

    @classmethod
    def estimate(cls, n_customers: int, max_safe_ram_gb: float = DEFAULT_MAX_SAFE_RAM_GB) -> ResourceProfile:
        """Estimate required resources for N customers with N^2 order-based QUBO encoding."""
        n_qubits = n_customers * n_customers

        # Avoid float overflow when n_qubits > 64 (e.g. 100 or 10000 qubits)
        if n_qubits > 64:
            hilbert_dim = 2 ** n_qubits
            ram_bytes = float("inf")
            ram_mb = float("inf")
            ram_gb = float("inf")
            ram_human = f"> 10^{int(n_qubits * 0.30103) - 12} TB"
            feasible_routes = math.factorial(n_customers)
            feasible_fraction = 0.0
            feasible_pct_str = "0.0% (Infinitesimal)"
        else:
            hilbert_dim = 2 ** n_qubits
            ram_bytes = float(hilbert_dim * 16)
            ram_mb = ram_bytes / (1024.0 ** 2)
            ram_gb = ram_bytes / (1024.0 ** 3)
            ram_tb = ram_bytes / (1024.0 ** 4)

            if ram_tb >= 1.0:
                ram_human = f"{ram_tb:,.1f} TB"
            elif ram_gb >= 1.0:
                ram_human = f"{ram_gb:,.2f} GB"
            elif ram_mb >= 1.0:
                ram_human = f"{ram_mb:,.2f} MB"
            else:
                ram_human = f"{ram_bytes / 1024.0:,.1f} KB"

            feasible_routes = math.factorial(n_customers)
            feasible_fraction = float(feasible_routes) / float(hilbert_dim)
            if feasible_fraction < 1e-4:
                feasible_pct_str = f"{feasible_fraction * 100.0:.2e}%"
            else:
                feasible_pct_str = f"{feasible_fraction * 100.0:.4f}%"

        # Hamiltonian terms estimate: N^2 linear + interactions between valid variable pairs
        # Roughly O(N^4) pairs
        pauli_terms = n_qubits + (n_qubits * (n_qubits - 1)) // 2

        # Safety decision logic
        if ram_gb > max_safe_ram_gb or n_qubits > 28:
            is_safe = False
            status_label = "EXCEEDS_MEMORY_LIMIT"
            recommendation = (
                f"SAFELY ABORT: {n_customers} customers requires {n_qubits} qubits and {ram_human} RAM. "
                f"Full statevector simulation on a laptop CPU will exhaust memory (OOM crash)."
            )
        elif n_qubits >= 25:
            # 25 qubits = 512 MB, simulateable but requires conservative iteration budget
            is_safe = True
            status_label = "HEAVY_COMPUTATION"
            recommendation = (
                f"CAUTION: {n_customers} customers requires {n_qubits} qubits ({ram_human} RAM). "
                f"Simulateable on laptop, but statevector evaluation will be CPU-intensive. Use bounded iterations."
            )
        else:
            is_safe = True
            status_label = "SAFE_FOR_LAPTOP"
            recommendation = f"SAFE: {n_customers} customers ({n_qubits} qubits, {ram_human} RAM) runs in seconds on laptop CPU."

        return ResourceProfile(
            n_customers=n_customers,
            n_qubits=n_qubits,
            hilbert_dimension=hilbert_dim,
            ram_bytes=ram_bytes,
            ram_mb=ram_mb,
            ram_gb=ram_gb,
            ram_human=ram_human,
            pauli_terms_estimate=pauli_terms,
            feasible_routes_count=feasible_routes,
            feasible_fraction=feasible_fraction,
            feasible_percent_str=feasible_pct_str,
            is_safe=is_safe,
            status_label=status_label,
            recommendation=recommendation,
        )

    @classmethod
    def generate_scaling_comparison_table(cls, customer_range: List[int] = None) -> List[Dict]:
        """Generate a comparison table across multiple customer problem sizes."""
        if customer_range is None:
            customer_range = [3, 4, 5, 6, 8, 10, 100]

        table = []
        for n in customer_range:
            profile = cls.estimate(n)
            table.append({
                "customers": n,
                "qubits": profile.n_qubits,
                "hilbert_states": f"2^{profile.n_qubits}" if profile.n_qubits > 30 else f"{profile.hilbert_dimension:,}",
                "ram_required": profile.ram_human,
                "feasible_routes": f"{profile.feasible_routes_count:,}",
                "feasible_fraction": profile.feasible_percent_str,
                "laptop_status": profile.status_label,
            })
        return table
