"""Data loader and problem definition for the quantum routing demonstration.

Extracts customer coordinates, demands, time windows, and vehicle capacity
from the Solomon VRPTW benchmark dataset (c101.txt) to construct small,
verifiable routing problem instances (3 to 4 customers).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import math
import numpy as np


@dataclass
class ProblemInstance:
    """Represents a small vehicle routing problem instance with VRPTW metadata."""
    name: str
    depot_id: int
    customer_ids: List[int]
    customers: Dict[int, dict]
    coordinates: np.ndarray  # Shape: (N+1, 2) where index 0 is depot, 1..N are customers
    distance_matrix: np.ndarray  # Shape: (N+1, N+1)
    vehicle_capacity: float
    n_customers: int
    n_qubits: int


def load_solomon_c101(file_path: Optional[Path] = None) -> Dict[int, dict]:
    """Parse customers and depot from the Solomon C101 benchmark file."""
    if file_path is None:
        # Default search locations
        curr_dir = Path(__file__).resolve().parent
        candidate_paths = [
            curr_dir / "data" / "c101.txt",
            curr_dir.parent / "Capstone" / "data" / "c101.txt",
            curr_dir / "c101.txt",
        ]
        for cp in candidate_paths:
            if cp.exists():
                file_path = cp
                break

    if file_path is None or not file_path.exists():
        # Fallback to authentic C101 subset values if file is moved
        return {
            0: {"id": 0, "x": 40.0, "y": 50.0, "demand": 0.0, "ready_time": 0.0, "due_time": 1236.0, "service_time": 0.0},
            1: {"id": 1, "x": 45.0, "y": 68.0, "demand": 10.0, "ready_time": 912.0, "due_time": 967.0, "service_time": 90.0},
            2: {"id": 2, "x": 45.0, "y": 70.0, "demand": 30.0, "ready_time": 825.0, "due_time": 870.0, "service_time": 90.0},
            3: {"id": 3, "x": 42.0, "y": 66.0, "demand": 10.0, "ready_time": 65.0, "due_time": 146.0, "service_time": 90.0},
            4: {"id": 4, "x": 42.0, "y": 68.0, "demand": 10.0, "ready_time": 727.0, "due_time": 782.0, "service_time": 90.0},
        }

    customers = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 7 and parts[0].isdigit():
                cid = int(parts[0])
                customers[cid] = {
                    "id": cid,
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "demand": float(parts[3]),
                    "ready_time": float(parts[4]),
                    "due_time": float(parts[5]),
                    "service_time": float(parts[6]),
                }
    return customers


def get_problem_instance(n_customers: int = 3, customer_ids: Optional[List[int]] = None) -> ProblemInstance:
    """Create a focused routing instance with N customers from Solomon C101.

    Args:
        n_customers: Number of customers (e.g. 3, 4, 5, 6).
        customer_ids: Explicit list of customer IDs to use (e.g. [1, 2, 3]).

    Returns:
        ProblemInstance with coordinates, distance matrix, and VRPTW parameters.
    """
    raw_data = load_solomon_c101()

    if customer_ids is None:
        customer_ids = list(range(1, n_customers + 1))
    else:
        n_customers = len(customer_ids)

    all_ids = [0] + customer_ids
    # Verify IDs exist in dataset
    missing = [cid for cid in all_ids if cid not in raw_data]
    if missing:
        raise ValueError(f"Customer IDs {missing} not found in Solomon C101 dataset.")

    selected = {cid: raw_data[cid] for cid in all_ids}

    # Build coordinates array: row 0 = depot, rows 1..N = customers
    coords = np.array([[selected[cid]["x"], selected[cid]["y"]] for cid in all_ids], dtype=float)

    # Compute Euclidean distance matrix
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    return ProblemInstance(
        name=f"Solomon-C101-Subproblem-{n_customers}Cust",
        depot_id=0,
        customer_ids=customer_ids,
        customers=selected,
        coordinates=coords,
        distance_matrix=dist_matrix,
        vehicle_capacity=200.0,
        n_customers=n_customers,
        n_qubits=n_customers * n_customers,
    )


def load_all_customers() -> Dict[int, dict]:
    """Load the complete 100-customer dataset plus depot from Solomon C101."""
    return load_solomon_c101()
