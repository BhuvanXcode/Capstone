"""Exact Classical Benchmark Solver for small routing instances.

Performs exhaustive permutation enumeration for 3 to 4 customer instances to find
the globally optimal minimum-distance route, as well as evaluating full VRPTW
capacity and time-window feasibility for each candidate permutation.
"""

from dataclasses import dataclass
import itertools
import time
from typing import Dict, List, Sequence, Tuple
import numpy as np

try:
    from .data_loader import ProblemInstance
except ImportError:
    from data_loader import ProblemInstance


@dataclass
class RouteEvaluation:
    """Detailed evaluation of a single routing candidate."""
    route: List[int]  # Full route including depot: [0, c1, ..., cN, 0]
    customer_order: List[int]  # Customers only: [c1, ..., cN]
    total_distance: float
    total_demand: float
    capacity_limit: float
    is_capacity_feasible: bool
    time_schedule: List[dict]  # Detailed arrival, wait, service, departure per stop
    total_wait_time: float
    total_service_time: float
    total_duration: float
    is_time_window_feasible: bool
    time_window_violations: List[dict]
    is_vrptw_feasible: bool


class ExactClassicalSolver:
    """Exhaustive classical permutation solver for small routing instances."""

    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.dist = instance.distance_matrix
        self.customers = instance.customers
        self.capacity = instance.vehicle_capacity
        self.customer_ids = instance.customer_ids
        self.id_to_idx = {0: 0, **{cid: i + 1 for i, cid in enumerate(self.customer_ids)}}

    def evaluate_route(self, customer_permutation: Sequence[int]) -> RouteEvaluation:
        """Evaluate a given customer permutation for distance, capacity, and time windows."""
        route = [0] + list(customer_permutation) + [0]

        # 1. Distance Calculation
        distance = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            distance += self.dist[self.id_to_idx[u], self.id_to_idx[v]]

        # 2. Capacity Check
        total_demand = sum(self.customers[c]["demand"] for c in customer_permutation)
        is_capacity_feasible = total_demand <= self.capacity

        # 3. Time Window Schedule Evaluation
        schedule = []
        current_time = self.customers[0]["ready_time"]
        total_wait = 0.0
        total_service = 0.0
        violations = []

        # Record Depot departure
        schedule.append({
            "node": 0,
            "arrival_time": 0.0,
            "waiting_time": 0.0,
            "service_start": 0.0,
            "service_time": 0.0,
            "departure_time": current_time,
            "due_time": self.customers[0]["due_time"],
            "lateness": 0.0,
        })

        for k in range(len(route) - 1):
            u = route[k]
            v = route[k + 1]
            travel_time = self.dist[self.id_to_idx[u], self.id_to_idx[v]]
            arrival_time = current_time + travel_time

            v_info = self.customers[v]
            ready_time = v_info["ready_time"]
            due_time = v_info["due_time"]
            service_time = v_info["service_time"]

            # Waiting if early
            wait_time = max(0.0, ready_time - arrival_time)
            service_start = arrival_time + wait_time
            lateness = max(0.0, arrival_time - due_time)

            if lateness > 1e-5:
                violations.append({
                    "customer": v,
                    "arrival": round(arrival_time, 2),
                    "due_date": round(due_time, 2),
                    "lateness": round(lateness, 2),
                })

            departure_time = service_start + service_time
            current_time = departure_time
            total_wait += wait_time
            total_service += service_time

            schedule.append({
                "node": v,
                "arrival_time": round(arrival_time, 2),
                "waiting_time": round(wait_time, 2),
                "service_start": round(service_start, 2),
                "service_time": round(service_time, 2),
                "departure_time": round(departure_time, 2),
                "due_time": round(due_time, 2),
                "lateness": round(lateness, 2),
            })

        is_tw_feasible = len(violations) == 0
        is_vrptw_feasible = is_capacity_feasible and is_tw_feasible

        return RouteEvaluation(
            route=route,
            customer_order=list(customer_permutation),
            total_distance=float(distance),
            total_demand=float(total_demand),
            capacity_limit=float(self.capacity),
            is_capacity_feasible=is_capacity_feasible,
            time_schedule=schedule,
            total_wait_time=float(total_wait),
            total_service_time=float(total_service),
            total_duration=float(current_time),
            is_time_window_feasible=is_tw_feasible,
            time_window_violations=violations,
            is_vrptw_feasible=is_vrptw_feasible,
        )

    def solve(self) -> Dict:
        """Enumerate all customer permutations and return exhaustive classical results."""
        start_time = time.perf_counter()

        all_permutations = list(itertools.permutations(self.customer_ids))
        evaluations: List[RouteEvaluation] = [self.evaluate_route(p) for p in all_permutations]

        elapsed_time = time.perf_counter() - start_time

        # Sort by distance first, then prefer VRPTW-feasible routes among distance ties
        evaluations.sort(key=lambda ev: (round(ev.total_distance, 6), not ev.is_vrptw_feasible))
        optimal_distance = evaluations[0].total_distance

        # Find all routes matching the optimal distance (e.g. forward and reverse directions)
        optimal_evals = [ev for ev in evaluations if np.isclose(ev.total_distance, optimal_distance)]

        return {
            "routes_evaluated": len(evaluations),
            "solve_time_seconds": round(elapsed_time, 6),
            "optimal_distance": round(optimal_distance, 4),
            "optimal_routes": [ev.route for ev in optimal_evals],
            "best_evaluation": evaluations[0],
            "all_evaluations": evaluations,
        }
