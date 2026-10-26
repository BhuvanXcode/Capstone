"""Comprehensive Fleet Validation Engine for Multi-Vehicle VRPTW.

Validates complete fleet routing solutions covering all 100 customers from Solomon C101:
1. Customer Coverage: Every customer in 1..100 must be visited exactly once (no omissions, no duplicates).
2. Vehicle Capacity: Total delivered demand per vehicle must not exceed vehicle capacity (200.0).
3. Operational Time Windows: Arrival times, waiting times, service starts, service times (90.0),
   departure times, and due times are computed along each route from Depot departure (t=0)
   to Depot return (t <= 1236.0). Lateness and violations are strictly tracked.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class StopSchedule:
    """Schedule details for a single stop along a vehicle route."""
    stop_index: int
    customer_id: int
    arrival_time: float
    ready_time: float
    due_time: float
    waiting_time: float
    service_start: float
    service_time: float
    departure_time: float
    is_time_window_feasible: bool
    lateness: float
    demand: float
    cumulative_load: float


@dataclass
class VehicleRouteEvaluation:
    """Detailed evaluation of an individual vehicle route."""
    vehicle_id: int
    route: List[int]  # [0, c1, ..., cm, 0]
    customer_ids: List[int]  # [c1, ..., cm]
    total_distance: float
    total_demand: float
    capacity_limit: float
    is_capacity_feasible: bool
    capacity_overload: float
    schedule: List[StopSchedule]
    total_waiting_time: float
    total_service_time: float
    total_travel_time: float
    total_duration: float
    return_to_depot_time: float
    is_depot_return_feasible: bool
    is_time_window_feasible: bool
    time_window_violations_count: int
    violations_detail: List[dict]
    is_route_feasible: bool


@dataclass
class FleetEvaluation:
    """Comprehensive evaluation of a full fleet routing solution across all 100 customers."""
    dataset_name: str
    total_customers_expected: int
    total_customers_served: int
    unique_customers_served: int
    missing_customer_ids: List[int]
    duplicate_customer_ids: List[int]
    is_coverage_valid: bool
    coverage_report: str
    
    vehicles_used: int
    vehicle_capacity_limit: float
    total_demand_expected: float
    total_demand_delivered: float
    is_capacity_valid: bool
    capacity_violations_count: int
    capacity_overloads_detail: List[dict]
    
    total_distance: float
    total_travel_time: float
    total_waiting_time: float
    total_service_time: float
    total_fleet_duration: float
    
    is_time_window_valid: bool
    time_window_violations_count: int
    total_lateness: float
    time_window_violations_detail: List[dict]
    
    is_fully_feasible: bool
    status_summary: str
    routes: List[VehicleRouteEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert fleet evaluation to a JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_customers_expected": self.total_customers_expected,
            "total_customers_served": self.total_customers_served,
            "unique_customers_served": self.unique_customers_served,
            "missing_customer_ids": self.missing_customer_ids,
            "duplicate_customer_ids": self.duplicate_customer_ids,
            "is_coverage_valid": self.is_coverage_valid,
            "coverage_report": self.coverage_report,
            "vehicles_used": self.vehicles_used,
            "vehicle_capacity_limit": self.vehicle_capacity_limit,
            "total_demand_expected": self.total_demand_expected,
            "total_demand_delivered": self.total_demand_delivered,
            "is_capacity_valid": self.is_capacity_valid,
            "capacity_violations_count": self.capacity_violations_count,
            "capacity_overloads_detail": self.capacity_overloads_detail,
            "total_distance": round(self.total_distance, 4),
            "total_travel_time": round(self.total_travel_time, 4),
            "total_waiting_time": round(self.total_waiting_time, 4),
            "total_service_time": round(self.total_service_time, 4),
            "total_fleet_duration": round(self.total_fleet_duration, 4),
            "is_time_window_valid": self.is_time_window_valid,
            "time_window_violations_count": self.time_window_violations_count,
            "total_lateness": round(self.total_lateness, 4),
            "time_window_violations_detail": self.time_window_violations_detail,
            "is_fully_feasible": self.is_fully_feasible,
            "status_summary": self.status_summary,
            "routes_summary": [
                {
                    "vehicle_id": r.vehicle_id,
                    "customer_count": len(r.customer_ids),
                    "route": r.route,
                    "total_distance": round(r.total_distance, 2),
                    "total_demand": r.total_demand,
                    "is_capacity_feasible": r.is_capacity_feasible,
                    "is_time_window_feasible": r.is_time_window_feasible,
                    "violations_count": r.time_window_violations_count,
                    "return_time": round(r.return_to_depot_time, 2),
                }
                for r in self.routes
            ],
        }


class FleetValidator:
    """Validates multi-vehicle fleet routes against Solomon C101 benchmark constraints."""

    def __init__(self, raw_customers: Dict[int, dict], vehicle_capacity: float = 200.0):
        self.customers = raw_customers
        self.capacity = vehicle_capacity
        self.depot = raw_customers[0]
        self.all_customer_ids = sorted([cid for cid in raw_customers.keys() if cid != 0])
        self.expected_total_customers = len(self.all_customer_ids)
        self.expected_total_demand = sum(self.customers[c]["demand"] for c in self.all_customer_ids)

        # Precompute Euclidean distance helper
        self._dist_cache = {}

    def get_distance(self, u: int, v: int) -> float:
        """Compute exact Euclidean distance between two customer nodes."""
        pair = (min(u, v), max(u, v))
        if pair not in self._dist_cache:
            cu, cv = self.customers[u], self.customers[v]
            d = math.hypot(cu["x"] - cv["x"], cu["y"] - cv["y"])
            self._dist_cache[pair] = d
        return self._dist_cache[pair]

    def validate_fleet(self, routes: List[List[int]], solution_name: str = "Solution") -> FleetEvaluation:
        """Thoroughly audit a collection of vehicle routes.
        
        Args:
            routes: List of routes, where each route is either [c1, ..., cm] or [0, c1, ..., cm, 0].
            solution_name: Identifier label for reporting.
            
        Returns:
            FleetEvaluation containing coverage, capacity, and time-window audit metrics.
        """
        # Standardize routes to [0, c1, ..., cm, 0] format
        normalized_routes = []
        for r in routes:
            # Strip extra depot entries
            custs = [c for c in r if c != 0]
            if custs:  # Only active routes with at least 1 customer
                normalized_routes.append([0] + custs + [0])

        # 1. Customer Coverage Validation
        all_visited_customers = []
        for r in normalized_routes:
            all_visited_customers.extend(r[1:-1])

        visited_set = set(all_visited_customers)
        expected_set = set(self.all_customer_ids)

        missing_customers = sorted(list(expected_set - visited_set))
        
        # Check for duplicates
        seen = set()
        duplicates = []
        for cid in all_visited_customers:
            if cid in seen:
                if cid not in duplicates:
                    duplicates.append(cid)
            seen.add(cid)
        duplicates.sort()

        is_coverage_valid = (len(missing_customers) == 0) and (len(duplicates) == 0) and (len(all_visited_customers) == self.expected_total_customers)

        if is_coverage_valid:
            coverage_report = f"PASSED: All {self.expected_total_customers} customers are visited exactly once."
        else:
            issues = []
            if missing_customers:
                issues.append(f"{len(missing_customers)} customers missing ({missing_customers[:5]}...)")
            if duplicates:
                issues.append(f"{len(duplicates)} customers duplicated ({duplicates[:5]}...)")
            coverage_report = f"FAILED: {'; '.join(issues)}"

        # 2. Evaluate Each Individual Vehicle Route
        evaluated_routes: List[VehicleRouteEvaluation] = []
        total_fleet_distance = 0.0
        total_fleet_travel_time = 0.0
        total_fleet_wait_time = 0.0
        total_fleet_service_time = 0.0
        total_fleet_demand = 0.0
        all_capacity_overloads = []
        all_time_window_violations = []
        total_lateness = 0.0

        for v_idx, route in enumerate(normalized_routes):
            cust_ids = route[1:-1]
            route_demand = sum(self.customers[c]["demand"] for c in cust_ids)
            total_fleet_demand += route_demand

            is_cap_ok = (route_demand <= self.capacity)
            overload = max(0.0, route_demand - self.capacity)
            if not is_cap_ok:
                all_capacity_overloads.append({
                    "vehicle_id": v_idx + 1,
                    "total_demand": route_demand,
                    "capacity_limit": self.capacity,
                    "overload": overload,
                })

            # Time schedule evaluation
            schedule: List[StopSchedule] = []
            curr_time = 0.0  # Vehicle leaves depot at t = 0
            route_dist = 0.0
            route_wait = 0.0
            route_service = 0.0
            route_violations = []
            cum_load = 0.0

            # Depot departure
            schedule.append(StopSchedule(
                stop_index=0,
                customer_id=0,
                arrival_time=0.0,
                ready_time=self.depot["ready_time"],
                due_time=self.depot["due_time"],
                waiting_time=0.0,
                service_start=0.0,
                service_time=0.0,
                departure_time=0.0,
                is_time_window_feasible=True,
                lateness=0.0,
                demand=0.0,
                cumulative_load=0.0,
            ))

            for k in range(len(route) - 1):
                u, v = route[k], route[k + 1]
                leg_dist = self.get_distance(u, v)
                route_dist += leg_dist
                arrival_time = curr_time + leg_dist

                v_info = self.customers[v]
                r_time = v_info["ready_time"]
                d_time = v_info["due_time"]
                s_time = v_info["service_time"]
                dem = v_info["demand"]
                cum_load += dem

                # Early arrival -> wait until ready time
                wait_time = max(0.0, r_time - arrival_time)
                service_start = arrival_time + wait_time

                # Lateness check
                lateness = max(0.0, arrival_time - d_time)
                is_tw_ok = (lateness <= 1e-4)

                if not is_tw_ok:
                    violation_record = {
                        "vehicle_id": v_idx + 1,
                        "stop_index": k + 1,
                        "customer_id": v,
                        "arrival_time": round(arrival_time, 2),
                        "due_time": round(d_time, 2),
                        "lateness": round(lateness, 2),
                    }
                    route_violations.append(violation_record)
                    all_time_window_violations.append(violation_record)
                    total_lateness += lateness

                departure_time = service_start + s_time
                curr_time = departure_time
                route_wait += wait_time
                route_service += s_time

                schedule.append(StopSchedule(
                    stop_index=k + 1,
                    customer_id=v,
                    arrival_time=round(arrival_time, 2),
                    ready_time=r_time,
                    due_time=d_time,
                    waiting_time=round(wait_time, 2),
                    service_start=round(service_start, 2),
                    service_time=s_time,
                    departure_time=round(departure_time, 2),
                    is_time_window_feasible=is_tw_ok,
                    lateness=round(lateness, 2),
                    demand=dem,
                    cumulative_load=cum_load,
                ))

            # Depot return check
            depot_return_arrival = schedule[-1].arrival_time
            is_depot_return_ok = (depot_return_arrival <= self.depot["due_time"] + 1e-4)
            if not is_depot_return_ok:
                depot_lateness = depot_return_arrival - self.depot["due_time"]
                violation_record = {
                    "vehicle_id": v_idx + 1,
                    "stop_index": len(route) - 1,
                    "customer_id": 0,
                    "arrival_time": round(depot_return_arrival, 2),
                    "due_time": round(self.depot["due_time"], 2),
                    "lateness": round(depot_lateness, 2),
                }
                route_violations.append(violation_record)
                all_time_window_violations.append(violation_record)
                total_lateness += depot_lateness

            route_eval = VehicleRouteEvaluation(
                vehicle_id=v_idx + 1,
                route=route,
                customer_ids=cust_ids,
                total_distance=round(route_dist, 4),
                total_demand=route_demand,
                capacity_limit=self.capacity,
                is_capacity_feasible=is_cap_ok,
                capacity_overload=round(overload, 2),
                schedule=schedule,
                total_waiting_time=round(route_wait, 2),
                total_service_time=round(route_service, 2),
                total_travel_time=round(route_dist, 2),
                total_duration=round(curr_time, 2),
                return_to_depot_time=round(depot_return_arrival, 2),
                is_depot_return_feasible=is_depot_return_ok,
                is_time_window_feasible=(len(route_violations) == 0),
                time_window_violations_count=len(route_violations),
                violations_detail=route_violations,
                is_route_feasible=is_cap_ok and (len(route_violations) == 0),
            )
            evaluated_routes.append(route_eval)

            total_fleet_distance += route_dist
            total_fleet_travel_time += route_dist
            total_fleet_wait_time += route_wait
            total_fleet_service_time += route_service

        is_capacity_valid = (len(all_capacity_overloads) == 0)
        is_time_window_valid = (len(all_time_window_violations) == 0)
        is_fully_feasible = is_coverage_valid and is_capacity_valid and is_time_window_valid

        total_fleet_duration = total_fleet_travel_time + total_fleet_wait_time + total_fleet_service_time

        if is_fully_feasible:
            status_summary = (
                f"100% FEASIBLE SOLUTION: All {self.expected_total_customers} customers served. "
                f"0 capacity violations, 0 time window violations."
            )
        else:
            reasons = []
            if not is_coverage_valid:
                reasons.append("Coverage incomplete")
            if not is_capacity_valid:
                reasons.append(f"{len(all_capacity_overloads)} vehicle capacity overloads")
            if not is_time_window_valid:
                reasons.append(f"{len(all_time_window_violations)} customer time window violations (Total lateness: {total_lateness:.1f})")
            status_summary = f"INFEASIBLE SOLUTION: {'; '.join(reasons)}."

        return FleetEvaluation(
            dataset_name=solution_name,
            total_customers_expected=self.expected_total_customers,
            total_customers_served=len(all_visited_customers),
            unique_customers_served=len(visited_set),
            missing_customer_ids=missing_customers,
            duplicate_customer_ids=duplicates,
            is_coverage_valid=is_coverage_valid,
            coverage_report=coverage_report,
            vehicles_used=len(normalized_routes),
            vehicle_capacity_limit=self.capacity,
            total_demand_expected=self.expected_total_demand,
            total_demand_delivered=total_fleet_demand,
            is_capacity_valid=is_capacity_valid,
            capacity_violations_count=len(all_capacity_overloads),
            capacity_overloads_detail=all_capacity_overloads,
            total_distance=round(total_fleet_distance, 4),
            total_travel_time=round(total_fleet_travel_time, 4),
            total_waiting_time=round(total_fleet_wait_time, 4),
            total_service_time=round(total_fleet_service_time, 4),
            total_fleet_duration=round(total_fleet_duration, 4),
            is_time_window_valid=is_time_window_valid,
            time_window_violations_count=len(all_time_window_violations),
            total_lateness=round(total_lateness, 4),
            time_window_violations_detail=all_time_window_violations,
            is_fully_feasible=is_fully_feasible,
            status_summary=status_summary,
            routes=evaluated_routes,
        )
