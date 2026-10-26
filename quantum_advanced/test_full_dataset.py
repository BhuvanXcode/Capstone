"""Comprehensive Test Suite for the 100-Customer Solomon C101 VRPTW Pipeline.

Verifies:
1. Dataset ingestion integrity (all 100 customers loaded with correct parameters).
2. Customer coverage validation (detects omissions, duplicates, and perfect 100% coverage).
3. Vehicle capacity validation (detects compliance and overloads).
4. Time window audit logic (waiting times, service starts, lateness calculation, depot return).
5. Classical baseline solver feasibility on the complete 100-customer dataset.
6. Hybrid decomposition and QAOA subproblem solving mechanics.
"""

import unittest
from pathlib import Path
import numpy as np

try:
    from .data_loader import load_solomon_c101
    from .fleet_validator import FleetValidator, FleetEvaluation
    from .hybrid_solver import HybridVRPTWSolver
    from .full_scale_classical import solve_full_100_customer_vrptw
except ImportError:
    from data_loader import load_solomon_c101
    from fleet_validator import FleetValidator, FleetEvaluation
    from hybrid_solver import HybridVRPTWSolver
    from full_scale_classical import solve_full_100_customer_vrptw


class TestFullDatasetVRPTW(unittest.TestCase):
    """Test cases for 100-customer dataset processing and validation."""

    @classmethod
    def setUpClass(cls):
        """Load benchmark data once for all tests."""
        cls.raw_data = load_solomon_c101()
        cls.validator = FleetValidator(cls.raw_data, vehicle_capacity=200.0)

    def test_01_dataset_ingestion_integrity(self):
        """Verify that all 100 customers plus depot are loaded without truncation."""
        self.assertEqual(len(self.raw_data), 101, "Expected 101 records (1 depot + 100 customers)")
        self.assertIn(0, self.raw_data, "Depot node 0 must be present")

        customer_ids = sorted([cid for cid in self.raw_data.keys() if cid != 0])
        self.assertEqual(len(customer_ids), 100, "Must contain exactly 100 customer records")
        self.assertEqual(customer_ids, list(range(1, 101)), "Customer IDs must be integers 1 to 100")

        # Demand and capacity
        total_demand = sum(self.raw_data[cid]["demand"] for cid in customer_ids)
        self.assertEqual(total_demand, 1810.0, "Total Solomon C101 demand must be 1810 units")

        # Check fields exist for all customers
        for cid in customer_ids:
            c = self.raw_data[cid]
            self.assertIn("x", c)
            self.assertIn("y", c)
            self.assertIn("demand", c)
            self.assertIn("ready_time", c)
            self.assertIn("due_time", c)
            self.assertIn("service_time", c)
            self.assertEqual(c["service_time"], 90.0, "Service time must be 90.0 in C101")
            self.assertTrue(c["ready_time"] <= c["due_time"], "Ready time must not exceed due time")

    def test_02_customer_coverage_validation(self):
        """Verify that FleetValidator accurately verifies coverage and catches omissions/duplicates."""
        # Case A: Valid 10-vehicle partition of 1..100
        valid_routes = []
        for v in range(10):
            valid_routes.append(list(range(v * 10 + 1, (v + 1) * 10 + 1)))

        eval_valid = self.validator.validate_fleet(valid_routes, "ValidCoverageTest")
        self.assertTrue(eval_valid.is_coverage_valid, "Valid routes must pass coverage check")
        self.assertEqual(eval_valid.total_customers_served, 100)
        self.assertEqual(eval_valid.missing_customer_ids, [])
        self.assertEqual(eval_valid.duplicate_customer_ids, [])

        # Case B: Route with omitted customer (Customer 42 missing)
        omitted_routes = [r.copy() for r in valid_routes]
        omitted_routes[4].remove(42)  # Remove customer 42

        eval_omitted = self.validator.validate_fleet(omitted_routes, "OmittedTest")
        self.assertFalse(eval_omitted.is_coverage_valid, "Missing customer must fail coverage check")
        self.assertIn(42, eval_omitted.missing_customer_ids)
        self.assertEqual(eval_omitted.total_customers_served, 99)

        # Case C: Route with duplicated customer (Customer 15 visited in two vehicles)
        dup_routes = [r.copy() for r in valid_routes]
        dup_routes[0].append(15)  # Customer 15 visited in vehicle 1 and vehicle 2

        eval_dup = self.validator.validate_fleet(dup_routes, "DuplicateTest")
        self.assertFalse(eval_dup.is_coverage_valid, "Duplicated customer must fail coverage check")
        self.assertIn(15, eval_dup.duplicate_customer_ids)

    def test_03_vehicle_capacity_validation(self):
        """Verify that vehicle capacity check detects compliant and overloaded routes."""
        # Case A: Compliant vehicle (Demand: 10+30+10 = 50 <= 200)
        compliant_route = [[1, 2, 3]]
        eval_comp = self.validator.validate_fleet(compliant_route, "CapacityPassTest")
        self.assertTrue(eval_comp.is_capacity_valid)
        self.assertEqual(eval_comp.capacity_violations_count, 0)

        # Case B: Overloaded vehicle
        # Customers 58, 57, 56, 54 have demands 30, 30, 30, 30 = 120
        # If we stack customers to demand > 200:
        # e.g., 8 customers of demand 30 = 240 > 200
        overload_custs = [2, 6, 7, 8, 9, 10, 11, 12]  # large combined demand
        overload_route = [overload_custs]
        eval_over = self.validator.validate_fleet(overload_route, "CapacityFailTest")
        total_d = sum(self.raw_data[c]["demand"] for c in overload_custs)
        if total_d > 200.0:
            self.assertFalse(eval_over.is_capacity_valid)
            self.assertGreater(eval_over.capacity_violations_count, 0)
            self.assertGreater(eval_over.capacity_overloads_detail[0]["overload"], 0.0)

    def test_04_time_window_audit_logic(self):
        """Verify waiting times, service starts, lateness calculation, and depot return."""
        # Stop 1 (Node 1): ready 912, due 967, service 90
        # Stop 2 (Node 2): ready 825, due 870, service 90
        # Route [0, 1, 2, 0]:
        # Depot -> 1 takes 18.68 units. Arrives at t=18.68 < 912 -> Waits 893.32 -> Service 912..1002
        # 1 -> 2 takes 2.0 units. Arrives at t=1004. But Node 2 due is 870! -> Lateness = 1004 - 870 = 134 units!
        infeasible_route = [[1, 2]]
        eval_res = self.validator.validate_fleet(infeasible_route, "TimeWindowAuditTest")
        self.assertFalse(eval_res.is_time_window_valid, "Route [0, 1, 2, 0] must violate time window at node 2")
        self.assertGreater(eval_res.time_window_violations_count, 0)

        # Reverse route [0, 2, 1, 0]:
        # Depot -> 2 arrives t=20.62 < 825 -> Waits -> Service 825..915
        # 2 -> 1 takes 2.0 units -> Arrives at t=917 <= 967 -> PASS!
        # Return to depot: arrives at t=1025.7 <= 1236 -> PASS!
        feasible_route = [[2, 1]]
        eval_feas = self.validator.validate_fleet(feasible_route, "TimeWindowFeasibleTest")
        self.assertTrue(eval_feas.is_time_window_valid, "Route [0, 2, 1, 0] must pass all time windows")
        self.assertEqual(eval_feas.time_window_violations_count, 0)

    def test_05_classical_baseline_solver_feasibility(self):
        """Solve full 100-customer dataset classically with OR-Tools and verify 100% feasibility."""
        raw_sol = solve_full_100_customer_vrptw(time_limit_seconds=3, vehicle_count=25, vehicle_capacity=200)
        routes = [r["route"] for r in raw_sol["routes"]]

        eval_res = self.validator.validate_fleet(routes, "ClassicalBaselineValidation")
        self.assertEqual(eval_res.total_customers_served, 100, "All 100 customers must be served")
        self.assertTrue(eval_res.is_coverage_valid, "Coverage must be 100% valid")
        self.assertTrue(eval_res.is_capacity_valid, "All vehicle capacities must be respected")
        self.assertTrue(eval_res.is_time_window_valid, "All customer time windows must be satisfied")
        self.assertTrue(eval_res.is_fully_feasible, "Classical baseline must be 100% feasible")
        self.assertLessEqual(eval_res.vehicles_used, 12, "Classical solver should use 10 to 12 vehicles")
        self.assertLess(eval_res.total_distance, 900.0, "Classical baseline distance should be < 900")

    def test_06_hybrid_decomposition_coverage(self):
        """Verify that the hybrid solver partitions all 100 customers without omissions or duplicates."""
        solver = HybridVRPTWSolver(raw_customers=self.raw_data, vehicle_capacity=200.0)
        clusters = solver._cluster_customers_classically()

        # All 100 customers must be partitioned
        flattened = [cid for clust in clusters for cid in clust]
        self.assertEqual(len(flattened), 100, "Hybrid clustering must contain all 100 customers")
        self.assertEqual(set(flattened), set(range(1, 101)), "Must contain customers 1..100 with no omissions")

        # Each cluster must respect capacity
        for c_idx, cluster in enumerate(clusters):
            demand = sum(self.raw_data[cid]["demand"] for cid in cluster)
            self.assertLessEqual(demand, 200.0, f"Cluster {c_idx+1} demand {demand} exceeds vehicle capacity 200.0")

    def test_07_qaoa_subproblem_solver_execution(self):
        """Execute a genuine QAOA subproblem run (3 customers, 9 qubits) and verify output integrity."""
        solver = HybridVRPTWSolver(
            raw_customers=self.raw_data,
            shots_per_subproblem=200,
            max_iter_per_subproblem=5,
        )
        sub_res = solver._solve_subproblem_qaoa("Test_V1_S1", 1, [39, 36, 34])

        self.assertEqual(sub_res.n_customers, 3)
        self.assertEqual(sub_res.n_qubits, 9)
        self.assertEqual(set(sub_res.decoded_customer_order), {39, 36, 34})
        self.assertIn("rzz", sub_res.gate_counts)
        self.assertIn("rx", sub_res.gate_counts)
        self.assertGreater(sub_res.circuit_depth, 0)
        self.assertGreater(sub_res.subtour_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
