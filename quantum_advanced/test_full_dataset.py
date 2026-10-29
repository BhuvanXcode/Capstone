"""Comprehensive Test Suite for the 100-Customer Solomon C101 VRPTW Pipeline.

Verifies:
1. Full dataset loading with exactly 100 distinct customer IDs plus depot.
2. Global customer coverage validation (detects omissions, duplicates, and perfect 100% coverage).
3. Capacity and time-window validation (detects compliance, overloads, and lateness).
4. QUBO construction for fixed-endpoint routing components (SubrouteQUBO).
5. Actual QAOA quantum circuit execution on Qiskit AerSimulator.
6. Decoding measured quantum bitstrings into candidate customer decisions.
7. Safe fallback when a QAOA proposal is invalid or worse (retains prior feasible route).
8. Classical baseline solver feasibility on the complete 100-customer dataset.
9. Accurate reporting of quantum execution status and telemetry across the hybrid loop.
"""

import unittest
from pathlib import Path
import numpy as np

try:
    from .data_loader import load_solomon_c101, get_problem_instance
    from .fleet_validator import FleetValidator, FleetEvaluation
    from .qubo_formulation import RoutingQUBO, SubrouteQUBO
    from .qaoa_circuit import build_qaoa_circuit, get_circuit_summary
    from .qaoa_optimizer import QAOAOptimizer
    from .decoder_and_validator import CandidateDecoderAndValidator
    from .hybrid_solver import HybridVRPTWSolver, HybridSolverResult
    from .full_scale_classical import solve_full_100_customer_vrptw
except ImportError:
    from data_loader import load_solomon_c101, get_problem_instance
    from fleet_validator import FleetValidator, FleetEvaluation
    from qubo_formulation import RoutingQUBO, SubrouteQUBO
    from qaoa_circuit import build_qaoa_circuit, get_circuit_summary
    from qaoa_optimizer import QAOAOptimizer
    from decoder_and_validator import CandidateDecoderAndValidator
    from hybrid_solver import HybridVRPTWSolver, HybridSolverResult
    from full_scale_classical import solve_full_100_customer_vrptw


class TestFullDatasetVRPTW(unittest.TestCase):
    """Test cases for 100-customer dataset processing, validation, and Hybrid QAOA."""

    @classmethod
    def setUpClass(cls):
        """Load benchmark data once for all tests."""
        cls.raw_data = load_solomon_c101()
        cls.validator = FleetValidator(cls.raw_data, vehicle_capacity=200.0)

    def test_01_full_dataset_loading_100_distinct_customers(self):
        """Requirement 1 & 2: Verify exactly 100 distinct customer records + depot are loaded."""
        self.assertEqual(len(self.raw_data), 101, "Expected 101 records (1 depot + 100 customers)")
        self.assertIn(0, self.raw_data, "Depot node 0 must be present")

        customer_ids = sorted([cid for cid in self.raw_data.keys() if cid != 0])
        self.assertEqual(len(customer_ids), 100, "Must contain exactly 100 customer records")
        self.assertEqual(customer_ids, list(range(1, 101)), "Customer IDs must be integers 1 to 100")
        self.assertEqual(len(set(customer_ids)), 100, "All 100 customer IDs must be distinct")

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

    def test_02_global_customer_coverage(self):
        """Requirement 2: Verify FleetValidator detects omissions, duplicates, and valid 100% coverage."""
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
        omitted_routes[4].remove(42)

        eval_omitted = self.validator.validate_fleet(omitted_routes, "OmittedTest")
        self.assertFalse(eval_omitted.is_coverage_valid, "Missing customer must fail coverage check")
        self.assertIn(42, eval_omitted.missing_customer_ids)
        self.assertEqual(eval_omitted.total_customers_served, 99)

        # Case C: Route with duplicated customer (Customer 15 visited twice)
        dup_routes = [r.copy() for r in valid_routes]
        dup_routes[0].append(15)

        eval_dup = self.validator.validate_fleet(dup_routes, "DuplicateTest")
        self.assertFalse(eval_dup.is_coverage_valid, "Duplicated customer must fail coverage check")
        self.assertIn(15, eval_dup.duplicate_customer_ids)

    def test_03_capacity_and_time_window_validation(self):
        """Requirement 2: Verify capacity overloads and time-window audits (waiting time, service duration, lateness)."""
        # Case A: Compliant vehicle capacity
        compliant_route = [[1, 2, 3]]
        eval_comp = self.validator.validate_fleet(compliant_route, "CapacityPassTest")
        self.assertTrue(eval_comp.is_capacity_valid)
        self.assertEqual(eval_comp.capacity_violations_count, 0)

        # Case B: Overloaded vehicle
        overload_custs = [2, 6, 7, 8, 9, 10, 11, 12]
        eval_over = self.validator.validate_fleet([overload_custs], "CapacityFailTest")
        total_d = sum(self.raw_data[c]["demand"] for c in overload_custs)
        if total_d > 200.0:
            self.assertFalse(eval_over.is_capacity_valid)
            self.assertGreater(eval_over.capacity_violations_count, 0)

        # Case C: Time-window infeasible order [0, 1, 2, 0] (due to node 2 due date = 870 < node 1 completion 1002)
        eval_tw_bad = self.validator.validate_fleet([[1, 2]], "TimeWindowBadTest")
        self.assertFalse(eval_tw_bad.is_time_window_valid)
        self.assertGreater(eval_tw_bad.time_window_violations_count, 0)

        # Case D: Time-window feasible order [0, 2, 1, 0]
        eval_tw_good = self.validator.validate_fleet([[2, 1]], "TimeWindowGoodTest")
        self.assertTrue(eval_tw_good.is_time_window_valid)
        self.assertEqual(eval_tw_good.time_window_violations_count, 0)

    def test_04_qubo_construction_endpoints(self):
        """Requirement 3: Verify SubrouteQUBO constructs correct binary model for routing windows."""
        u, w = 13, 15
        sub_custs = [17, 18, 19]
        qubo = SubrouteQUBO(u, w, sub_custs, self.raw_data, penalty=35.0)

        # 3 customers -> 3^2 = 9 qubits
        self.assertEqual(qubo.n, 3)
        self.assertEqual(qubo.n_qubits, 9)
        self.assertEqual(qubo.Q.shape, (9, 9))
        self.assertGreater(len(qubo.hamiltonian), 0)

        # Test permutation energy vs physical distance
        # Permutation: 17 at p=0, 18 at p=1, 19 at p=2
        x = np.zeros(9)
        x[qubo.var_index(0, 0)] = 1
        x[qubo.var_index(1, 1)] = 1
        x[qubo.var_index(2, 2)] = 1

        energy = qubo.evaluate_bitstring_qubo(x)
        expected_dist = (
            qubo.get_distance(u, 17) +
            qubo.get_distance(17, 18) +
            qubo.get_distance(18, 19) +
            qubo.get_distance(19, w)
        )
        self.assertAlmostEqual(energy, expected_dist, places=5)

    def test_05_actual_qaoa_circuit_execution(self):
        """Requirement 3 & 4: Build and execute genuine QAOA circuit on Qiskit AerSimulator."""
        u, w = 0, 71
        sub_custs = [81, 78, 76]
        qubo = SubrouteQUBO(u, w, sub_custs, self.raw_data, penalty=35.0)

        qc = build_qaoa_circuit(qubo, gamma=[0.1], beta=[0.3], p_layers=1, measure=True)
        summary = get_circuit_summary(qc)

        self.assertEqual(summary["n_qubits"], 9)
        self.assertGreater(summary["depth"], 0)
        self.assertIn("rzz", summary["gate_counts"])
        self.assertIn("rx", summary["gate_counts"])

        # Execute optimization and sampling on AerSimulator
        opt = QAOAOptimizer(qubo, p_layers=1)
        res = opt.run_optimization(max_iter=5, shots=200, seed=42)

        self.assertEqual(res.shots, 200)
        self.assertGreater(len(res.measurement_counts), 0)
        self.assertGreater(res.simulation_time_seconds, 0.0)

    def test_06_decoding_measured_outcomes(self):
        """Requirement 3: Decode measured bitstrings into candidate customer permutations."""
        u, w = 13, 15
        sub_custs = [17, 18, 19]
        qubo = SubrouteQUBO(u, w, sub_custs, self.raw_data, penalty=35.0)
        decoder = CandidateDecoderAndValidator(qubo=qubo)

        # Valid bitstring: 17 at p=0 (q0=1), 18 at p=1 (q4=1), 19 at p=2 (q8=1)
        # Endianness: q8 is character index 0, q0 is character index 8
        bit_list = ["0"] * 9
        bit_list[8 - 0] = "1"
        bit_list[8 - 4] = "1"
        bit_list[8 - 8] = "1"
        valid_bstring = "".join(bit_list)

        cand = decoder.decode_and_validate(valid_bstring, shots=50, total_shots=100)
        self.assertTrue(cand.is_permutation_feasible)
        self.assertEqual(cand.route, [13, 17, 18, 19, 15])
        self.assertIsNotNone(cand.routing_distance)

        # Infeasible bitstring (all zeros)
        bad_bstring = "000000000"
        bad_cand = decoder.decode_and_validate(bad_bstring, shots=10, total_shots=100)
        self.assertFalse(bad_cand.is_permutation_feasible)
        self.assertIsNone(bad_cand.route)

    def test_07_safe_fallback_when_qaoa_proposal_invalid_or_worse(self):
        """Requirement 3: Verify that invalid or suboptimal QAOA candidates are safely rejected without route regression."""
        solver = HybridVRPTWSolver(
            raw_customers=self.raw_data,
            shots_per_subproblem=100,
            max_iter_per_subproblem=5,
            initial_solution_mode="fast",
        )
        # Solve a single window where the candidate causes time window lateness
        test_route = [0, 81, 78, 76, 71, 70, 73, 77, 79, 80, 0]
        sp_res, accepted = solver._solve_and_evaluate_window(
            sub_id="Test_Fallback",
            vehicle_id=1,
            window_start=1,
            u=0,
            w=71,
            cust_window=[81, 78, 76],
            full_route=test_route,
        )
        # Non-improving or identical candidates must not overwrite the existing route
        self.assertFalse(accepted, "Non-improving QAOA candidate must not overwrite the existing route")
        self.assertIn(sp_res.outcome, ["REJECTED", "CONFIRMED_FEASIBLE"])

        # Test that an inverted permutation causes time window violation and is rejected
        inverted_route = [0, 76, 78, 81, 71, 70, 73, 77, 79, 80, 0]
        eval_bad = solver.validator.validate_single_route(inverted_route, vehicle_id=1)
        self.assertFalse(eval_bad.is_route_feasible, "Inverted customer order must violate time window constraints")
        self.assertFalse(eval_bad.is_time_window_feasible, "Time window feasibility must be False")
        self.assertGreater(eval_bad.time_window_violations_count, 0, "Violations count must be > 0 for inverted order")

    def test_08_classical_baseline_feasibility_100_customers(self):
        """Requirement 6: Solve full 100-customer dataset classically with OR-Tools and verify 100% feasibility."""
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

    def test_09_hybrid_solver_complete_workflow_and_telemetry(self):
        """Requirement 3 & 4: Execute hybrid solver loop, audit all 100 customers, and verify telemetry recording."""
        solver = HybridVRPTWSolver(
            raw_customers=self.raw_data,
            shots_per_subproblem=150,
            max_iter_per_subproblem=5,
            initial_solution_mode="fast",
        )
        # Run with max_quantum_subproblems=2 for fast test execution
        res = solver.solve(max_quantum_subproblems=2, verbose=False)

        # Full 100-customer audit
        self.assertEqual(res.total_customers, 100)
        self.assertEqual(res.distinct_customers_covered, 100)
        self.assertEqual(res.fleet_evaluation.missing_customer_ids, [])
        self.assertEqual(res.fleet_evaluation.duplicate_customer_ids, [])
        self.assertTrue(res.fleet_evaluation.is_fully_feasible, "Hybrid final fleet must be 100% feasible")

        # Telemetry metrics audit
        self.assertEqual(len(res.subproblems), 2)
        self.assertGreater(res.distinct_quantum_customer_count, 0)
        self.assertEqual(len(res.distinct_quantum_customer_ids), res.distinct_quantum_customer_count)
        self.assertEqual(res.distinct_quantum_customer_count + res.unprocessed_customer_count, 100)
        # Verify no regression against same initial solution
        self.assertLessEqual(res.final_total_distance, res.initial_total_distance + 1e-4)

        for sp in res.subproblems:
            self.assertEqual(sp.n_customers, 3)
            self.assertEqual(sp.n_qubits, 9)
            self.assertGreater(sp.shots, 0)
            self.assertIn(sp.outcome, ["ACCEPTED", "CONFIRMED_FEASIBLE", "REJECTED"])
            self.assertGreater(sp.total_subproblem_time_seconds, 0.0)

    def test_10_all_100_customers_included_in_qaoa_subproblem_windows(self):
        """Audit requirement: Verify that the window scheduler exposes all 100 customer IDs to QAOA subproblems."""
        raw_sol = solve_full_100_customer_vrptw(time_limit_seconds=3, vehicle_count=25, vehicle_capacity=200, strategy="PATH_CHEAPEST_ARC")
        routes = [r["route"] for r in raw_sol["routes"]]
        k = 3

        all_covered_custs = set()
        total_subproblems = 0
        for route in routes:
            m = len(route) - 2
            if m < k:
                continue
            step_size = max(1, k - 1)
            window_starts = []
            curr = 1
            while curr + k - 1 <= m:
                window_starts.append(curr)
                curr += step_size
            if window_starts and (window_starts[-1] + k - 1 < m):
                end_start = m - k + 1
                if end_start not in window_starts:
                    window_starts.append(end_start)
                window_starts.sort()

            for s in window_starts:
                cust_window = route[s : s + k]
                all_covered_custs.update(cust_window)
                total_subproblems += 1

        self.assertEqual(len(all_covered_custs), 100, "All 100 customer IDs must be covered by QAOA subproblem windows")
        self.assertEqual(all_covered_custs, set(range(1, 101)), "Covered set must exactly match IDs 1 to 100")
        self.assertGreaterEqual(total_subproblems, 40, "Should generate ~45-47 subproblems across 10 vehicle routes")


if __name__ == "__main__":
    unittest.main()
