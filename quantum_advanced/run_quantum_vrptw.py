"""Main Execution Script for the Focused Quantum Routing Demonstration.

Runs an end-to-end demonstrable pipeline:
1. Formulates small customer routing (from Solomon C101) as a QUBO and Ising Hamiltonian.
2. Builds and executes a genuine gate-level QAOA quantum circuit on Qiskit AerSimulator.
3. Optimizes variational parameters (gamma, beta) and records convergence.
4. Decodes measured quantum bitstrings into routing candidates.
5. Validates candidates against model permutation constraints and executes independent
   post-quantum classical VRPTW checks (vehicle capacity and time windows).
6. Solves the exact classical benchmark for comparison.
7. Generates publication-quality plots and writes a comprehensive JSON results report.
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np

try:
    from .data_loader import get_problem_instance
    from .qubo_formulation import RoutingQUBO
    from .qaoa_circuit import build_qaoa_circuit, get_circuit_summary
    from .classical_solver import ExactClassicalSolver
    from .qaoa_optimizer import QAOAOptimizer
    from .decoder_and_validator import CandidateDecoderAndValidator
    from .resource_estimator import ResourceEstimator
    from .visualizer import (
        plot_convergence,
        plot_bitstring_distribution,
        plot_route_comparison,
        save_circuit_visualization,
    )
except ImportError:
    from data_loader import get_problem_instance
    from qubo_formulation import RoutingQUBO
    from qaoa_circuit import build_qaoa_circuit, get_circuit_summary
    from classical_solver import ExactClassicalSolver
    from qaoa_optimizer import QAOAOptimizer
    from decoder_and_validator import CandidateDecoderAndValidator
    from resource_estimator import ResourceEstimator
    from visualizer import (
        plot_convergence,
        plot_bitstring_distribution,
        plot_route_comparison,
        save_circuit_visualization,
    )


def run_demonstration(
    n_customers: int = 3,
    p_layers: int = 1,
    shots: int = 5000,
    penalty: float = 35.0,
    max_iter: int = 50,
    seed: int = 42,
    output_dir: Path = None,
) -> dict:
    """Execute the full quantum routing demonstration pipeline."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("      FOCUSED QUANTUM COMPUTING DEMONSTRATION (QAOA ROUTING)")
    print("      Capstone Project: VRPTW Quantum Proof-of-Concept")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. DEFINE AND LOAD PROBLEM INSTANCE & PRE-FLIGHT CHECK
    # -------------------------------------------------------------
    print(f"\n[1/7] Loading Problem Instance from Solomon C101...")
    instance = get_problem_instance(n_customers=n_customers)
    print(f"  Instance Name:     {instance.name}")
    print(f"  Depot ID:          Node {instance.depot_id} (Coords: {instance.coordinates[0]})")
    print(f"  Customer IDs:      {instance.customer_ids}")
    print(f"  Vehicle Capacity:  {instance.vehicle_capacity}")
    print(f"  Qubits Required:   {instance.n_qubits} (N^2 = {instance.n_customers}^2)")

    # Pre-Flight Resource Gate Check
    profile = ResourceEstimator.estimate(n_customers)
    print(f"  Estimated RAM:     {profile.ram_human} ({profile.hilbert_dimension:,} statevector states)")
    print(f"  Feasible Subspace: {profile.feasible_routes_count} / {profile.hilbert_dimension:,} ({profile.feasible_percent_str})")
    print(f"  Pre-Flight Status: {profile.status_label}")

    if not profile.is_safe:
        print(f"\n" + "!" * 75)
        print(f"  [PRE-FLIGHT GATE ABORT] Quantum simulation unsafe on laptop:")
        print(f"  {profile.recommendation}")
        print(f"!" * 75)
        print(f"\n  Running classical exact permutation solver for benchmark ground truth...")
        classical_solver = ExactClassicalSolver(instance)
        classical_results = classical_solver.solve()
        print(f"  Classical Optimal Distance: {classical_results['optimal_distance']:.4f}")
        print(f"  Classical Optimal Routes:   {classical_results['optimal_routes']}")
        return {
            "n_customers": n_customers,
            "status": "ABORTED_EXCEEDS_MEMORY_LIMIT",
            "resource_profile": profile.__dict__,
            "classical_results": classical_results,
        }

    print("  Distance Matrix (Node 0=Depot, 1..N=Customers):")
    for r in range(len(instance.distance_matrix)):
        row_str = " ".join(f"{val:6.2f}" for val in instance.distance_matrix[r])
        print(f"    Node {r}: [ {row_str} ]")

    # -------------------------------------------------------------
    # 2. EXACT CLASSICAL BENCHMARK SOLVER
    # -------------------------------------------------------------
    print(f"\n[2/7] Running Exact Classical Permutation Solver (Benchmark Ground Truth)...")
    classical_solver = ExactClassicalSolver(instance)
    classical_results = classical_solver.solve()
    exact_best = classical_results["best_evaluation"]

    print(f"  Total Permutations Evaluated: {classical_results['routes_evaluated']}")
    print(f"  Classical Solve Time:         {classical_results['solve_time_seconds'] * 1000:.3f} ms")
    print(f"  Exact Optimal Distance:       {classical_results['optimal_distance']:.4f}")
    print(f"  Exact Optimal Route(s):       {classical_results['optimal_routes']}")
    print(f"  Optimal Route Time Windows:   {'FEASIBLE (0 violations)' if exact_best.is_time_window_feasible else 'VIOLATED'}")
    print(f"  Optimal Route Capacity:       {exact_best.total_demand:.1f} / {exact_best.capacity_limit:.1f} (FEASIBLE)")

    # -------------------------------------------------------------
    # 3. QUBO AND ISING HAMILTONIAN FORMULATION
    # -------------------------------------------------------------
    print(f"\n[3/7] Formulating Problem as QUBO and Mapping to Ising Hamiltonian...")
    qubo = RoutingQUBO(instance, penalty=penalty)
    print(f"  Binary Variables / Qubits:   {qubo.n_qubits}")
    print(f"  Lagrange Penalty Multiplier: {qubo.penalty}")
    print(f"  Ising Constant Offset C0:    {qubo.C0:.4f}")
    print(f"  Ising Pauli Terms Count:     {len(qubo.hamiltonian)}")

    # -------------------------------------------------------------
    # 4. BUILD GENUINE QAOA QUANTUM CIRCUIT
    # -------------------------------------------------------------
    print(f"\n[4/7] Constructing Genuine Gate-Level QAOA Circuit...")
    # Initial test circuit for inspection
    test_qc = build_qaoa_circuit(
        qubo,
        gamma=[0.05] * p_layers,
        beta=[0.35] * p_layers,
        p_layers=p_layers,
        insert_barriers=True,
        measure=True,
    )
    circ_info = get_circuit_summary(test_qc)
    print(f"  Circuit Qubits:      {circ_info['n_qubits']}")
    print(f"  Circuit Depth:       {circ_info['depth']} (Decomposed Depth: {circ_info['decomposed_depth']})")
    print(f"  Gate Counts:         {circ_info['gate_counts']}")
    print(f"  QAOA Architecture:   Hadamard Superposition -> [Cost Unitary (Rz, Rzz) -> Mixer (Rx)] x {p_layers} -> Measure")

    # -------------------------------------------------------------
    # 5. HYBRID VARIATIONAL OPTIMIZATION & AER EXECUTION
    # -------------------------------------------------------------
    print(f"\n[5/7] Executing QAOA Parameter Optimization & AerSimulator Sampling...")
    optimizer = QAOAOptimizer(qubo, p_layers=p_layers)
    opt_result = optimizer.run_optimization(
        initial_params=None,
        max_iter=max_iter,
        shots=shots,
        seed=seed,
    )
    print(f"  Optimizer:                 COBYLA ({opt_result.iterations_count} iterations, {opt_result.optimization_time_seconds:.2f}s)")
    print(f"  Optimizer Success:         {opt_result.optimizer_success}")
    print(f"  Initial Expected Energy:   {opt_result.initial_expected_energy:.4f}")
    print(f"  Optimized Expected Energy: {opt_result.optimized_expected_energy:.4f}")
    print(f"  Optimized Gamma*:          {opt_result.optimized_parameters[:p_layers]}")
    print(f"  Optimized Beta*:           {opt_result.optimized_parameters[p_layers:]}")
    print(f"  Simulation Backend:        {opt_result.simulator_name} ({opt_result.shots} shots, {opt_result.simulation_time_seconds:.2f}s)")
    print(f"  Unique Bitstrings Sampled: {len(opt_result.measurement_counts)}")

    # -------------------------------------------------------------
    # 6. DECODE BITSTRINGS AND VALIDATE CONSTRAINTS
    # -------------------------------------------------------------
    print(f"\n[6/7] Decoding Measured Bitstrings and Validating Constraints...")
    decoder = CandidateDecoderAndValidator(instance, qubo)

    decoded_list = []
    for bstring, count in opt_result.measurement_counts.items():
        cand = decoder.decode_and_validate(bstring, shots=count, total_shots=shots)
        decoded_list.append(cand)

    # Filter feasible permutation routes
    feasible_cands = [c for c in decoded_list if c.is_permutation_feasible]
    feasible_shots = sum(c.shots_count for c in feasible_cands)
    feasible_rate = (feasible_shots / shots) * 100.0

    # Sort feasible by routing distance, breaking ties with VRPTW feasibility
    feasible_cands.sort(key=lambda c: (round(c.routing_distance, 6), not c.vrptw_evaluation.is_vrptw_feasible))

    # Check for optimal routes
    optimal_routes_set = [list(r) for r in classical_results["optimal_routes"]]
    optimal_cands = [c for c in feasible_cands if c.route in optimal_routes_set]
    optimal_shots = sum(c.shots_count for c in optimal_cands)
    optimal_rate = (optimal_shots / shots) * 100.0

    best_qaoa_cand = feasible_cands[0] if feasible_cands else None
    best_qaoa_eval = best_qaoa_cand.vrptw_evaluation if best_qaoa_cand else None

    print(f"  Total Feasible Shots:        {feasible_shots} / {shots} ({feasible_rate:.2f}%)")
    print(f"  Unique Feasible Routes:      {len(feasible_cands)}")
    print(f"  Optimal Sample Shots:        {optimal_shots} / {shots} ({optimal_rate:.2f}%)")
    if best_qaoa_cand:
        print(f"  Best QAOA Route:             {best_qaoa_cand.route}")
        print(f"  Best QAOA Distance:          {best_qaoa_cand.routing_distance:.4f}")
        print(f"  Best QAOA Measured Bitstring:{best_qaoa_cand.bitstring}")
        print(f"  Matches Exact Classical:     {best_qaoa_cand.route in optimal_routes_set}")
        print(f"  Vehicle Capacity Check:      {best_qaoa_eval.total_demand:.1f} / {best_qaoa_eval.capacity_limit:.1f} ({'PASSED' if best_qaoa_eval.is_capacity_feasible else 'FAILED'})")
        print(f"  Time Window Schedule Check:  {'PASSED (0 violations)' if best_qaoa_eval.is_time_window_feasible else f'FAILED ({len(best_qaoa_eval.time_window_violations)} violations)'}")
    else:
        print("  WARNING: No feasible routes sampled. Consider increasing penalty or layers.")

    # Top sampled feasible routes table
    print("\n  Top Sampled Feasible Routes from QAOA:")
    for i, c in enumerate(feasible_cands[:5], start=1):
        is_opt = c.route in optimal_routes_set
        tw_status = "TW: PASS" if c.vrptw_evaluation.is_time_window_feasible else f"TW: {len(c.vrptw_evaluation.time_window_violations)} VIOL"
        print(f"    {i}. Route {c.route} | Dist: {c.routing_distance:6.2f} | Shots: {c.shots_count:4d} ({c.probability*100:5.2f}%) | {tw_status} {'[EXACT OPTIMUM]' if is_opt else ''}")

    # -------------------------------------------------------------
    # 7. GENERATE AND SAVE PLOTS AND REPORT
    # -------------------------------------------------------------
    print(f"\n[7/7] Generating Plots and Saving Machine-Readable Results to '{output_dir}'...")

    conv_plot_path = output_dir / "optimization_convergence.png"
    dist_plot_path = output_dir / "bitstring_distribution.png"
    route_plot_path = output_dir / "route_comparison.png"
    circuit_png_path = output_dir / "circuit_summary.png"
    circuit_txt_path = output_dir / "circuit_diagram.txt"
    json_path = output_dir / "quantum_results.json"

    plot_convergence(opt_result.convergence_history, conv_plot_path)
    plot_bitstring_distribution(decoded_list, optimal_routes_set, dist_plot_path)
    plot_route_comparison(instance, best_qaoa_eval, exact_best, route_plot_path)
    save_circuit_visualization(circ_info["ascii_diagram"], circ_info, opt_result, circuit_txt_path, circuit_png_path)

    # Build JSON summary report
    results_payload = {
        "metadata": {
            "title": "Focused QAOA Routing Demonstration",
            "framework": "Qiskit 2.x & Qiskit Aer",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "Local CPU Simulator (AerSimulator)",
        },
        "problem": {
            "name": instance.name,
            "depot_id": instance.depot_id,
            "customer_ids": instance.customer_ids,
            "n_customers": instance.n_customers,
            "n_qubits": instance.n_qubits,
            "vehicle_capacity": instance.vehicle_capacity,
            "coordinates": instance.coordinates.tolist(),
        },
        "qubo": {
            "penalty_multiplier": penalty,
            "ising_offset": qubo.C0,
            "pauli_terms": len(qubo.hamiltonian),
        },
        "circuit": {
            "p_layers": p_layers,
            "qubits": circ_info["n_qubits"],
            "depth": circ_info["depth"],
            "decomposed_depth": circ_info["decomposed_depth"],
            "gate_counts": circ_info["gate_counts"],
        },
        "optimization": {
            "optimizer": "COBYLA",
            "max_iter": max_iter,
            "iterations_evaluated": opt_result.iterations_count,
            "initial_parameters": opt_result.initial_parameters,
            "optimized_parameters": opt_result.optimized_parameters,
            "initial_energy": opt_result.initial_expected_energy,
            "optimized_energy": opt_result.optimized_expected_energy,
            "runtime_seconds": opt_result.optimization_time_seconds,
        },
        "sampling": {
            "simulator": opt_result.simulator_name,
            "shots": shots,
            "feasible_shots": feasible_shots,
            "feasible_rate_percent": round(feasible_rate, 2),
            "optimal_shots": optimal_shots,
            "optimal_rate_percent": round(optimal_rate, 2),
            "unique_bitstrings": len(opt_result.measurement_counts),
        },
        "qaoa_best_result": {
            "route": best_qaoa_cand.route if best_qaoa_cand else None,
            "distance": best_qaoa_cand.routing_distance if best_qaoa_cand else None,
            "bitstring": best_qaoa_cand.bitstring if best_qaoa_cand else None,
            "matches_exact_classical": bool(best_qaoa_cand.route in optimal_routes_set) if best_qaoa_cand else False,
            "vrptw_capacity_passed": best_qaoa_eval.is_capacity_feasible if best_qaoa_eval else False,
            "vrptw_time_windows_passed": best_qaoa_eval.is_time_window_feasible if best_qaoa_eval else False,
            "time_window_violations": best_qaoa_eval.time_window_violations if best_qaoa_eval else [],
        },
        "exact_classical_benchmark": {
            "optimal_distance": classical_results["optimal_distance"],
            "optimal_routes": classical_results["optimal_routes"],
            "permutations_evaluated": classical_results["routes_evaluated"],
            "solve_time_seconds": classical_results["solve_time_seconds"],
        },
        "generated_artifacts": [
            str(conv_plot_path.name),
            str(dist_plot_path.name),
            str(route_plot_path.name),
            str(circuit_png_path.name),
            str(circuit_txt_path.name),
            str(json_path.name),
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"  Saved: {conv_plot_path.name}")
    print(f"  Saved: {dist_plot_path.name}")
    print(f"  Saved: {route_plot_path.name}")
    print(f"  Saved: {circuit_png_path.name}")
    print(f"  Saved: {circuit_txt_path.name}")
    print(f"  Saved: {json_path.name}")

    print("\n" + "=" * 75)
    print("      DEMONSTRATION RUN SUCCESSFULLY COMPLETED")
    print("=" * 75 + "\n")

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focused QAOA Routing Demonstration & Scaling Analysis")
    parser.add_argument("--customers", type=int, default=3, help="Number of customers (e.g. 3, 4, 5, 6; default: 3)")
    parser.add_argument("--layers", type=int, default=1, help="QAOA p layers (default: 1)")
    parser.add_argument("--shots", type=int, default=5000, help="AerSimulator measurement shots (default: 5000)")
    parser.add_argument("--penalty", type=float, default=35.0, help="QUBO constraint penalty multiplier (default: 35.0)")
    parser.add_argument("--maxiter", type=int, default=50, help="Max COBYLA iterations (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-instance scaling benchmark across 3, 4, 5, 6 customers")
    parser.add_argument("--full-scale", action="store_true", help="Solve full 100-customer Solomon C101 dataset with classical OR-Tools")
    parser.add_argument("--full-dataset", action="store_true", help="Run complete 100-customer workflow (Classical Baseline vs Hybrid QAOA)")
    parser.add_argument("--decompose", action="store_true", help="Run hybrid cluster-first route-second decomposition and encoding analysis")
    parser.add_argument("--estimate", action="store_true", help="Print pre-flight quantum resource scaling table (N=3..100)")

    args = parser.parse_args()

    if args.estimate:
        print("\n" + "=" * 80)
        print("          PRE-FLIGHT QUANTUM HARDWARE RESOURCE ESTIMATION TABLE")
        print("=" * 80)
        table = ResourceEstimator.generate_scaling_comparison_table([3, 4, 5, 6, 8, 10, 100])
        print(f"{'Customers':>10} | {'Qubits':>8} | {'Hilbert States':>18} | {'RAM Required':>14} | {'Feasible Routes':>16} | {'Feasible %':>14} | {'Status':>20}")
        print("-" * 115)
        for row in table:
            print(f"{row['customers']:>10} | {row['qubits']:>8} | {str(row['hilbert_states']):>18} | {row['ram_required']:>14} | {row['feasible_routes']:>16} | {row['feasible_fraction']:>14} | {row['laptop_status']:>20}")
        print("=" * 115 + "\n")

    elif args.full_dataset:
        try:
            from .run_full_dataset import run_full_dataset_pipeline
        except ImportError:
            from run_full_dataset import run_full_dataset_pipeline
        run_full_dataset_pipeline()

    elif args.benchmark:
        try:
            from .scaling_benchmark import run_scaling_benchmark
        except ImportError:
            from scaling_benchmark import run_scaling_benchmark
        run_scaling_benchmark(customer_counts=[3, 4, 5, 6])

    elif args.full_scale:
        try:
            from .full_scale_classical import solve_full_100_customer_vrptw
        except ImportError:
            from full_scale_classical import solve_full_100_customer_vrptw
        solve_full_100_customer_vrptw()

    elif args.decompose:
        try:
            from .decomposition import CompactEncodingAnalyzer, HybridVRPTWDecomposer
            from .data_loader import load_solomon_c101
        except ImportError:
            from decomposition import CompactEncodingAnalyzer, HybridVRPTWDecomposer
            from data_loader import load_solomon_c101

        print("\n" + "=" * 80)
        print("     COMPACT LOGARITHMIC ENCODING VS 1-HOT N^2 ENCODING COMPARISON")
        print("=" * 80)
        table = CompactEncodingAnalyzer.get_comparison_table([3, 4, 5, 6, 8, 10, 20, 100])
        print(f"{'Customers':>10} | {'1-Hot Qubits':>14} | {'Compact Qubits':>14} | {'Qubit Reduction':>16} | {'Coupling Complexity':>34}")
        print("-" * 98)
        for row in table:
            print(f"{row['customers']:>10} | {row['one_hot_qubits']:>14} | {row['compact_qubits']:>14} | {row['qubit_savings']:>16} | {row['tradeoff']:>34}")

        print("\n" + "=" * 80)
        print("     HYBRID POLAR SWEEP DECOMPOSITION (100 CUSTOMERS -> SUBPROBLEMS)")
        print("=" * 80)
        dataset = load_solomon_c101()
        decomposer = HybridVRPTWDecomposer(dataset, vehicle_capacity=200.0)
        clusters = decomposer.decompose_by_sweep(target_cluster_size=4)
        print(f"Partitioned 100 Solomon customers into {len(clusters)} capacity-feasible subproblems:")
        for c in clusters[:8]:
            print(f"  Cluster #{c['cluster_index']:02d}: {c['size']} stops, Demand {c['total_demand']:4.0f}/200.0, Stops: {c['customer_ids']}, Qubits: {c['quantum_subproblem_qubits']} (Quantum Tractable: {c['is_quantum_tractable']})")
        print(f"  ... and {len(clusters) - 8} more clusters.")

    else:
        run_demonstration(
            n_customers=args.customers,
            p_layers=args.layers,
            shots=args.shots,
            penalty=args.penalty,
            max_iter=args.maxiter,
            seed=args.seed,
        )
