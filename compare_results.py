
import json
from pathlib import Path

from src.evaluation.metrics import (
    summarize_classical_results,
    summarize_qaoa_results,
)

PROJECT_ROOT = Path(__file__).resolve().parent

CLASSICAL_FILE = PROJECT_ROOT / "outputs" / "ortools_routes.json"
QUANTUM_FILE = PROJECT_ROOT / "outputs" / "qaoa_tsp_results.json"


def load_json(file_path):
    """Load results from a JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    classical_data = load_json(CLASSICAL_FILE)
    quantum_data = load_json(QUANTUM_FILE)

    classical = summarize_classical_results(classical_data)
    quantum = summarize_qaoa_results(quantum_data)

    print("\n========== CAPSTONE RESULTS COMPARISON ==========")

    print("\nCLASSICAL: OR-Tools VRPTW")
    print(f"Vehicles used: {classical['vehicles_used']}")
    print(f"Total distance: {classical['total_distance']}")
    print(
        "Total demand delivered: "
        f"{classical['total_demand_delivered']}"
    )
    print(f"Solve time: {classical['solve_time_seconds']} seconds")
    print(f"Routes: {classical['routes_count']}")
    print(
        "All routes within capacity: "
        f"{classical['all_routes_within_capacity']}"
    )

    print("\nQUANTUM: QAOA TSP")
    print(f"Customers: {quantum['n_customers']}")
    print(f"Qubits: {quantum['n_qubits']}")
    print(f"Best route: {quantum['best_route']}")
    print(f"Best distance: {quantum['best_distance']}")
    print(f"Shots: {quantum['total_shots']}")
    print(
        "Feasible sample rate: "
        f"{quantum['feasible_sample_rate_percent']}%"
    )
    print(
        "Optimal sample rate: "
        f"{quantum['optimal_sample_rate_percent']}%"
    )
    print(
        "Matches exact optimum: "
        f"{quantum['matches_exact_optimum']}"
    )

    print("\n========== INTERPRETATION ==========")
    print(
        "The classical experiment solves a 100-customer VRPTW "
        "with vehicle capacity and time windows."
    )
    print(
        "The quantum experiment solves a separate synthetic "
        "TSP with 3 customers and no VRPTW constraints."
    )
    print(
        "Their distances and solve times are not directly "
        "comparable because the problem definitions differ."
    )

    print("\nComparison report completed successfully.")


if __name__ == "__main__":
    main()
