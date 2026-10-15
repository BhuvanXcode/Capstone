
def summarize_classical_results(results):
    """Summarize results from the OR-Tools VRPTW solver."""
    routes = results.get("routes", [])

    return {
        "method": "OR-Tools VRPTW",
        "vehicles_used": results.get("vehicles_used"),
        "total_distance": results.get("total_distance"),
        "total_demand_delivered": results.get("total_delivered_demand"),
        "solve_time_seconds": results.get("solve_time_seconds"),
        "routes_count": len(routes),
        "all_routes_within_capacity": all(
            route["load"] <= route["capacity"]
            for route in routes
        ),
    }


def summarize_qaoa_results(results):
    """Summarize QAOA metrics from the saved experiment JSON."""
    qaoa = results.get("qaoa", {})
    exact = results.get("exact_classical", {})

    return {
        "method": "QAOA",
        "problem": results.get("experiment"),
        "dataset": results.get("dataset"),
        "n_customers": results.get("problem", {}).get("n_customers"),
        "n_qubits": results.get("problem", {}).get("n_qubits"),
        "best_route": qaoa.get("best_route"),
        "best_distance": qaoa.get("best_route_distance"),
        "feasible_samples": qaoa.get("feasible_samples"),
        "total_shots": qaoa.get("shots"),
        "feasible_sample_rate_percent": qaoa.get(
            "feasible_sample_rate_percent"
        ),
        "optimal_samples": qaoa.get("optimal_samples"),
        "optimal_sample_rate_percent": qaoa.get(
            "optimal_sample_rate_percent_of_all_shots"
        ),
        "exact_optimal_distance": exact.get("optimal_distance"),
        "matches_exact_optimum": (
            qaoa.get("best_route_distance") == exact.get("optimal_distance")
            if qaoa.get("best_route_distance") is not None
            and exact.get("optimal_distance") is not None
            else None
        ),
    }
