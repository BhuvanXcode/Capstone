"""Full-Scale Classical VRPTW Solver for the 100-Customer Solomon Benchmark.

Solves the complete multi-vehicle, capacity-constrained, time-window-constrained
VRPTW for all 100 customers in Solomon C101 using Google OR-Tools.

CRITICAL SEPARATION PRINCIPLE:
These results represent macroscopic classical fleet operations research.
They are completely separated from the small-scale quantum proof-of-concept simulations
(3 to 5 customers) to maintain absolute scientific integrity and avoid false quantum claims.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

try:
    from .data_loader import load_solomon_c101
except ImportError:
    from data_loader import load_solomon_c101


def solve_full_100_customer_vrptw(
    time_limit_seconds: int = 10,
    vehicle_count: int = 25,
    vehicle_capacity: int = 200,
) -> Dict:
    """Solve the complete 100-customer Solomon C101 problem using OR-Tools."""
    raw_customers = load_solomon_c101()
    customers_list = [raw_customers[cid] for cid in sorted(raw_customers.keys())]
    n_nodes = len(customers_list)  # 101 nodes (Node 0 = depot, 1..100 = customers)

    # Calculate exact distance matrix
    locations = [(c["x"], c["y"]) for c in customers_list]
    distance_matrix = [
        [
            float(np.hypot(locations[i][0] - locations[j][0], locations[i][1] - locations[j][1]))
            for j in range(n_nodes)
        ]
        for i in range(n_nodes)
    ]

    # Integer distance matrix required by OR-Tools
    int_distance_matrix = [
        [int(round(distance_matrix[i][j] * 10.0)) for j in range(n_nodes)]
        for i in range(n_nodes)
    ]

    # Model definition
    manager = pywrapcp.RoutingIndexManager(n_nodes, vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # 1. Distance Callback
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Fixed cost per vehicle to minimize vehicle count
    routing.SetFixedCostOfAllVehicles(10000)

    # 2. Demand & Capacity Dimension
    demands = [int(c["demand"]) for c in customers_list]

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [vehicle_capacity] * vehicle_count,
        True,  # start cumul to zero
        "Capacity",
    )

    # 3. Time Windows Dimension
    service_times = [int(c["service_time"]) for c in customers_list]

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = int(round(distance_matrix[from_node][to_node]))
        return travel_time + service_times[from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        int(raw_customers[0]["due_time"]),  # max waiting time allowed
        int(raw_customers[0]["due_time"]),  # max total route duration
        False,  # Don't force start to zero
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Add time window constraints
    for node_idx, cust in enumerate(customers_list):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetRange(
            int(cust["ready_time"]),
            int(cust["due_time"]),
        )

    for v_id in range(vehicle_count):
        start = routing.Start(v_id)
        end = routing.End(v_id)
        time_dimension.CumulVar(start).SetRange(
            int(customers_list[0]["ready_time"]),
            int(customers_list[0]["due_time"]),
        )
        time_dimension.CumulVar(end).SetRange(
            int(customers_list[0]["ready_time"]),
            int(customers_list[0]["due_time"]),
        )

    # Search parameters
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = time_limit_seconds

    start_timer = time.perf_counter()
    solution = routing.SolveWithParameters(search)
    solve_time = time.perf_counter() - start_timer

    if solution is None:
        raise RuntimeError("OR-Tools failed to find a feasible solution within time limit.")

    # Extract detailed route schedules
    active_routes = []
    total_exact_distance = 0.0
    total_demand_delivered = 0

    for v in range(vehicle_count):
        idx = routing.Start(v)
        route_nodes = []
        route_dist = 0.0
        route_demand = 0

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            next_idx = solution.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(next_idx)

            route_dist += distance_matrix[node][next_node]
            if node != 0:
                route_demand += demands[node]

            route_nodes.append(node)
            idx = next_idx

        route_nodes.append(0)  # Close loop at depot

        if len(route_nodes) > 2:  # Vehicle actually visited customers
            active_routes.append({
                "vehicle_id": v + 1,
                "route": route_nodes,
                "customer_count": len(route_nodes) - 2,
                "distance": round(route_dist, 2),
                "total_demand": route_demand,
                "is_capacity_feasible": route_demand <= vehicle_capacity,
            })
            total_exact_distance += route_dist
            total_demand_delivered += route_demand

    return {
        "status": "CLASSICAL_OPTIMIZATION_SUCCESS",
        "solver": "Google OR-Tools Constraint Solver (Guided Local Search)",
        "dataset": "Solomon C101 Benchmark (100 Customers + Depot)",
        "solve_time_seconds": round(solve_time, 3),
        "total_customers_served": 100,
        "vehicles_used": len(active_routes),
        "available_vehicles": vehicle_count,
        "vehicle_capacity": vehicle_capacity,
        "total_demand_delivered": total_demand_delivered,
        "total_distance": round(total_exact_distance, 2),
        "routes": active_routes,
    }


def plot_full_100_customer_solution(solution: Dict, output_path: Path) -> None:
    """Render a comprehensive visualization of all 100 customer routes across the fleet."""
    raw_customers = load_solomon_c101()
    customers_list = [raw_customers[cid] for cid in sorted(raw_customers.keys())]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    # Plot all customers
    xs = [c["x"] for c in customers_list]
    ys = [c["y"] for c in customers_list]
    ax.scatter(xs[1:], ys[1:], c="#4a90e2", s=40, alpha=0.7, edgecolors="none", label="Customers (1..100)", zorder=3)

    # Plot Depot
    ax.scatter(xs[0], ys[0], c="#d0021b", s=250, marker="s", edgecolors="black", linewidth=1.5, label="Central Depot (0)", zorder=6)
    ax.text(xs[0] + 0.8, ys[0] - 1.2, "Depot", fontsize=10, fontweight="bold", color="#900000")

    # Distinct route colors for up to 15 vehicles
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]

    for i, r_info in enumerate(solution["routes"]):
        route = r_info["route"]
        col = colors[i % len(colors)]
        r_xs = [customers_list[n]["x"] for n in route]
        r_ys = [customers_list[n]["y"] for n in route]

        ax.plot(r_xs, r_ys, color=col, linewidth=1.8, alpha=0.85, linestyle="-", zorder=4)

    # Watermark and Banner
    title = (
        "Full 100-Customer Solomon C101 Benchmark Fleet Solution\n"
        f"Classical OR-Tools Solver: {solution['vehicles_used']} Vehicles | Total Distance: {solution['total_distance']:.1f} | Solve Time: {solution['solve_time_seconds']}s"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("X Coordinate", fontsize=10)
    ax.set_ylabel("Y Coordinate", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Prominent Disclaimer Card
    disclaimer = (
        "CLASSICAL FLEET BENCHMARK (OR-Tools CP-SAT)\n"
        "───────────────────────────────────────────────\n"
        "• Customers:       100 Customers + 1 Depot\n"
        f"• Fleet Size:      {solution['vehicles_used']} Vehicles (Capacity: 200 each)\n"
        f"• Fleet Distance:  {solution['total_distance']:.2f}\n"
        f"• Solve Runtime:   {solution['solve_time_seconds']} seconds\n"
        "• Methodology:     Exact Classical Metaheuristic\n"
        "• Distinction:     SEPARATE FROM QUANTUM POC.\n"
        "                   (100 customers = 10,000 qubits;\n"
        "                   unsolvable on NISQ hardware)."
    )
    ax.text(
        0.02,
        0.98,
        disclaimer,
        transform=ax.transAxes,
        fontsize=8.5,
        family="monospace",
        verticalalignment="top",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#e74c3c", linewidth=1.5, alpha=0.95),
    )

    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
