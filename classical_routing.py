
import json
import math
import time
from pathlib import Path

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "c101.txt"
OUTPUT_FILE = ROOT / "outputs" / "ortools_routes.json"

# Set to 0 for distance-only optimization.
# A positive fixed cost discourages activating extra vehicles.
VEHICLE_FIXED_COST = 1000
SEARCH_TIME_SECONDS = 15


# --------------------------------------------------
# 1. Load the Solomon C101 dataset
# --------------------------------------------------
def load_solomon_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    lines = file_path.read_text(encoding="utf-8-sig").splitlines()

    vehicle_header = next(
        (
            i for i, line in enumerate(lines)
            if "NUMBER" in line.upper()
            and "CAPACITY" in line.upper()
        ),
        None,
    )

    if vehicle_header is None:
        raise ValueError("Vehicle specification header not found.")

    vehicle_info = None

    for line in lines[vehicle_header + 1:]:
        parts = line.split()

        if len(parts) >= 2 and parts[0].isdigit():
            try:
                vehicle_info = (int(parts[0]), int(float(parts[1])))
                break
            except ValueError:
                continue

    if vehicle_info is None:
        raise ValueError("Could not read vehicle count and capacity.")

    vehicle_count, vehicle_capacity = vehicle_info

    customers = []

    for line in lines:
        parts = line.split()

        # A standard Solomon customer record has seven fields.
        if len(parts) != 7 or not parts[0].isdigit():
            continue

        try:
            customers.append({
                "id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "demand": int(float(parts[3])),
                "ready_time": int(float(parts[4])),
                "due_time": int(float(parts[5])),
                "service_time": int(float(parts[6])),
            })
        except ValueError:
            continue

    if not customers:
        raise ValueError("No customer records were loaded.")

    ids = [c["id"] for c in customers]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate customer IDs found.")

    if 0 not in ids:
        raise ValueError("Depot with ID 0 was not found.")

    depot_index = ids.index(0)

    # This baseline assumes one common capacity for all vehicles.
    if any(c["demand"] < 0 for c in customers):
        raise ValueError("Negative customer demand is unsupported.")

    if any(c["demand"] > vehicle_capacity for c in customers):
        raise ValueError("A customer demand exceeds vehicle capacity.")

    total_demand = sum(c["demand"] for c in customers if c["id"] != 0)

    if total_demand > vehicle_count * vehicle_capacity:
        raise ValueError("Total demand exceeds the available fleet capacity.")

    return {
        "customers": customers,
        "depot": depot_index,
        "vehicle_count": vehicle_count,
        "vehicle_capacities": [vehicle_capacity] * vehicle_count,
    }


# --------------------------------------------------
# 2. Build the distance matrix
# --------------------------------------------------
def build_distance_matrix(customers):
    locations = [(c["x"], c["y"]) for c in customers]
    n = len(locations)

    # OR-Tools routing costs must be integer-valued.
    # Rounded Euclidean units are used consistently here.
    return [
        [
            int(round(math.hypot(
                locations[i][0] - locations[j][0],
                locations[i][1] - locations[j][1],
            )))
            for j in range(n)
        ]
        for i in range(n)
    ]


# --------------------------------------------------
# 3. Optimize the Vehicle Routing Problem with
#    Time Windows (VRPTW)
# --------------------------------------------------
def solve_vrptw(data):
    customers = data["customers"]
    demands = [c["demand"] for c in customers]
    capacities = data["vehicle_capacities"]
    depot = data["depot"]

    n = len(customers)
    vehicle_count = data["vehicle_count"]

    distances = build_distance_matrix(customers)

    manager = pywrapcp.RoutingIndexManager(
        n,
        vehicle_count,
        depot,
    )

    routing = pywrapcp.RoutingModel(manager)

    # Travel-distance objective
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distances[from_node][to_node]

    distance_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        distance_callback_index
    )

    routing.SetFixedCostOfAllVehicles(VEHICLE_FIXED_COST)

    # Vehicle capacity constraints
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        capacities,
        True,
        "Capacity",
    )

    # Time windows:
    # time at a node = service-start time.
    # Each arc adds service at its origin plus travel time.
    max_time = max(c["due_time"] for c in customers)
    max_service = max(c["service_time"] for c in customers)
    horizon = max_time + max_service

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time = distances[from_node][to_node]
        service_time = customers[from_node]["service_time"]

        return travel_time + service_time

    time_callback_index = routing.RegisterTransitCallback(
        time_callback
    )

    routing.AddDimension(
        time_callback_index,
        horizon,       # Maximum waiting time per transition
        horizon,       # Maximum cumulative route time
        False,         # Allow flexible departure time
        "Time",
    )

    time_dimension = routing.GetDimensionOrDie("Time")

    # Apply each customer's service time window.
    for node, customer in enumerate(customers):
        if node == depot:
            continue

        index = manager.NodeToIndex(node)

        time_dimension.CumulVar(index).SetRange(
            customer["ready_time"],
            customer["due_time"],
        )

    depot_data = customers[depot]

    # Apply depot time window to all vehicle starts and ends.
    for vehicle_id in range(vehicle_count):
        start = routing.Start(vehicle_id)
        end = routing.End(vehicle_id)

        time_dimension.CumulVar(start).SetRange(
            depot_data["ready_time"],
            depot_data["due_time"],
        )

        time_dimension.CumulVar(end).SetRange(
            depot_data["ready_time"],
            depot_data["due_time"],
        )

    # Search configuration
    search = pywrapcp.DefaultRoutingSearchParameters()

    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search.time_limit.seconds = SEARCH_TIME_SECONDS

    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search)
    solve_time = time.perf_counter() - start_time

    if solution is None:
        print("No feasible solution found within the search settings.")
        return None

    # --------------------------------------------------
    # 4. Extract routes and independently verify metrics
    # --------------------------------------------------
    routes = []
    visited = []
    total_distance = 0
    total_load = 0
    used_vehicles = 0

    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)

        route_ids = [customers[depot]["id"]]
        customer_nodes = []
        route_distance = 0
        route_load = 0
        schedule = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            route_distance += distances[node][next_node]

            if node != depot:
                customer = customers[node]

                route_load += demands[node]
                visited.append(node)
                customer_nodes.append(node)

                schedule.append({
                    "customer_id": customer["id"],
                    "service_start": solution.Value(
                        time_dimension.CumulVar(index)
                    ),
                    "ready_time": customer["ready_time"],
                    "due_time": customer["due_time"],
                })

            index = next_index
            route_ids.append(customers[next_node]["id"])

        # Ignore unused vehicles in the displayed results.
        if not customer_nodes:
            continue

        used_vehicles += 1
        total_distance += route_distance
        total_load += route_load

        routes.append({
            "vehicle": vehicle_id + 1,
            "route": route_ids,
            "distance": route_distance,
            "load": route_load,
            "capacity": capacities[vehicle_id],
            "schedule": schedule,
        })

    expected = set(range(n)) - {depot}

    assert len(visited) == len(set(visited)), "Duplicate customer visit."
    assert set(visited) == expected, "Some customers were not visited."

    for route in routes:
        assert route["load"] <= route["capacity"]
        assert route["route"][0] == customers[depot]["id"]
        assert route["route"][-1] == customers[depot]["id"]

    result_data = {
        "dataset": DATA_FILE.name,
        "vehicles_available": vehicle_count,
        "vehicles_used": used_vehicles,
        "total_distance": total_distance,
        "total_delivered_demand": total_load,
        "vehicle_fixed_cost": VEHICLE_FIXED_COST,
        "solver_objective": solution.ObjectiveValue(),
        "solve_time_seconds": round(solve_time, 3),
        "routes": routes,
    }

    # Persist results so the dashboard or QAOA experiment
    # can use them later.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(result_data, indent=2),
        encoding="utf-8",
    )

    print("\n========== OR-TOOLS VRPTW RESULTS ==========")

    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", " -> ".join(map(str, route["route"])))
        print(f"Distance: {route['distance']}")
        print(f"Load: {route['load']} / {route['capacity']}")

        for stop in route["schedule"]:
            print(
                f"  Customer {stop['customer_id']}: "
                f"service starts at {stop['service_start']} "
                f"(window {stop['ready_time']}-"
                f"{stop['due_time']})"
            )

    print("\nVehicles used:", used_vehicles)
    print("Total distance:", total_distance)
    print("Total demand delivered:", total_load)
    print("Solver objective:", solution.ObjectiveValue())
    print("Solve time (seconds):", round(solve_time, 3))
    print("Feasibility checks passed.")
    print("Saved results to:", OUTPUT_FILE)

    return result_data


if __name__ == "__main__":
    dataset = load_solomon_data(DATA_FILE)

    print("Loaded dataset:", DATA_FILE.name)
    print("Number of customer records:", len(dataset["customers"]))
    print("Vehicles available:", dataset["vehicle_count"])
    print("Vehicle capacity:", dataset["vehicle_capacities"][0])

    solve_vrptw(dataset)
