from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from customers import (
    load_c101,
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
)


def solve_ortools():
    warehouse, customers = load_c101()
    locations = [warehouse] + customers

    num_locations = len(locations)
    depot = 0

    # Create distance and time matrices
    distance_matrix = []
    time_matrix = []

    for i in range(num_locations):
        distance_row = []
        time_row = []

        for j in range(num_locations):
            dx = locations[i]["x"] - locations[j]["x"]
            dy = locations[i]["y"] - locations[j]["y"]

            distance = (dx ** 2 + dy ** 2) ** 0.5

            # Distance is multiplied by 100 to retain precision
            distance_row.append(int(round(distance * 100)))

            # Travel time is rounded to integer time units
            time_row.append(int(round(distance)))

        distance_matrix.append(distance_row)
        time_matrix.append(time_row)

    # Create routing manager and model
    manager = pywrapcp.RoutingIndexManager(
        num_locations,
        VEHICLE_COUNT,
        depot,
    )

    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return distance_matrix[from_node][to_node]

    distance_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    # Demand callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)

        return locations[from_node]["demand"]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [VEHICLE_CAPACITY] * VEHICLE_COUNT,
        True,
        "Capacity",
    )

    # Time callback: travel time + service time
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time = time_matrix[from_node][to_node]
        service_time = locations[from_node]["service_time"]

        return travel_time + service_time

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time dimension
    routing.AddDimension(
        time_callback_index,
        1236,       # Maximum waiting time
        1236,       # Maximum route time
        False,      # Do not force start time to zero
        "Time",
    )

    time_dimension = routing.GetDimensionOrDie("Time")

    # Apply customer time windows
    for node in range(1, num_locations):
        index = manager.NodeToIndex(node)

        time_dimension.CumulVar(index).SetRange(
            locations[node]["ready_time"],
            locations[node]["due_date"],
        )

    # Apply warehouse time window
    for vehicle_id in range(VEHICLE_COUNT):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)

        time_dimension.CumulVar(start_index).SetRange(0, 1236)
        time_dimension.CumulVar(end_index).SetRange(0, 1236)

    # Search strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = 10

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        print("No feasible solution found.")
        return

    # Validate all customer visits
    visited_customers = []
    total_distance = 0
    vehicles_used = 0
    capacity_valid = True
    time_valid = True

    print("\nOR-TOOLS ROUTING VALIDATION")
    print("=" * 50)

    for vehicle_id in range(VEHICLE_COUNT):
        index = routing.Start(vehicle_id)
        route = [depot]
        route_load = 0
        route_distance = 0
        route_has_customers = False

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot:
                visited_customers.append(node)
                route_has_customers = True
                route_load += locations[node]["demand"]
                arrival_time = solution.Min(
                    time_dimension.CumulVar(index)
                )
                if not (
                    locations[node]["ready_time"]
                    <= arrival_time
                    <= locations[node]["due_date"]
                ):
                    time_valid = False

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            route_distance += distance_matrix[node][next_node]
            route.append(next_node)
            index = next_index

        if route_has_customers:
            vehicles_used += 1
            total_distance += route_distance / 100
            if route_load > VEHICLE_CAPACITY:
                capacity_valid = False
            print(f"\nVehicle {vehicle_id + 1}")
            print("Route:", route)
            print("Load:", route_load)
            print("Distance:", round(route_distance / 100, 2))

    expected_customers = set(range(1, num_locations))
    visited_set = set(visited_customers)
    all_customers_visited = visited_set == expected_customers
    duplicate_visits = len(visited_customers) != len(visited_set)

    print("\nValidation summary")
    print("Vehicles used:", vehicles_used)
    print("Customers visited:", len(visited_set))
    print("All customers visited:", all_customers_visited)
    print("Duplicate customer visits:", duplicate_visits)
    print("Capacity valid:", capacity_valid)
    print("Time windows valid:", time_valid)
    print("Total distance:", round(total_distance, 2))

    return {
        "vehicles_used": vehicles_used,
        "customers_visited": len(visited_set),
        "all_customers_visited": all_customers_visited,
        "capacity_valid": capacity_valid,
        "time_valid": time_valid,
        "total_distance": total_distance,
    }


if __name__ == "__main__":
    solve_ortools()
