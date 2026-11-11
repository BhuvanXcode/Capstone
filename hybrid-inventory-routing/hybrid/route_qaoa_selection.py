import math

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)


WAREHOUSE = 0
SELECTED_CUSTOMERS = [1, 2, 3]
STOCK = 50
VEHICLE_CAPACITY = 50
DEMANDS = {
    1: 10,
    2: 30,
    3: 10,
    4: 10,
}
COORDINATES = {
    0: (0, 0),
    1: (10, 10),
    2: (20, 10),
    3: (10, 20),
    4: (20, 20),
}

NODES = [WAREHOUSE] + SELECTED_CUSTOMERS
DISTANCE_MATRIX = [
    [
        round(
            math.dist(COORDINATES[from_node], COORDINATES[to_node])
            * 100
        )
        for to_node in NODES
    ]
    for from_node in NODES
]


def calculate_independent_distance():
    return sum(
        2 * math.dist(COORDINATES[WAREHOUSE], COORDINATES[customer])
        for customer in SELECTED_CUSTOMERS
    )


def solve_route():
    manager = pywrapcp.RoutingIndexManager(
        len(NODES),
        1,
        NODES.index(WAREHOUSE),
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return DISTANCE_MATRIX[
            manager.IndexToNode(from_index)
        ][manager.IndexToNode(to_index)]

    distance_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else DEMANDS[NODES[node]]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [VEHICLE_CAPACITY],
        True,
        "Capacity",
    )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools did not find a feasible route.")

    index = routing.Start(0)
    route = []
    route_distance = 0
    route_load = 0

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(NODES[node])
        next_index = solution.Value(routing.NextVar(index))
        next_node = manager.IndexToNode(next_index)
        route_distance += DISTANCE_MATRIX[node][next_node]
        if node != 0:
            route_load += DEMANDS[NODES[node]]
        index = next_index

    route.append(WAREHOUSE)
    return route, route_load, route_distance / 100


def main():
    total_demand = sum(
        DEMANDS[customer] for customer in SELECTED_CUSTOMERS
    )
    independent_distance = calculate_independent_distance()
    route, route_load, combined_distance = solve_route()

    print("QAOA-SELECTED CUSTOMER ROUTING")
    print("------------------------------")
    print("Selected customers:", SELECTED_CUSTOMERS)
    print("Total demand:", total_demand)
    print("Available stock:", STOCK)
    print("Vehicle capacity:", VEHICLE_CAPACITY)
    print(
        "Independent round-trip estimate:",
        round(independent_distance, 2),
    )

    print("\nOR-TOOLS COMBINED ROUTE")
    print("----------------------")
    print("Route:", route)
    print("Route load:", route_load)
    print("Combined route distance:", round(combined_distance, 2))

    print("\nROUTING VALIDATION")
    print("------------------")
    print("Inventory valid:", total_demand <= STOCK)
    print("Capacity valid:", route_load <= VEHICLE_CAPACITY)
    print(
        "All selected customers visited:",
        set(route[1:-1]) == set(SELECTED_CUSTOMERS),
    )
    print(
        "Route starts and ends at warehouse:",
        route[0] == WAREHOUSE and route[-1] == WAREHOUSE,
    )
    print(
        "Distance reduction:",
        round(independent_distance - combined_distance, 2),
    )


if __name__ == "__main__":
    main()
