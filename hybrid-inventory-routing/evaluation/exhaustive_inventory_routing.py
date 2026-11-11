import itertools
import math

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)


CUSTOMERS = [1, 2, 3, 4]
WAREHOUSE = 0
STOCK = 50
VEHICLE_CAPACITY = 50
REWARD = 100
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


def build_distance_matrix(nodes):
    return [
        [
            round(
                math.dist(COORDINATES[first], COORDINATES[second])
                * 100
            )
            for second in nodes
        ]
        for first in nodes
    ]


def solve_route(selected_customers):
    nodes = [WAREHOUSE] + list(selected_customers)
    distance_matrix = build_distance_matrix(nodes)
    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[
            manager.IndexToNode(from_index)
        ][manager.IndexToNode(to_index)]

    distance_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else DEMANDS[nodes[node]]

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
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = 2
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return None

    index = routing.Start(0)
    route = []
    route_distance = 0
    route_load = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(nodes[node])
        next_index = solution.Value(routing.NextVar(index))
        next_node = manager.IndexToNode(next_index)
        route_distance += distance_matrix[node][next_node]
        if node != 0:
            route_load += DEMANDS[nodes[node]]
        index = next_index
    route.append(WAREHOUSE)

    return {
        "route": route,
        "load": route_load,
        "distance": route_distance / 100,
    }


def main():
    results = []
    rejected = []

    for size in range(len(CUSTOMERS) + 1):
        for combination in itertools.combinations(CUSTOMERS, size):
            demand = sum(DEMANDS[customer] for customer in combination)
            if demand > STOCK or demand > VEHICLE_CAPACITY:
                rejected.append((combination, "inventory/capacity"))
                continue

            route_result = solve_route(combination)
            if route_result is None:
                rejected.append((combination, "no feasible route"))
                continue

            objective = (
                REWARD * len(combination)
                - route_result["distance"]
            )
            results.append(
                {
                    "customers": combination,
                    "demand": demand,
                    "objective": objective,
                    **route_result,
                }
            )

    best = max(results, key=lambda result: result["objective"])
    best_distance = min(results, key=lambda result: result["distance"])

    print("EXHAUSTIVE INVENTORY-ROUTING BENCHMARK")
    print("---------------------------------------")
    print("Total combinations:", 2 ** len(CUSTOMERS))
    print("Feasible combinations:", len(results))
    print("Rejected combinations:", len(rejected))

    print("\nFEASIBLE COMBINATION RESULTS")
    print("----------------------------")
    for result in results:
        print(
            f"Customers {list(result['customers'])}: "
            f"demand={result['demand']}, "
            f"distance={result['distance']:.2f}, "
            f"objective={result['objective']:.2f}, "
            f"route={result['route']}"
        )

    print("\nBEST INTEGRATED SOLUTION")
    print("------------------------")
    print("Selected customers:", list(best["customers"]))
    print("Demand:", best["demand"])
    print("Route:", best["route"])
    print("Distance:", round(best["distance"], 2))
    print("Objective:", round(best["objective"], 2))

    print("\nSHORTEST ROUTE SOLUTION")
    print("-----------------------")
    print("Selected customers:", list(best_distance["customers"]))
    print("Distance:", round(best_distance["distance"], 2))

    print("\nBENCHMARK VALIDATION")
    print("--------------------")
    print("Inventory valid:", best["demand"] <= STOCK)
    print("Capacity valid:", best["load"] <= VEHICLE_CAPACITY)
    print(
        "Route starts and ends at warehouse:",
        best["route"][0] == WAREHOUSE
        and best["route"][-1] == WAREHOUSE,
    )
    print(
        "All selected customers visited:",
        set(best["route"][1:-1]) == set(best["customers"]),
    )


if __name__ == "__main__":
    main()
