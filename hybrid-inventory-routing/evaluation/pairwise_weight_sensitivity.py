import itertools
import math

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)


CUSTOMERS = [1, 2, 3, 4]
STOCK = 50
VEHICLE_CAPACITY = 50
REWARD = 100
WEIGHTS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2]
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


def independent_distance(customers):
    return sum(
        2 * math.dist(COORDINATES[0], COORDINATES[customer])
        for customer in customers
    )


def pairwise_saving(first_customer, second_customer):
    return (
        math.dist(COORDINATES[0], COORDINATES[first_customer])
        + math.dist(COORDINATES[0], COORDINATES[second_customer])
        - math.dist(
            COORDINATES[first_customer],
            COORDINATES[second_customer],
        )
    )


def pairwise_savings(customers):
    return sum(
        pairwise_saving(first, second)
        for first, second in itertools.combinations(customers, 2)
    )


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


def solve_route(customers):
    nodes = [0] + list(customers)
    distance_matrix = build_distance_matrix(nodes)
    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[
            manager.IndexToNode(from_index)
        ][manager.IndexToNode(to_index)]

    distance_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_index)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else DEMANDS[nodes[node]]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
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
    distance = 0
    load = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(nodes[node])
        next_index = solution.Value(routing.NextVar(index))
        next_node = manager.IndexToNode(next_index)
        distance += distance_matrix[node][next_node]
        if node != 0:
            load += DEMANDS[nodes[node]]
        index = next_index
    route.append(0)

    return {
        "route": route,
        "distance": distance / 100,
        "load": load,
    }


def enumerate_results():
    results = []
    for size in range(len(CUSTOMERS) + 1):
        for combination in itertools.combinations(CUSTOMERS, size):
            demand = sum(DEMANDS[customer] for customer in combination)
            if demand > STOCK or demand > VEHICLE_CAPACITY:
                continue
            route = solve_route(combination)
            if route is None:
                continue
            results.append(
                {
                    "customers": combination,
                    "demand": demand,
                    "route": route["route"],
                    "route_distance": route["distance"],
                    "independent_distance": independent_distance(
                        combination
                    ),
                    "pairwise_savings": pairwise_savings(combination),
                }
            )
    return results


def qubo_objective(result, weight):
    return (
        REWARD * len(result["customers"])
        - result["independent_distance"]
        + weight * result["pairwise_savings"]
    )


def main():
    results = enumerate_results()
    actual_optimum = max(
        results,
        key=lambda result: (
            REWARD * len(result["customers"])
            - result["route_distance"]
        ),
    )

    print("PAIRWISE-WEIGHT SENSITIVITY ANALYSIS")
    print("-------------------------------------")
    print("Total customer combinations:", 2 ** len(CUSTOMERS))
    print("Feasible combinations:", len(results))
    print("Weights:", WEIGHTS)
    print("\nACTUAL CLASSICAL ROUTING OPTIMUM")
    print("---------------------------------")
    print("Customers:", list(actual_optimum["customers"]))
    print("Demand:", actual_optimum["demand"])
    print("Route:", actual_optimum["route"])
    print("Route distance:", round(actual_optimum["route_distance"], 2))
    print(
        "Integrated objective:",
        round(
            REWARD * len(actual_optimum["customers"])
            - actual_optimum["route_distance"],
            2,
        ),
    )

    print("\nFEASIBLE SUBSET ROUTING DATA")
    print("----------------------------")
    for result in results:
        print(
            f"Customers {list(result['customers'])}: "
            f"demand={result['demand']}, "
            f"route_distance={result['route_distance']:.2f}, "
            f"pairwise_savings={result['pairwise_savings']:.2f}"
        )

    print("\nWEIGHT SENSITIVITY RESULTS")
    print("--------------------------")
    for weight in WEIGHTS:
        selected = max(
            results,
            key=lambda result: qubo_objective(result, weight),
        )
        integrated_objective = (
            REWARD * len(selected["customers"])
            - selected["route_distance"]
        )
        matches = (
            selected["customers"] == actual_optimum["customers"]
        )
        print(
            f"Weight {weight:>4}: "
            f"customers={list(selected['customers'])}, "
            f"QUBO objective={qubo_objective(selected, weight):.2f}, "
            f"actual distance={selected['route_distance']:.2f}, "
            f"integrated objective={integrated_objective:.2f}, "
            f"matches optimum={matches}"
        )

    print("\nSENSITIVITY VALIDATION")
    print("----------------------")
    print("All subsets respect stock:", all(
        result["demand"] <= STOCK for result in results
    ))
    print("All routes respect capacity:", all(
        result["demand"] <= VEHICLE_CAPACITY for result in results
    ))
    print(
        "Actual optimum route starts and ends at warehouse:",
        actual_optimum["route"][0] == 0
        and actual_optimum["route"][-1] == 0,
    )


if __name__ == "__main__":
    main()
