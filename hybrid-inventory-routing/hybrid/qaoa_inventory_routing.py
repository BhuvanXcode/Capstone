import math

import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from scipy.optimize import differential_evolution


CUSTOMER_IDS = [1, 2, 3, 4]
DEMANDS = {1: 10, 2: 30, 3: 10, 4: 10}
COORDINATES = {
    0: (0, 0),
    1: (10, 10),
    2: (20, 10),
    3: (10, 20),
    4: (20, 20),
}
STOCK = 60
VEHICLE_CAPACITY = 50
VEHICLE_COUNT = 3
PENALTY = 100
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32]
SAMPLE_COUNT = 10_000
RANDOM_SEED = 42


def qubo_energy(bitstring):
    selected = bitstring[:len(CUSTOMER_IDS)]
    slack_bits = bitstring[len(CUSTOMER_IDS):]
    demand = sum(
        DEMANDS[customer_id] * decision
        for customer_id, decision in zip(CUSTOMER_IDS, selected)
    )
    slack = sum(
        weight * bit
        for weight, bit in zip(SLACK_WEIGHTS, slack_bits)
    )
    return -sum(selected) + PENALTY * (demand + slack - STOCK) ** 2


def bits_from_index(index, qubit_count):
    return tuple(
        int(bit)
        for bit in format(index, f"0{qubit_count}b")
    )


def apply_mixer(state, beta, qubit_count):
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    indices = np.arange(len(state))

    for qubit in range(qubit_count):
        zero_indices = indices[
            (indices & (1 << qubit)) == 0
        ]
        one_indices = zero_indices | (1 << qubit)
        zero_values = mixed_state[zero_indices].copy()
        one_values = mixed_state[one_indices].copy()
        mixed_state[zero_indices] = (
            cosine * zero_values - 1j * sine * one_values
        )
        mixed_state[one_indices] = (
            -1j * sine * zero_values + cosine * one_values
        )

    return mixed_state


def qaoa_state(parameters, energies, qubit_count):
    qubit_count = len(CUSTOMER_IDS) + len(SLACK_WEIGHTS)
    state_count = 2 ** qubit_count
    state = np.full(
        state_count,
        1 / math.sqrt(state_count),
        dtype=complex,
    )
    gamma, beta = parameters
    state *= np.exp(-1j * gamma * energies)
    state = apply_mixer(state, beta, qubit_count)
    return state


def expected_energy(parameters, energies, qubit_count):
    state = qaoa_state(parameters, energies, qubit_count)
    probabilities = np.abs(state) ** 2
    return float(np.dot(probabilities, energies))


def optimize_qaoa(energies, qubit_count):
    result = differential_evolution(
        expected_energy,
        bounds=[(0, 2 * math.pi), (0, 2 * math.pi)],
        args=(energies, qubit_count),
        seed=RANDOM_SEED,
        maxiter=40,
        popsize=8,
        tol=1e-5,
        polish=True,
    )
    return result


def qaoa_probabilities():
    qubit_count = len(CUSTOMER_IDS) + len(SLACK_WEIGHTS)
    state_count = 2 ** qubit_count
    energies = np.array(
        [
            qubo_energy(bits_from_index(index, qubit_count))
            for index in range(state_count)
        ],
        dtype=float,
    )
    optimization = optimize_qaoa(energies, qubit_count)
    state = qaoa_state(optimization.x, energies, qubit_count)
    probabilities = np.abs(state) ** 2
    return probabilities, energies, qubit_count, optimization


def is_feasible_inventory(bitstring):
    selected = bitstring[:len(CUSTOMER_IDS)]
    demand = sum(
        DEMANDS[customer_id] * decision
        for customer_id, decision in zip(CUSTOMER_IDS, selected)
    )
    slack = sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bitstring[len(CUSTOMER_IDS):],
        )
    )
    return demand <= STOCK and demand + slack == STOCK


def select_inventory(probabilities):
    qubit_count = len(CUSTOMER_IDS) + len(SLACK_WEIGHTS)
    candidates = []
    for index in range(2 ** qubit_count):
        bitstring = bits_from_index(index, qubit_count)
        if is_feasible_inventory(bitstring):
            candidates.append(
                (
                    qubo_energy(bitstring),
                    -probabilities[index],
                    bitstring,
                )
            )

    if not candidates:
        raise RuntimeError("QAOA did not produce a feasible inventory allocation.")

    _, _, bitstring = min(candidates)
    selected = [
        customer_id
        for customer_id, decision in zip(CUSTOMER_IDS, bitstring)
        if decision
    ]
    return bitstring, selected


def route_selected_customers(selected_customers):
    locations = [0] + selected_customers
    manager = pywrapcp.RoutingIndexManager(
        len(locations),
        VEHICLE_COUNT,
        0,
    )
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = [
        [
            int(round(
                math.dist(
                    COORDINATES[locations[first]],
                    COORDINATES[locations[second]],
                ) * 100
            ))
            for second in range(len(locations))
        ]
        for first in range(len(locations))
    ]

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
        return 0 if node == 0 else DEMANDS[locations[node]]

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

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = 5
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools found no feasible routing solution.")

    routes = []
    total_distance = 0
    for vehicle in range(VEHICLE_COUNT):
        index = routing.Start(vehicle)
        route = [0]
        load = 0
        distance_value = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            if node != 0:
                load += DEMANDS[locations[node]]
            distance_value += distance_matrix[node][next_node]
            index = next_index
            if not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
        route.append(0)
        if len(route) > 2:
            distance_value /= 100
            routes.append(
                {
                    "vehicle": vehicle + 1,
                    "route": route,
                    "load": load,
                    "distance": distance_value,
                }
            )
            total_distance += distance_value

    return routes, total_distance


def main():
    probabilities, energies, qubit_count, optimization = (
        qaoa_probabilities()
    )
    bitstring, selected_customers = select_inventory(probabilities)
    routes, total_distance = route_selected_customers(selected_customers)
    routed_customers = [
        customer_id
        for route in routes
        for customer_id in route["route"]
        if customer_id != 0
    ]
    selected_demand = sum(
        DEMANDS[customer_id] for customer_id in selected_customers
    )
    feasible_mask = np.array(
        [
            is_feasible_inventory(bits_from_index(index, qubit_count))
            for index in range(len(probabilities))
        ]
    )
    feasible_probability = float(probabilities[feasible_mask].sum())
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(
        len(probabilities),
        size=SAMPLE_COUNT,
        p=probabilities,
    )
    feasible_measurements = sum(
        is_feasible_inventory(bits_from_index(int(index), qubit_count))
        for index in samples
    )

    print("\nHYBRID QAOA + CLASSICAL ROUTING")
    print("-------------------------------")
    print("QAOA qubits:", qubit_count)
    print("QAOA gamma:", round(float(optimization.x[0]), 6))
    print("QAOA beta:", round(float(optimization.x[1]), 6))

    print("\nQAOA INVENTORY OPTIMIZATION")
    print("---------------------------")
    print("Expected energy:", round(float(optimization.fun), 6))
    print("Feasible probability:", round(feasible_probability, 6))
    print("Feasible measurements:", feasible_measurements)
    print("Selected inventory bitstring:", list(bitstring))

    print("\nQAOA SELECTED CUSTOMERS")
    print("-----------------------")
    print("Selected customers:", selected_customers)
    print("Selected demand:", selected_demand)
    print("Inventory:", STOCK)

    print("\nCLASSICAL OR-TOOLS ROUTING")
    print("--------------------------")
    for route in routes:
        print(f"Vehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))
    print("Vehicles used:", len(routes))
    print("Total distance:", round(total_distance, 2))

    print("\nINDEPENDENT VALIDATION")
    print("----------------------")
    print("Inventory valid:", selected_demand <= STOCK)
    print(
        "All selected customers routed:",
        set(routed_customers) == set(selected_customers),
    )
    print(
        "No duplicate deliveries:",
        len(routed_customers) == len(set(routed_customers)),
    )
    print(
        "Capacity valid:",
        all(route["load"] <= VEHICLE_CAPACITY for route in routes),
    )
    print(
        "Routes start and end at warehouse:",
        all(
            route["route"][0] == 0
            and route["route"][-1] == 0
            for route in routes
        ),
    )
    print("QAOA inventory feasible:", is_feasible_inventory(bitstring))
    print("\nHYBRID OPTIMIZATION COMPLETED")


if __name__ == "__main__":
    main()
