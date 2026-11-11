import itertools
import math

import numpy as np
from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)
from scipy.optimize import differential_evolution


CUSTOMERS = [1, 2, 3, 4]
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
STOCK = 50
REWARD = 100
PENALTY = 100
PAIRWISE_WEIGHT = 0.5
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32]
QUBIT_COUNT = len(CUSTOMERS) + len(SLACK_WEIGHTS)
STATE_COUNT = 2 ** QUBIT_COUNT
SAMPLE_COUNT = 10_000
RANDOM_SEED = 42


def round_trip_distance(customer):
    return 2 * math.dist(COORDINATES[0], COORDINATES[customer])


def pairwise_saving(first_customer, second_customer):
    independent = (
        math.dist(COORDINATES[0], COORDINATES[first_customer])
        + math.dist(COORDINATES[0], COORDINATES[second_customer])
    )
    direct = math.dist(
        COORDINATES[first_customer],
        COORDINATES[second_customer],
    )
    return independent - direct


INDEPENDENT_COSTS = {
    customer: round_trip_distance(customer)
    for customer in CUSTOMERS
}
PAIRWISE_SAVINGS = {
    (first, second): pairwise_saving(first, second)
    for first, second in itertools.combinations(CUSTOMERS, 2)
}


def bits_from_index(index):
    return tuple(
        int(bit)
        for bit in format(index, f"0{QUBIT_COUNT}b")
    )


def selected_customers(bitstring):
    return [
        customer
        for customer, selected in zip(CUSTOMERS, bitstring)
        if selected
    ]


def slack_value(bitstring):
    return sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bitstring[len(CUSTOMERS):],
        )
    )


def demand_for(bitstring):
    return sum(
        DEMANDS[customer] * selected
        for customer, selected in zip(CUSTOMERS, bitstring)
    )


def is_feasible(bitstring):
    demand = demand_for(bitstring)
    return demand <= STOCK and demand + slack_value(bitstring) == STOCK


def independent_energy(bitstring):
    objective = sum(
        (INDEPENDENT_COSTS[customer] - REWARD) * selected
        for customer, selected in zip(CUSTOMERS, bitstring)
    )
    return objective + PENALTY * (
        demand_for(bitstring) + slack_value(bitstring) - STOCK
    ) ** 2


def pairwise_energy(bitstring):
    selected = selected_customers(bitstring)
    objective = sum(
        INDEPENDENT_COSTS[customer] - REWARD
        for customer in selected
    )
    objective -= PAIRWISE_WEIGHT * sum(
        PAIRWISE_SAVINGS[(first, second)]
        for first, second in itertools.combinations(selected, 2)
    )
    return objective + PENALTY * (
        demand_for(bitstring) + slack_value(bitstring) - STOCK
    ) ** 2


def apply_mixer(state, beta):
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    indices = np.arange(len(state))

    for qubit in range(QUBIT_COUNT):
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


def qaoa_state(parameters, energies):
    gamma, beta = parameters
    state = np.full(
        STATE_COUNT,
        1 / math.sqrt(STATE_COUNT),
        dtype=complex,
    )
    state *= np.exp(-1j * gamma * energies)
    return apply_mixer(state, beta)


def optimize_qaoa(energies):
    return differential_evolution(
        lambda parameters: float(
            np.dot(
                np.abs(qaoa_state(parameters, energies)) ** 2,
                energies,
            )
        ),
        bounds=[(0, 2 * math.pi), (0, 2 * math.pi)],
        seed=RANDOM_SEED,
        maxiter=40,
        popsize=8,
        tol=1e-5,
        polish=True,
    )


def describe(bitstring, energy):
    customers = selected_customers(bitstring)
    return {
        "customers": customers,
        "demand": demand_for(bitstring),
        "slack": slack_value(bitstring),
        "energy": float(energy),
    }


def solve_actual_route(customers):
    nodes = [0] + customers
    distance_matrix = [
        [
            round(
                math.dist(COORDINATES[first], COORDINATES[second])
                * 100
            )
            for second in nodes
        ]
        for first in nodes
    ]
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
        [STOCK],
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
    distance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(nodes[node])
        next_index = solution.Value(routing.NextVar(index))
        distance += distance_matrix[node][manager.IndexToNode(next_index)]
        index = next_index
    route.append(0)
    return route, distance / 100


def classical_best(energy_function):
    feasible = [
        (index, energy_function(bits_from_index(index)))
        for index in range(STATE_COUNT)
        if is_feasible(bits_from_index(index))
    ]
    return min(feasible, key=lambda item: item[1])


def run_qaoa(energy_function, label):
    energies = np.array(
        [
            energy_function(bits_from_index(index))
            for index in range(STATE_COUNT)
        ],
        dtype=float,
    )
    classical_index, classical_energy = classical_best(energy_function)
    optimization = optimize_qaoa(energies)
    state = qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2
    samples = np.random.default_rng(RANDOM_SEED).choice(
        STATE_COUNT,
        size=SAMPLE_COUNT,
        p=probabilities,
    )
    measured_feasible = [
        int(index)
        for index in samples
        if is_feasible(bits_from_index(int(index)))
    ]
    best_measured = min(
        measured_feasible,
        key=lambda index: energies[index],
        default=None,
    )
    feasible_mask = np.array(
        [is_feasible(bits_from_index(index)) for index in range(STATE_COUNT)]
    )
    return {
        "label": label,
        "energies": energies,
        "classical": describe(
            bits_from_index(classical_index),
            classical_energy,
        ),
        "optimization": optimization,
        "feasible_probability": float(probabilities[feasible_mask].sum()),
        "optimal_probability": float(probabilities[classical_index]),
        "best_measured": (
            None
            if best_measured is None
            else describe(
                bits_from_index(best_measured),
                energies[best_measured],
            )
        ),
        "feasible_measurements": len(measured_feasible),
    }


def main():
    independent = run_qaoa(independent_energy, "INDEPENDENT-DISTANCE QUBO")
    pairwise = run_qaoa(pairwise_energy, "PAIRWISE-DISTANCE QUBO")

    print("PAIRWISE ROUTING-AWARE QUBO")
    print("---------------------------")
    print("Customers:", CUSTOMERS)
    print("Demands:", DEMANDS)
    print("Available stock:", STOCK)
    print("Reward:", REWARD)
    print("Pairwise weight:", PAIRWISE_WEIGHT)
    print("Independent round-trip costs:", {
        customer: round(cost, 2)
        for customer, cost in INDEPENDENT_COSTS.items()
    })
    print("Pairwise savings:", {
        pair: round(saving, 2)
        for pair, saving in PAIRWISE_SAVINGS.items()
    })

    print("\nCLASSICAL EXHAUSTIVE QUBO COMPARISON")
    print("-------------------------------------")
    for result in (independent, pairwise):
        print(f"{result['label']}:")
        print("  Selected customers:", result["classical"]["customers"])
        print("  Energy:", round(result["classical"]["energy"], 6))

    print("\nQAOA COMPARISON")
    print("---------------")
    for result in (independent, pairwise):
        optimization = result["optimization"]
        print(f"{result['label']}:")
        print("  Gamma:", round(float(optimization.x[0]), 6))
        print("  Beta:", round(float(optimization.x[1]), 6))
        print("  Expected energy:", round(float(optimization.fun), 6))
        print(
            "  Feasible probability:",
            round(result["feasible_probability"], 6),
        )
        print(
            "  Optimal probability:",
            round(result["optimal_probability"], 6),
        )
        print("  Feasible measurements:", result["feasible_measurements"])
        print("  Best measured:", result["best_measured"])

    print("\nACTUAL OR-TOOLS ROUTE COMPARISON")
    print("--------------------------------")
    for result in (independent, pairwise):
        customers = result["classical"]["customers"]
        route, distance = solve_actual_route(customers)
        print(f"{result['label']} classical selection:")
        print("  Customers:", customers)
        print("  Route:", route)
        print("  Actual route distance:", round(distance, 2))
    for result in (independent, pairwise):
        if result["best_measured"] is None:
            continue
        customers = result["best_measured"]["customers"]
        route, distance = solve_actual_route(customers)
        print(f"{result['label']} QAOA selection:")
        print("  Customers:", customers)
        print("  Route:", route)
        print("  Actual route distance:", round(distance, 2))

    print("\nPAIRWISE QUBO VALIDATION")
    print("------------------------")
    print(
        "Classical pairwise solution feasible:",
        is_feasible(
            tuple(
                int(customer in pairwise["classical"]["customers"])
                for customer in CUSTOMERS
            )
            + (0,) * len(SLACK_WEIGHTS),
        ),
    )
    print(
        "QAOA pairwise feasible solution found:",
        pairwise["best_measured"] is not None,
    )


if __name__ == "__main__":
    main()
