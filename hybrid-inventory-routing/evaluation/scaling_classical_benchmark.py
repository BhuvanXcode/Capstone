import itertools
import math

import numpy as np


CUSTOMERS = list(range(1, 9))
DEMANDS = {
    1: 10,
    2: 30,
    3: 10,
    4: 10,
    5: 20,
    6: 15,
    7: 25,
    8: 10,
}
COORDINATES = {
    0: (0, 0),
    1: (10, 10),
    2: (20, 10),
    3: (10, 20),
    4: (20, 20),
    5: (30, 10),
    6: (30, 20),
    7: (10, 30),
    8: (20, 30),
}
STOCK = 100
VEHICLE_CAPACITY = 100
REWARD = 100
PENALTY = 100
PAIRWISE_WEIGHT = 0.5
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32, 64]
CUSTOMER_COUNT = len(CUSTOMERS)
QUBIT_COUNT = CUSTOMER_COUNT + len(SLACK_WEIGHTS)
STATE_COUNT = 2 ** QUBIT_COUNT


def bits_from_index(index):
    return tuple(
        int(bit)
        for bit in format(index, f"0{QUBIT_COUNT}b")
    )


def selected_customers(bits):
    return [
        customer
        for customer, selected in zip(CUSTOMERS, bits)
        if selected
    ]


def demand_for(bits):
    return sum(
        DEMANDS[customer] * selected
        for customer, selected in zip(CUSTOMERS, bits)
    )


def slack_for(bits):
    return sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bits[CUSTOMER_COUNT:],
        )
    )


def round_trip_cost(customer):
    return 2 * math.dist(
        COORDINATES[0],
        COORDINATES[customer],
    )


ROUND_TRIP_COSTS = {
    customer: round_trip_cost(customer)
    for customer in CUSTOMERS
}
PAIRWISE_SAVINGS = {
    pair: (
        math.dist(COORDINATES[0], COORDINATES[pair[0]])
        + math.dist(COORDINATES[0], COORDINATES[pair[1]])
        - math.dist(
            COORDINATES[pair[0]],
            COORDINATES[pair[1]],
        )
    )
    for pair in itertools.combinations(CUSTOMERS, 2)
}


def is_feasible(bits):
    demand = demand_for(bits)
    return (
        demand <= STOCK
        and demand <= VEHICLE_CAPACITY
        and demand + slack_for(bits) == STOCK
    )


def qubo_energy(bits):
    selected_customers_list = selected_customers(bits)
    objective = sum(
        ROUND_TRIP_COSTS[customer] - REWARD
        for customer in selected_customers_list
    )
    objective -= PAIRWISE_WEIGHT * sum(
        PAIRWISE_SAVINGS[pair]
        for pair in itertools.combinations(
            selected_customers_list,
            2,
        )
    )
    imbalance = demand_for(bits) + slack_for(bits) - STOCK
    return objective + PENALTY * imbalance**2


def describe(index, energy):
    bits = bits_from_index(index)
    customers = selected_customers(bits)
    return {
        "index": index,
        "customers": customers,
        "demand": demand_for(bits),
        "slack": slack_for(bits),
        "energy": float(energy),
    }


def main():
    energies = np.empty(STATE_COUNT, dtype=float)
    feasible_indices = []

    for index in range(STATE_COUNT):
        bits = bits_from_index(index)
        energies[index] = qubo_energy(bits)
        if is_feasible(bits):
            feasible_indices.append(index)

    classical_index = min(
        feasible_indices,
        key=lambda index: energies[index],
    )
    solution = describe(
        classical_index,
        energies[classical_index],
    )

    print("8-CUSTOMER CLASSICAL QUBO BENCHMARK")
    print("------------------------------------")
    print("Customers:", CUSTOMERS)
    print("Demands:", DEMANDS)
    print("Available stock:", STOCK)
    print("Vehicle capacity:", VEHICLE_CAPACITY)
    print("Reward per fulfilled customer:", REWARD)
    print("Pairwise weight:", PAIRWISE_WEIGHT)
    print("Slack weights:", SLACK_WEIGHTS)
    print("Customer variables:", CUSTOMER_COUNT)
    print("Slack variables:", len(SLACK_WEIGHTS))
    print("Total qubits:", QUBIT_COUNT)
    print("Quantum states evaluated:", STATE_COUNT)
    print("Feasible states:", len(feasible_indices))

    print("\nROUND-TRIP COSTS")
    print("----------------")
    print({
        customer: round(cost, 2)
        for customer, cost in ROUND_TRIP_COSTS.items()
    })
    print("Pairwise savings:", {
        pair: round(saving, 2)
        for pair, saving in PAIRWISE_SAVINGS.items()
    })

    print("\nCLASSICAL QUBO SOLUTION")
    print("-----------------------")
    print("State index:", solution["index"])
    print("Selected customers:", solution["customers"])
    print("Fulfilled demand:", solution["demand"])
    print("Slack:", solution["slack"])
    print("QUBO energy:", round(solution["energy"], 6))

    print("\nVALIDATION")
    print("----------")
    print("All states evaluated:", len(energies) == STATE_COUNT)
    print("Feasible states identified:", len(feasible_indices) > 0)
    print("Inventory valid:", solution["demand"] <= STOCK)
    print("Vehicle capacity valid:", solution["demand"] <= VEHICLE_CAPACITY)
    print(
        "Inventory equality valid:",
        solution["demand"] + solution["slack"] == STOCK,
    )
    print(
        "Selected customer count:",
        len(solution["customers"]),
    )


if __name__ == "__main__":
    main()
