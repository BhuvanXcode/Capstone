import itertools
import math

import numpy as np
from scipy.optimize import differential_evolution


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
REWARD = 100
PENALTY = 100
PAIRWISE_WEIGHT = 0.5
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32, 64]
N_CUSTOMERS = len(CUSTOMERS)
N_QUBITS = N_CUSTOMERS + len(SLACK_WEIGHTS)
N_STATES = 2 ** N_QUBITS
SAMPLE_COUNT = 10_000
RANDOM_SEED = 42


def bits_from_index(index):
    return tuple(
        int(bit)
        for bit in format(index, f"0{N_QUBITS}b")
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
            bits[N_CUSTOMERS:],
        )
    )


def is_feasible(bits):
    demand = demand_for(bits)
    return (
        demand <= STOCK
        and demand + slack_for(bits) == STOCK
    )


def round_trip_distance(customer):
    return 2 * math.dist(
        COORDINATES[0],
        COORDINATES[customer],
    )


ROUND_TRIP_COSTS = {
    customer: round_trip_distance(customer)
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


def qubo_energy(bits):
    selected = selected_customers(bits)
    objective = sum(
        ROUND_TRIP_COSTS[customer] - REWARD
        for customer in selected
    )
    objective -= PAIRWISE_WEIGHT * sum(
        PAIRWISE_SAVINGS[pair]
        for pair in itertools.combinations(selected, 2)
    )
    imbalance = demand_for(bits) + slack_for(bits) - STOCK
    return objective + PENALTY * imbalance**2


def apply_mixer(state, beta):
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    indices = np.arange(N_STATES)

    for qubit in range(N_QUBITS):
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
        N_STATES,
        1 / math.sqrt(N_STATES),
        dtype=complex,
    )
    state *= np.exp(-1j * gamma * energies)
    return apply_mixer(state, beta)


def optimize_parameters(energies):
    return differential_evolution(
        lambda parameters: float(
            np.dot(
                np.abs(qaoa_state(parameters, energies)) ** 2,
                energies,
            )
        ),
        bounds=[(0, 2 * math.pi), (0, 2 * math.pi)],
        seed=RANDOM_SEED,
        maxiter=30,
        popsize=6,
        tol=1e-5,
        polish=True,
    )


def describe(index, energies):
    bits = bits_from_index(index)
    return {
        "state_index": index,
        "customers": selected_customers(bits),
        "demand": demand_for(bits),
        "slack": slack_for(bits),
        "energy": float(energies[index]),
    }


def main():
    energies = np.array(
        [
            qubo_energy(bits_from_index(index))
            for index in range(N_STATES)
        ],
        dtype=float,
    )
    feasible_indices = [
        index
        for index in range(N_STATES)
        if is_feasible(bits_from_index(index))
    ]
    classical_index = min(
        feasible_indices,
        key=lambda index: energies[index],
    )

    print("8-CUSTOMER QAOA SCALING EXPERIMENT")
    print("----------------------------------")
    print("Customers:", CUSTOMERS)
    print("Available stock:", STOCK)
    print("Reward:", REWARD)
    print("Pairwise weight:", PAIRWISE_WEIGHT)
    print("Customer variables:", N_CUSTOMERS)
    print("Slack variables:", len(SLACK_WEIGHTS))
    print("Total qubits:", N_QUBITS)
    print("Quantum states:", N_STATES)
    print("Feasible states:", len(feasible_indices))

    print("\nCLASSICAL REFERENCE")
    print("-------------------")
    print(describe(classical_index, energies))

    optimization = optimize_parameters(energies)
    state = qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2
    feasible_mask = np.array(
        [
            is_feasible(bits_from_index(index))
            for index in range(N_STATES)
        ]
    )
    samples = np.random.default_rng(RANDOM_SEED).choice(
        N_STATES,
        size=SAMPLE_COUNT,
        p=probabilities,
    )
    feasible_samples = [
        int(index)
        for index in samples
        if is_feasible(bits_from_index(int(index)))
    ]
    best_measured_index = min(
        feasible_samples,
        key=lambda index: energies[index],
        default=None,
    )

    print("\nQAOA RESULTS")
    print("------------")
    print("Gamma:", round(float(optimization.x[0]), 6))
    print("Beta:", round(float(optimization.x[1]), 6))
    print("Expected energy:", round(float(optimization.fun), 6))
    print(
        "Feasible probability:",
        round(float(probabilities[feasible_mask].sum()), 6),
    )
    print(
        "Optimal probability:",
        round(float(probabilities[classical_index]), 6),
    )
    print("Feasible measurements:", len(feasible_samples))
    if best_measured_index is None:
        print("Best feasible measured solution: None")
    else:
        print(
            "Best feasible measured solution:",
            describe(best_measured_index, energies),
        )

    print("\nVALIDATION")
    print("----------")
    classical_bits = bits_from_index(classical_index)
    print("Classical solution feasible:", is_feasible(classical_bits))
    print("Classical inventory valid:", demand_for(classical_bits) <= STOCK)
    print(
        "Classical equality valid:",
        demand_for(classical_bits) + slack_for(classical_bits) == STOCK,
    )
    print(
        "QAOA feasible solution found:",
        best_measured_index is not None,
    )
    print(
        "QAOA found classical optimum:",
        best_measured_index == classical_index,
    )


if __name__ == "__main__":
    main()
