import itertools
import math

import numpy as np
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
STOCK = 50
REWARD = 100
PENALTY = 100
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32]
QUBIT_COUNT = len(CUSTOMER_IDS) + len(SLACK_WEIGHTS)
STATE_COUNT = 2 ** QUBIT_COUNT
SAMPLE_COUNT = 10_000
RANDOM_SEED = 42


def round_trip_cost(customer_id):
    return 2 * math.dist(COORDINATES[0], COORDINATES[customer_id])


DELIVERY_COSTS = {
    customer_id: round_trip_cost(customer_id)
    for customer_id in CUSTOMER_IDS
}


def bits_from_index(index):
    return tuple(
        int(bit)
        for bit in format(index, f"0{QUBIT_COUNT}b")
    )


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
    objective = sum(
        (DELIVERY_COSTS[customer_id] - REWARD) * decision
        for customer_id, decision in zip(CUSTOMER_IDS, selected)
    )
    return objective + PENALTY * (demand + slack - STOCK) ** 2


def is_feasible(bitstring):
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


def describe_solution(bitstring):
    selected = bitstring[:len(CUSTOMER_IDS)]
    selected_customers = [
        customer_id
        for customer_id, decision in zip(CUSTOMER_IDS, selected)
        if decision
    ]
    demand = sum(DEMANDS[customer_id] for customer_id in selected_customers)
    cost = sum(DELIVERY_COSTS[customer_id] for customer_id in selected_customers)
    slack = sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bitstring[len(CUSTOMER_IDS):],
        )
    )
    return selected_customers, demand, cost, slack, qubo_energy(bitstring)


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
    result = differential_evolution(
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
    return result


def main():
    energies = np.array(
        [qubo_energy(bits_from_index(index)) for index in range(STATE_COUNT)],
        dtype=float,
    )
    feasible_indices = [
        index
        for index in range(STATE_COUNT)
        if is_feasible(bits_from_index(index))
    ]
    classical_index = min(
        feasible_indices,
        key=lambda index: energies[index],
    )
    classical_bits = bits_from_index(classical_index)
    optimization = optimize_qaoa(energies)
    state = qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2

    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(
        STATE_COUNT,
        size=SAMPLE_COUNT,
        p=probabilities,
    )
    counts = np.bincount(samples, minlength=STATE_COUNT)
    feasible_sample_indices = [
        index
        for index in np.argsort(-counts)
        if is_feasible(bits_from_index(int(index)))
    ]
    best_measured_index = min(
        feasible_sample_indices,
        key=lambda index: energies[index],
        default=None,
    )
    feasible_probability = float(
        probabilities[[is_feasible(bits_from_index(index))
                       for index in range(STATE_COUNT)]].sum()
    )
    optimal_probability = float(probabilities[classical_index])

    print("\nROUTING-AWARE INVENTORY QUBO")
    print("----------------------------")
    print("Customers:", CUSTOMER_IDS)
    print("Demands:", DEMANDS)
    print("Round-trip delivery costs:", {
        customer_id: round(cost, 2)
        for customer_id, cost in DELIVERY_COSTS.items()
    })
    print("Available stock:", STOCK)
    print("Reward per fulfilled customer:", REWARD)
    print("Qubits:", QUBIT_COUNT)
    print("Quantum states:", STATE_COUNT)

    print("\nCLASSICAL REFERENCE")
    print("-------------------")
    print("Selected customers:", describe_solution(classical_bits)[0])
    print("Details:", describe_solution(classical_bits)[1:])
    print("Classical energy:", round(float(energies[classical_index]), 6))

    print("\nQAOA RESULTS")
    print("------------")
    print("Gamma:", round(float(optimization.x[0]), 6))
    print("Beta:", round(float(optimization.x[1]), 6))
    print("Expected energy:", round(float(optimization.fun), 6))
    print("Feasible probability:", round(feasible_probability, 6))
    print("Optimal probability:", round(optimal_probability, 6))
    print("Feasible measurements:", sum(
        is_feasible(bits_from_index(int(index))) for index in samples
    ))

    if best_measured_index is None:
        print("Best feasible measured solution: None")
    else:
        best_bits = bits_from_index(best_measured_index)
        print(
            "Best feasible measured solution:",
            describe_solution(best_bits),
        )

    print("\nQUBO VALIDATION")
    print("----------------")
    print("Classical solution feasible:", is_feasible(classical_bits))
    print(
        "Measured feasible solution found:",
        best_measured_index is not None,
    )
    if best_measured_index is not None:
        print(
            "Measured solution within stock:",
            describe_solution(
                bits_from_index(best_measured_index)
            )[1] <= STOCK,
        )


if __name__ == "__main__":
    main()
