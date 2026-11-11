import itertools
import math

import numpy as np
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
PAIRWISE_WEIGHTS = [0.5, 1.0]
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
    pair: pairwise_saving(*pair)
    for pair in itertools.combinations(CUSTOMERS, 2)
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


def demand_for(bitstring):
    return sum(
        DEMANDS[customer] * selected
        for customer, selected in zip(CUSTOMERS, bitstring)
    )


def slack_for(bitstring):
    return sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bitstring[len(CUSTOMERS):],
        )
    )


def is_feasible(bitstring):
    demand = demand_for(bitstring)
    return demand <= STOCK and demand + slack_for(bitstring) == STOCK


def qubo_energy(bitstring, pairwise_weight):
    selected = selected_customers(bitstring)
    objective = sum(
        INDEPENDENT_COSTS[customer] - REWARD
        for customer in selected
    )
    objective -= pairwise_weight * sum(
        PAIRWISE_SAVINGS[pair]
        for pair in itertools.combinations(selected, 2)
    )
    imbalance = demand_for(bitstring) + slack_for(bitstring) - STOCK
    return objective + PENALTY * imbalance**2


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
    return {
        "customers": selected_customers(bitstring),
        "demand": demand_for(bitstring),
        "slack": slack_for(bitstring),
        "energy": float(energy),
    }


def run_comparison(pairwise_weight):
    energies = np.array(
        [
            qubo_energy(bits_from_index(index), pairwise_weight)
            for index in range(STATE_COUNT)
        ],
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
    optimization = optimize_qaoa(energies)
    state = qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2
    samples = np.random.default_rng(RANDOM_SEED).choice(
        STATE_COUNT,
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
    feasible_mask = np.array(
        [is_feasible(bits_from_index(index)) for index in range(STATE_COUNT)]
    )

    return {
        "weight": pairwise_weight,
        "energies": energies,
        "classical": describe(
            bits_from_index(classical_index),
            energies[classical_index],
        ),
        "optimization": optimization,
        "feasible_probability": float(probabilities[feasible_mask].sum()),
        "optimal_probability": float(probabilities[classical_index]),
        "feasible_measurements": len(feasible_samples),
        "best_measured": (
            None
            if best_measured_index is None
            else describe(
                bits_from_index(best_measured_index),
                energies[best_measured_index],
            )
        ),
    }


def main():
    results = [
        run_comparison(weight)
        for weight in PAIRWISE_WEIGHTS
    ]

    print("QAOA PAIRWISE-WEIGHT COMPARISON")
    print("-------------------------------")
    print("Customers:", CUSTOMERS)
    print("Demands:", DEMANDS)
    print("Available stock:", STOCK)
    print("Pairwise weights:", PAIRWISE_WEIGHTS)
    print("Qubits:", QUBIT_COUNT)
    print("Quantum states:", STATE_COUNT)
    print("Samples per experiment:", SAMPLE_COUNT)

    print("\nCLASSICAL EXHAUSTIVE QUBO REFERENCES")
    print("-------------------------------------")
    for result in results:
        print(f"Pairwise weight {result['weight']}:")
        print("  Selected customers:", result["classical"]["customers"])
        print("  Demand:", result["classical"]["demand"])
        print("  Energy:", round(result["classical"]["energy"], 6))

    print("\nQAOA RESULTS")
    print("------------")
    for result in results:
        optimization = result["optimization"]
        print(f"Pairwise weight {result['weight']}:")
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
        print("  Best feasible measured solution:", result["best_measured"])

    print("\nVALIDATION")
    print("----------")
    for result in results:
        best = result["best_measured"]
        print(
            f"Weight {result['weight']} classical solution feasible:",
            result["classical"]["demand"] <= STOCK,
        )
        print(
            f"Weight {result['weight']} QAOA feasible solution found:",
            best is not None,
        )
        print(
            f"Weight {result['weight']} QAOA found classical optimum:",
            best is not None
            and best["customers"] == result["classical"]["customers"],
        )


if __name__ == "__main__":
    main()
