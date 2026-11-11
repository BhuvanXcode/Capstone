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
N_QUBITS = len(CUSTOMERS) + len(SLACK_WEIGHTS)
N_STATES = 2 ** N_QUBITS
SAMPLES_PER_RUN = 10_000
RUN_COUNT = 10
OPTIMIZATION_SEED = 42
MEASUREMENT_SEEDS = range(RUN_COUNT)


INDEPENDENT_COSTS = {
    customer: 2 * math.dist(COORDINATES[0], COORDINATES[customer])
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
            bits[len(CUSTOMERS):],
        )
    )


def is_feasible(bits):
    demand = demand_for(bits)
    return demand <= STOCK and demand + slack_for(bits) == STOCK


def qubo_energy(bits, pairwise_weight):
    selected = selected_customers(bits)
    objective = sum(
        INDEPENDENT_COSTS[customer] - REWARD
        for customer in selected
    )
    objective -= pairwise_weight * sum(
        PAIRWISE_SAVINGS[pair]
        for pair in itertools.combinations(selected, 2)
    )
    imbalance = demand_for(bits) + slack_for(bits) - STOCK
    return objective + PENALTY * imbalance**2


def apply_mixer(state, beta):
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    indices = np.arange(len(state))

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
        seed=OPTIMIZATION_SEED,
        maxiter=40,
        popsize=8,
        tol=1e-5,
        polish=True,
    )


def describe(index, energies):
    bits = bits_from_index(index)
    return {
        "customers": selected_customers(bits),
        "demand": demand_for(bits),
        "slack": slack_for(bits),
        "energy": float(energies[index]),
    }


def evaluate_weight(pairwise_weight):
    energies = np.array(
        [
            qubo_energy(bits_from_index(index), pairwise_weight)
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
    optimization = optimize_parameters(energies)
    state = qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2
    feasible_mask = np.array(
        [is_feasible(bits_from_index(index)) for index in range(N_STATES)]
    )

    runs = []
    for seed in MEASUREMENT_SEEDS:
        samples = np.random.default_rng(seed).choice(
            N_STATES,
            size=SAMPLES_PER_RUN,
            p=probabilities,
        )
        counts = np.bincount(samples, minlength=N_STATES)
        feasible_sample_indices = [
            index
            for index in np.flatnonzero(counts)
            if is_feasible(bits_from_index(int(index)))
        ]
        best_index = min(
            feasible_sample_indices,
            key=lambda index: energies[index],
            default=None,
        )
        runs.append(
            {
                "feasible_probability": float(
                    sum(
                        is_feasible(bits_from_index(int(index)))
                        for index in samples
                    )
                    / SAMPLES_PER_RUN
                ),
                "optimal_probability": float(
                    np.count_nonzero(samples == classical_index)
                    / SAMPLES_PER_RUN
                ),
                "feasible_measurements": sum(
                    is_feasible(bits_from_index(int(index)))
                    for index in samples
                ),
                "best_index": best_index,
            }
        )

    best_indices = [
        run["best_index"]
        for run in runs
        if run["best_index"] is not None
    ]
    best_index = min(
        best_indices,
        key=lambda index: energies[index],
        default=None,
    )
    return {
        "weight": pairwise_weight,
        "energies": energies,
        "classical_index": classical_index,
        "classical": describe(classical_index, energies),
        "optimization": optimization,
        "feasible_probability": float(probabilities[feasible_mask].sum()),
        "optimal_probability": float(probabilities[classical_index]),
        "runs": runs,
        "best_index": best_index,
    }


def main():
    results = [
        evaluate_weight(weight)
        for weight in PAIRWISE_WEIGHTS
    ]

    print("REPEATED-RUN QAOA EVALUATION")
    print("----------------------------")
    print("Customers:", CUSTOMERS)
    print("Pairwise weights:", PAIRWISE_WEIGHTS)
    print("Qubits:", N_QUBITS)
    print("Quantum states:", N_STATES)
    print("Measurement runs per weight:", RUN_COUNT)
    print("Samples per run:", SAMPLES_PER_RUN)

    for result in results:
        runs = result["runs"]
        average_feasible_probability = np.mean(
            [run["feasible_probability"] for run in runs]
        )
        average_optimal_probability = np.mean(
            [run["optimal_probability"] for run in runs]
        )
        average_feasible_measurements = np.mean(
            [run["feasible_measurements"] for run in runs]
        )
        optimum_runs = sum(
            run["best_index"] == result["classical_index"]
            for run in runs
        )

        print(f"\nPAIRWISE WEIGHT: {result['weight']}")
        print("-------------------------")
        print("Classical optimum:", result["classical"])
        print(
            "Optimized gamma:",
            round(float(result["optimization"].x[0]), 6),
        )
        print(
            "Optimized beta:",
            round(float(result["optimization"].x[1]), 6),
        )
        print(
            "State-vector feasible probability:",
            round(result["feasible_probability"], 6),
        )
        print(
            "State-vector optimal probability:",
            round(result["optimal_probability"], 6),
        )
        print(
            "Average feasible probability:",
            round(float(average_feasible_probability), 6),
        )
        print(
            "Average optimal probability:",
            round(float(average_optimal_probability), 6),
        )
        print(
            "Runs finding classical optimum:",
            f"{optimum_runs}/{RUN_COUNT}",
        )
        print(
            "Average feasible measurements:",
            round(float(average_feasible_measurements), 2),
        )
        if result["best_index"] is None:
            print("Best feasible customer selection: None")
        else:
            print(
                "Best feasible customer selection:",
                describe(result["best_index"], result["energies"]),
            )

        print("Per-run measurements:")
        for run_number, run in enumerate(runs, start=1):
            best = (
                None
                if run["best_index"] is None
                else describe(run["best_index"], result["energies"])
            )
            print(
                f"  Run {run_number}: "
                f"feasible_probability="
                f"{run['feasible_probability']:.6f}, "
                f"optimal_probability="
                f"{run['optimal_probability']:.6f}, "
                f"feasible_measurements="
                f"{run['feasible_measurements']}, "
                f"best={best}"
            )


if __name__ == "__main__":
    main()
