import itertools
import math

import numpy as np
from scipy.optimize import differential_evolution


CUSTOMER_IDS = [1, 2, 3, 4]
DEMANDS = [10, 30, 10, 10]
STOCK = 60
PENALTY = 100
CUSTOMER_COUNT = len(CUSTOMER_IDS)
SLACK_WEIGHTS = [1, 2, 4, 8, 16, 32]
VARIABLE_COUNT = CUSTOMER_COUNT + len(SLACK_WEIGHTS)
STATE_COUNT = 2 ** VARIABLE_COUNT
SAMPLE_COUNT = 10_000
RANDOM_SEED = 42


def qubo_energy(bitstring):
    selected = bitstring[:CUSTOMER_COUNT]
    slack_bits = bitstring[CUSTOMER_COUNT:]
    fulfilled_demand = sum(
        demand * decision
        for demand, decision in zip(DEMANDS, selected)
    )
    slack = sum(
        weight * bit
        for weight, bit in zip(SLACK_WEIGHTS, slack_bits)
    )
    balance = fulfilled_demand + slack - STOCK
    return -sum(selected) + PENALTY * balance ** 2


def all_energies():
    return np.array(
        [
            qubo_energy(bits)
            for bits in itertools.product((0, 1), repeat=VARIABLE_COUNT)
        ],
        dtype=float,
    )


def apply_mixer(state, beta):
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = np.sin(beta)
    indices = np.arange(len(state))

    for qubit in range(VARIABLE_COUNT):
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


def qaoa_state(parameters, depth, energies):
    state = np.full(
        STATE_COUNT,
        1 / math.sqrt(STATE_COUNT),
        dtype=complex,
    )
    for layer in range(depth):
        gamma = parameters[2 * layer]
        beta = parameters[2 * layer + 1]
        state *= np.exp(-1j * gamma * energies)
        state = apply_mixer(state, beta)
    return state


def expected_energy(parameters, depth, energies):
    state = qaoa_state(parameters, depth, energies)
    probabilities = np.abs(state) ** 2
    return float(np.dot(probabilities, energies))


def bits_from_index(index):
    return tuple(
        int(bit)
        for bit in format(index, f"0{VARIABLE_COUNT}b")
    )


def feasible_solution(bitstring):
    selected = bitstring[:CUSTOMER_COUNT]
    slack_bits = bitstring[CUSTOMER_COUNT:]
    fulfilled_demand = sum(
        demand * decision
        for demand, decision in zip(DEMANDS, selected)
    )
    slack = sum(
        weight * bit
        for weight, bit in zip(SLACK_WEIGHTS, slack_bits)
    )
    return (
        fulfilled_demand <= STOCK
        and fulfilled_demand + slack == STOCK
    )


def describe_solution(bitstring):
    selected = bitstring[:CUSTOMER_COUNT]
    fulfilled_ids = [
        customer_id
        for customer_id, decision in zip(CUSTOMER_IDS, selected)
        if decision
    ]
    fulfilled_demand = sum(
        demand * decision
        for demand, decision in zip(DEMANDS, selected)
    )
    slack = sum(
        weight * bit
        for weight, bit in zip(
            SLACK_WEIGHTS,
            bitstring[CUSTOMER_COUNT:],
        )
    )
    return fulfilled_ids, fulfilled_demand, slack, qubo_energy(bitstring)


def optimize_depth(depth, energies):
    bounds = [(0, 2 * math.pi)] * (2 * depth)

    def objective(parameters):
        return expected_energy(parameters, depth, energies)

    return differential_evolution(
        objective,
        bounds,
        seed=RANDOM_SEED + depth,
        maxiter=40,
        popsize=8,
        tol=1e-5,
        polish=True,
        updating="immediate",
    )


def analyze_depth(depth, energies, classical_energy):
    result = optimize_depth(depth, energies)
    state = qaoa_state(result.x, depth, energies)
    probabilities = np.abs(state) ** 2
    rng = np.random.default_rng(RANDOM_SEED + depth)
    samples = rng.choice(
        STATE_COUNT,
        size=SAMPLE_COUNT,
        p=probabilities,
    )
    counts = np.bincount(samples, minlength=STATE_COUNT)

    feasible_mask = np.array(
        [feasible_solution(bits_from_index(i)) for i in range(STATE_COUNT)]
    )
    feasible_probability = float(probabilities[feasible_mask].sum())
    optimal_probability = float(
        probabilities[np.isclose(energies, classical_energy)].sum()
    )

    feasible_sample_indices = [
        index
        for index in np.argsort(-counts)
        if feasible_solution(bits_from_index(int(index)))
    ]
    best_feasible_index = min(
        feasible_sample_indices,
        key=lambda index: energies[index],
        default=None,
    )
    most_frequent_index = int(np.argmax(counts))

    return {
        "result": result,
        "expected_energy": float(result.fun),
        "feasible_probability": feasible_probability,
        "optimal_probability": optimal_probability,
        "most_frequent_index": most_frequent_index,
        "best_feasible_index": best_feasible_index,
        "counts": counts,
    }


def main():
    energies = all_energies()
    classical_index = int(np.argmin(energies))
    classical_energy = float(energies[classical_index])
    classical_bits = bits_from_index(classical_index)

    print("\nQAOA INVENTORY DEPTH COMPARISON")
    print("------------------------------")
    print("Customers:", CUSTOMER_IDS)
    print("Demands:", DEMANDS)
    print("Inventory:", STOCK)
    print("Qubits:", VARIABLE_COUNT)
    print("Quantum states:", STATE_COUNT)
    print("Classical optimum energy:", classical_energy)
    print("Classical optimum:", describe_solution(classical_bits))

    results = {}
    for depth in (1, 2):
        print(f"\nOptimizing QAOA depth p={depth}...")
        results[depth] = analyze_depth(
            depth,
            energies,
            classical_energy,
        )

    for depth in (1, 2):
        result = results[depth]
        most_frequent = bits_from_index(result["most_frequent_index"])
        best_feasible_index = result["best_feasible_index"]

        print(f"\nQAOA DEPTH p={depth} RESULTS")
        print("--------------------------")
        print("Expected energy:", round(result["expected_energy"], 6))
        print(
            "Feasible probability:",
            round(result["feasible_probability"], 6),
        )
        print(
            "Optimal probability:",
            round(result["optimal_probability"], 6),
        )
        print(
            "Most frequent solution:",
            describe_solution(most_frequent),
        )
        if best_feasible_index is None:
            print("Best feasible measured solution: None")
        else:
            print(
                "Best feasible measured solution:",
                describe_solution(bits_from_index(best_feasible_index)),
            )
            print(
                "Best feasible sample count:",
                int(result["counts"][best_feasible_index]),
            )

    print("\nQAOA DEPTH COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
