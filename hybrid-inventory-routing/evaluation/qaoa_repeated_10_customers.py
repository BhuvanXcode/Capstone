import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quantum import qaoa_scaling_10_customers as qaoa


RUNS = 10
SHOTS_PER_RUN = 10_000


def main():
    print("10-CUSTOMER REPEATED QAOA EVALUATION")
    print("------------------------------------")

    energies = np.array(
        [
            qaoa.qubo_energy(qaoa.bits_from_index(index))
            for index in range(qaoa.N_STATES)
        ],
        dtype=float,
    )
    feasible_indices = [
        index
        for index in range(qaoa.N_STATES)
        if qaoa.is_feasible(qaoa.bits_from_index(index))
    ]
    classical_index = min(
        feasible_indices,
        key=lambda index: energies[index],
    )

    optimization = qaoa.optimize_parameters(energies)
    state = qaoa.qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2

    print("Customers:", qaoa.CUSTOMERS)
    print("Pairwise weight:", qaoa.PAIRWISE_WEIGHT)
    print("Qubits:", qaoa.N_QUBITS)
    print("States:", qaoa.N_STATES)
    print("Runs:", RUNS)
    print("Shots per run:", SHOTS_PER_RUN)
    print("\nCLASSICAL OPTIMUM")
    print("-----------------")
    print(qaoa.describe(classical_index, energies))

    print("\nOPTIMIZED PARAMETERS")
    print("--------------------")
    print("Gamma:", round(float(optimization.x[0]), 6))
    print("Beta:", round(float(optimization.x[1]), 6))
    print("Expected energy:", round(float(optimization.fun), 6))
    print(
        "State-vector feasible probability:",
        round(
            float(
                probabilities[
                    [
                        qaoa.is_feasible(qaoa.bits_from_index(index))
                        for index in range(qaoa.N_STATES)
                    ]
                ].sum()
            ),
            6,
        ),
    )
    print(
        "State-vector optimal probability:",
        round(float(probabilities[classical_index]), 6),
    )

    run_results = []
    for run_number in range(1, RUNS + 1):
        samples = np.random.default_rng(run_number).choice(
            qaoa.N_STATES,
            size=SHOTS_PER_RUN,
            p=probabilities,
        )
        feasible_samples = [
            int(index)
            for index in samples
            if qaoa.is_feasible(qaoa.bits_from_index(int(index)))
        ]
        best_index = min(
            feasible_samples,
            key=lambda index: energies[index],
            default=None,
        )
        feasible_probability = len(feasible_samples) / SHOTS_PER_RUN
        optimal_probability = float(
            np.count_nonzero(samples == classical_index)
            / SHOTS_PER_RUN
        )
        run_results.append(
            {
                "feasible_probability": feasible_probability,
                "optimal_probability": optimal_probability,
                "feasible_measurements": len(feasible_samples),
                "best_index": best_index,
            }
        )
        best = (
            None
            if best_index is None
            else qaoa.describe(best_index, energies)
        )
        print(
            f"Run {run_number}: "
            f"feasible_probability={feasible_probability:.6f}, "
            f"optimal_probability={optimal_probability:.6f}, "
            f"feasible_measurements={len(feasible_samples)}, "
            f"best={best}"
        )

    best_indices = [
        result["best_index"]
        for result in run_results
        if result["best_index"] is not None
    ]
    best_index = min(
        best_indices,
        key=lambda index: energies[index],
        default=None,
    )
    optimum_runs = sum(
        result["best_index"] == classical_index
        for result in run_results
    )

    print("\nREPEATED-RUN SUMMARY")
    print("--------------------")
    print(
        "Average feasible probability:",
        round(
            float(
                np.mean(
                    [
                        result["feasible_probability"]
                        for result in run_results
                    ]
                )
            ),
            6,
        ),
    )
    print(
        "Average optimal probability:",
        round(
            float(
                np.mean(
                    [
                        result["optimal_probability"]
                        for result in run_results
                    ]
                )
            ),
            6,
        ),
    )
    print(
        "Average feasible measurements:",
        round(
            float(
                np.mean(
                    [
                        result["feasible_measurements"]
                        for result in run_results
                    ]
                )
            ),
            2,
        ),
    )
    print("Runs finding classical optimum:", f"{optimum_runs}/{RUNS}")
    print(
        "Best feasible customer selection:",
        (
            None
            if best_index is None
            else qaoa.describe(best_index, energies)
        ),
    )


if __name__ == "__main__":
    main()
