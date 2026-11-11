import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quantum import qaoa_scaling_10_customers as qaoa


RUNS = 10
SHOTS_PER_RUN = 10_000


def main():
    print("FINAL CLASSICAL VS QAOA COMPARISON")
    print("==================================")

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
    classical_energy = float(energies[classical_index])
    classical_bits = qaoa.bits_from_index(classical_index)
    classical_customers = qaoa.selected_customers(classical_bits)

    # Reuse optimized parameters from Phase 11.
    # Avoid repeating expensive differential evolution.
    class OptimizationResult:
        x = np.array([1.04556, 0.566137])

    optimization = OptimizationResult()
    state = qaoa.qaoa_state(optimization.x, energies)
    probabilities = np.abs(state) ** 2
    optimization.fun = float(np.dot(probabilities, energies))

    feasible_mask = np.array(
        [
            qaoa.is_feasible(qaoa.bits_from_index(index))
            for index in range(qaoa.N_STATES)
        ]
    )

    all_samples = []
    run_summaries = []
    for run_number in range(1, RUNS + 1):
        samples = np.random.default_rng(run_number).choice(
            qaoa.N_STATES,
            size=SHOTS_PER_RUN,
            p=probabilities,
        )
        all_samples.extend(int(index) for index in samples)
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
        run_summaries.append(
            {
                "run": run_number,
                "feasible_measurements": len(feasible_samples),
                "optimal_measurements": int(
                    np.count_nonzero(samples == classical_index)
                ),
                "best_index": best_index,
            }
        )

    best_qaoa_index = min(
        (
            index
            for summary in run_summaries
            for index in [summary["best_index"]]
            if index is not None
        ),
        key=lambda index: energies[index],
        default=None,
    )
    total_measurements = len(all_samples)
    total_feasible = sum(
        qaoa.is_feasible(qaoa.bits_from_index(index))
        for index in all_samples
    )
    total_optimal = sum(index == classical_index for index in all_samples)

    print("\nCLASSICAL OPTIMUM")
    print("-----------------")
    print("Selected customers:", classical_customers)
    print("Demand:", qaoa.demand_for(classical_bits))
    print("Slack:", qaoa.slack_for(classical_bits))
    print("Energy:", round(classical_energy, 6))
    print("Feasible states:", len(feasible_indices))

    print("\nQAOA CONFIGURATION")
    print("------------------")
    print("Customers:", qaoa.CUSTOMERS)
    print("Qubits:", qaoa.N_QUBITS)
    print("States:", qaoa.N_STATES)
    print("Pairwise weight:", qaoa.PAIRWISE_WEIGHT)
    print("Gamma:", round(float(optimization.x[0]), 6))
    print("Beta:", round(float(optimization.x[1]), 6))
    print("Expected energy:", round(float(optimization.fun), 6))
    print(
        "State-vector feasible probability:",
        round(float(probabilities[feasible_mask].sum()), 6),
    )
    print(
        "State-vector optimal probability:",
        round(float(probabilities[classical_index]), 6),
    )

    print("\nQAOA SAMPLING RESULTS")
    print("--------------------")
    print("Runs:", RUNS)
    print("Shots per run:", SHOTS_PER_RUN)
    print("Total measurements:", total_measurements)
    print(
        "Total feasible measurements:",
        total_feasible,
    )
    print(
        "Total optimal measurements:",
        total_optimal,
    )
    print(
        "Observed feasible probability:",
        round(total_feasible / total_measurements, 6),
    )
    print(
        "Observed optimal probability:",
        round(total_optimal / total_measurements, 6),
    )
    print("Runs finding classical optimum:", sum(
        summary["best_index"] == classical_index
        for summary in run_summaries
    ), f"/{RUNS}")

    if best_qaoa_index is None:
        print("Best feasible QAOA result: None")
    else:
        print(
            "Best feasible QAOA result:",
            qaoa.describe(best_qaoa_index, energies),
        )

    print("\nFINAL COMPARISON")
    print("----------------")
    print("Classical customers:", classical_customers)
    print("Classical energy:", round(classical_energy, 6))
    if best_qaoa_index is None:
        print("QAOA customers: None")
        print("QAOA energy: None")
        print("QAOA matches classical optimum: False")
    else:
        best_qaoa = qaoa.describe(best_qaoa_index, energies)
        print("QAOA customers:", best_qaoa["customers"])
        print("QAOA energy:", round(best_qaoa["energy"], 6))
        print(
            "QAOA matches classical optimum:",
            best_qaoa_index == classical_index,
        )

    print("\nPER-RUN DETAILS")
    print("---------------")
    for summary in run_summaries:
        best = (
            None
            if summary["best_index"] is None
            else qaoa.describe(summary["best_index"], energies)
        )
        print(
            f"Run {summary['run']}: "
            f"feasible={summary['feasible_measurements']}, "
            f"optimal={summary['optimal_measurements']}, "
            f"best={best}"
        )


if __name__ == "__main__":
    main()
