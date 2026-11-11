import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSICAL_ENERGY = -774.378753
QAOA_ENERGY = -774.378753
TOTAL_MEASUREMENTS = 100_000
FEASIBLE_MEASUREMENTS = 984
OPTIMAL_MEASUREMENTS = 5
RUNS = list(range(1, 11))
FEASIBLE_PER_RUN = [83, 92, 108, 96, 98, 94, 114, 109, 103, 87]
OPTIMAL_PER_RUN = [1, 1, 2, 0, 0, 1, 0, 0, 0, 0]


def save_energy_plot():
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        ["Classical optimum", "QAOA best"],
        [CLASSICAL_ENERGY, QAOA_ENERGY],
        color=["steelblue", "darkorange"],
    )
    axis.set_title("Classical Optimum vs QAOA Best Energy")
    axis.set_ylabel("QUBO energy")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        os.path.join(RESULTS_DIR, "classical_vs_qaoa.png"),
        dpi=150,
    )
    plt.close(figure)


def save_measurement_plot(filename, values, title, ylabel, color):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(RUNS, values, color=color)
    axis.set_title(title)
    axis.set_xlabel("Measurement run")
    axis.set_ylabel(ylabel)
    axis.set_xticks(RUNS)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=150,
    )
    plt.close(figure)


def save_probability_plot():
    feasible_probability = FEASIBLE_MEASUREMENTS / TOTAL_MEASUREMENTS
    optimal_probability = OPTIMAL_MEASUREMENTS / TOTAL_MEASUREMENTS
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        ["Feasible", "Optimal"],
        [feasible_probability, optimal_probability],
        color=["seagreen", "firebrick"],
    )
    axis.set_title("QAOA Sampling Probabilities")
    axis.set_ylabel("Observed probability")
    axis.grid(axis="y", alpha=0.3)
    for position, value in enumerate(
        [feasible_probability, optimal_probability]
    ):
        axis.text(
            position,
            value,
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(
        os.path.join(RESULTS_DIR, "qaoa_probabilities.png"),
        dpi=150,
    )
    plt.close(figure)


def save_csv_files():
    run_path = os.path.join(RESULTS_DIR, "qaoa_run_results.csv")
    with open(run_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "run",
                "feasible_measurements",
                "optimal_measurements",
                "shots",
                "feasible_probability",
                "optimal_probability",
            ]
        )
        for run, feasible, optimal in zip(
            RUNS,
            FEASIBLE_PER_RUN,
            OPTIMAL_PER_RUN,
        ):
            writer.writerow(
                [
                    run,
                    feasible,
                    optimal,
                    TOTAL_MEASUREMENTS // len(RUNS),
                    f"{feasible / (TOTAL_MEASUREMENTS // len(RUNS)):.6f}",
                    f"{optimal / (TOTAL_MEASUREMENTS // len(RUNS)):.6f}",
                ]
            )

    comparison_path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    with open(
        comparison_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "energy",
                "selected_customers",
                "demand",
                "slack",
                "total_measurements",
                "feasible_measurements",
                "optimal_measurements",
            ]
        )
        writer.writerow(
            [
                "classical",
                f"{CLASSICAL_ENERGY:.6f}",
                "[3, 4, 5, 6, 8, 9, 10]",
                100,
                0,
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "qaoa_best",
                f"{QAOA_ENERGY:.6f}",
                "[3, 4, 5, 6, 8, 9, 10]",
                100,
                0,
                TOTAL_MEASUREMENTS,
                FEASIBLE_MEASUREMENTS,
                OPTIMAL_MEASUREMENTS,
            ]
        )


def main():
    save_energy_plot()
    save_measurement_plot(
        "feasible_measurements.png",
        FEASIBLE_PER_RUN,
        "Feasible Measurements per QAOA Run",
        "Feasible measurements",
        "seagreen",
    )
    save_measurement_plot(
        "optimal_measurements.png",
        OPTIMAL_PER_RUN,
        "Optimal Measurements per QAOA Run",
        "Optimal measurements",
        "firebrick",
    )
    save_probability_plot()
    save_csv_files()

    print("FINAL RESULTS GENERATED")
    print("-----------------------")
    for filename in (
        "classical_vs_qaoa.png",
        "feasible_measurements.png",
        "optimal_measurements.png",
        "qaoa_probabilities.png",
        "qaoa_run_results.csv",
        "final_comparison.csv",
    ):
        path = os.path.join(RESULTS_DIR, filename)
        print(f"{filename}: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
