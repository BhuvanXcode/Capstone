
from pathlib import Path

from classical_routing import (
    load_solomon_data,
    solve_vrptw,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "c101.txt"


def run_classical_solver(dataset_path=DEFAULT_DATASET):
    """Load a Solomon VRPTW dataset and solve it."""
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataset = load_solomon_data(dataset_path)

    print("Loaded dataset:", dataset_path.name)
    print("Number of customer records:", len(dataset["customers"]))
    print("Vehicles available:", dataset["vehicle_count"])
    print("Vehicle capacity:", dataset["vehicle_capacities"][0])

    return solve_vrptw(dataset)


if __name__ == "__main__":
    run_classical_solver()
