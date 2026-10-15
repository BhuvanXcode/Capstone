
from pathlib import Path

from src.data.loader import load_customers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "c101.txt"


def load_classical_data(file_path=DEFAULT_DATASET):
    """Convert loaded customer records into the OR-Tools data format."""
    customers_by_id = load_customers(file_path)
    customers = [
        customers_by_id[customer_id]
        for customer_id in sorted(customers_by_id)
    ]

    vehicle_header = None
    lines = Path(file_path).read_text(encoding="utf-8-sig").splitlines()

    for index, line in enumerate(lines):
        if "NUMBER" in line.upper() and "CAPACITY" in line.upper():
            for candidate in lines[index + 1:]:
                parts = candidate.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    vehicle_header = (
                        int(parts[0]),
                        int(float(parts[1])),
                    )
                    break
            break

    if vehicle_header is None:
        raise ValueError("Could not read vehicle count and capacity.")

    vehicle_count, vehicle_capacity = vehicle_header

    return {
        "customers": customers,
        "vehicle_count": vehicle_count,
        "vehicle_capacities": [vehicle_capacity] * vehicle_count,
        "depot": 0,
    }
