
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "c101.txt"


def load_customers(file_path=DEFAULT_DATASET):
    """Load depot and customer records from a Solomon VRPTW dataset."""
    customers = {}

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) == 7 and parts[0].isdigit():
                customer_id = int(parts[0])
                customers[customer_id] = {
                    "id": customer_id,
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "demand": float(parts[3]),
                    "ready_time": float(parts[4]),
                    "due_time": float(parts[5]),
                    "service_time": float(parts[6]),
                }

    if not customers:
        raise ValueError(f"No customer records found in {file_path}")

    if 0 not in customers:
        raise ValueError("Dataset does not contain depot ID 0.")

    return customers


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two customer records."""
    return math.hypot(
        point1["x"] - point2["x"],
        point1["y"] - point2["y"],
    )


def create_distance_matrix(customers):
    """Build a distance matrix ordered by customer ID."""
    customer_ids = sorted(customers)

    matrix = [
        [
            calculate_distance(customers[i], customers[j])
            for j in customer_ids
        ]
        for i in customer_ids
    ]

    return customer_ids, matrix
