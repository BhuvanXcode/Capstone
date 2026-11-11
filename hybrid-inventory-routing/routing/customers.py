from pathlib import Path


# Locate the C101 dataset
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "c101.txt"

# Solomon C101 vehicle details
VEHICLE_COUNT = 25
VEHICLE_CAPACITY = 200


def load_c101():
    customers = []
    warehouse = None

    with open(DATA_FILE, "r") as file:
        lines = file.readlines()

    reading_data = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue

        # Start reading after the column headings
        if parts[0] == "CUST":
            reading_data = True
            continue

        if not reading_data or len(parts) < 7:
            continue

        customer = {
            "id": int(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "demand": int(parts[3]),
            "ready_time": int(parts[4]),
            "due_date": int(parts[5]),
            "service_time": int(parts[6]),
        }

        if customer["id"] == 0:
            warehouse = customer
        else:
            customers.append(customer)

    return warehouse, customers


if __name__ == "__main__":
    warehouse, customers = load_c101()
    print("C101 Dataset Loaded Successfully")
    print("Total customers:", len(customers))
    print("Total customer demand:", sum(c["demand"] for c in customers))
    print("Vehicle count:", VEHICLE_COUNT)
    print("Vehicle capacity:", VEHICLE_CAPACITY)
