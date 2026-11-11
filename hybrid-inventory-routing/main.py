import sys
from pathlib import Path


ROUTING_DIRECTORY = Path(__file__).resolve().parent / "routing"
sys.path.insert(0, str(ROUTING_DIRECTORY))

from customers import (  # noqa: E402
    VEHICLE_CAPACITY,
    VEHICLE_COUNT,
    load_c101,
)
from distance_matrix import create_distance_matrix  # noqa: E402
from nearest_neighbour import nearest_neighbour  # noqa: E402
from ortools_model import solve_ortools  # noqa: E402


def main():
    print("HYBRID INVENTORY ROUTING PROJECT")

    warehouse, customers = load_c101()
    locations = [warehouse] + customers

    print("\nSTEP 1: LOADING C101 DATASET")
    print("Dataset loaded successfully")
    print("Total customers:", len(customers))
    print("Total customer demand:", sum(c["demand"] for c in customers))
    print("Number of vehicles:", VEHICLE_COUNT)
    print("Vehicle capacity:", VEHICLE_CAPACITY)

    print("\nSTEP 2: CREATING DISTANCE MATRIX")
    matrix = create_distance_matrix()
    print("Total locations:", len(locations))
    print("Matrix rows:", len(matrix))
    print("Matrix columns:", len(matrix[0]))
    print("\nSample distances:")
    print("Warehouse to Customer 1:", matrix[0][1])
    print("Warehouse to Customer 2:", matrix[0][2])

    print("\nSTEP 3: RUNNING NEAREST NEIGHBOUR")
    routes, unvisited = nearest_neighbour()
    total_distance = 0
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))
        total_distance += route["distance"]

    print("\nROUTING SUMMARY")
    print("Vehicles used:", len(routes))
    print("Total distance:", round(total_distance, 2))
    print("Customers served:", len(customers) - len(unvisited))
    print("Customers not served:", len(unvisited))
    if unvisited:
        print(
            "Unvisited customer IDs:",
            sorted(locations[node]["id"] for node in unvisited),
        )
    print("\nINITIAL ROUTING CHECK COMPLETED")

    # STEP 4: Run OR-Tools
    print("\nSTEP 4: RUNNING OR-TOOLS VRPTW")
    ortools_result = solve_ortools()


if __name__ == "__main__":
    main()
