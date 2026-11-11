from pathlib import Path
import sys
import math


# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inventory.inventory_data import INITIAL_STOCK, load_inventory_data
from routing.customers import (
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
    load_c101,
)


def calculate_distance(location_a, location_b):
    dx = location_a["x"] - location_b["x"]
    dy = location_a["y"] - location_b["y"]
    return math.sqrt(dx ** 2 + dy ** 2)


def allocate_inventory(stock, customers):
    sorted_customers = sorted(
        customers,
        key=lambda customer: (
            customer["demand"],
            customer["customer_id"],
        ),
    )
    remaining_stock = stock
    fulfilled = []
    rejected = []

    for customer in sorted_customers:
        if customer["demand"] <= remaining_stock:
            fulfilled.append(customer)
            remaining_stock -= customer["demand"]
        else:
            rejected.append(customer)

    return fulfilled, rejected, remaining_stock


def route_fulfilled_customers(warehouse, fulfilled_customers):
    locations = [warehouse] + fulfilled_customers
    unvisited = set(range(1, len(locations)))
    routes = []

    for vehicle in range(1, VEHICLE_COUNT + 1):
        if not unvisited:
            break

        current = 0
        current_time = warehouse["ready_time"]
        load = 0
        route_distance = 0
        route = [0]

        while unvisited:
            feasible = []
            for customer_index in unvisited:
                customer = locations[customer_index]
                if load + customer["demand"] > VEHICLE_CAPACITY:
                    continue

                travel_time = calculate_distance(
                    locations[current],
                    customer,
                )
                service_start = max(
                    current_time + travel_time,
                    customer["ready_time"],
                )
                if service_start > customer["due_date"]:
                    continue

                next_time = service_start + customer["service_time"]
                return_time = next_time + calculate_distance(
                    customer,
                    warehouse,
                )
                if return_time > warehouse["due_date"]:
                    continue

                feasible.append(
                    (
                        calculate_distance(
                            locations[current],
                            customer,
                        ),
                        customer_index,
                    )
                )

            if not feasible:
                break

            _, next_customer = min(feasible)
            customer = locations[next_customer]
            travel_time = calculate_distance(
                locations[current],
                customer,
            )
            service_start = max(
                current_time + travel_time,
                customer["ready_time"],
            )
            current_time = service_start + customer["service_time"]
            route_distance += travel_time
            load += customer["demand"]
            route.append(next_customer)
            unvisited.remove(next_customer)
            current = next_customer

        if len(route) == 1:
            continue

        route_distance += calculate_distance(
            locations[current],
            warehouse,
        )
        route.append(0)
        routes.append(
            {
                "vehicle": vehicle,
                "route": route,
                "load": load,
                "distance": route_distance,
            }
        )

    return routes, sorted(unvisited), locations


def validate_routes(routes, locations):
    visited = []
    capacity_valid = True
    time_windows_valid = True
    return_valid = True

    for route_data in routes:
        route = route_data["route"]
        if route[0] != 0 or route[-1] != 0:
            return_valid = False

        current = 0
        current_time = locations[0]["ready_time"]
        load = 0
        for customer_index in route[1:-1]:
            customer = locations[customer_index]
            visited.append(customer_index)
            load += customer["demand"]
            travel_time = calculate_distance(
                locations[current],
                customer,
            )
            service_start = max(
                current_time + travel_time,
                customer["ready_time"],
            )
            if service_start > customer["due_date"]:
                time_windows_valid = False
            current_time = service_start + customer["service_time"]
            current = customer_index

        if load > VEHICLE_CAPACITY or load != route_data["load"]:
            capacity_valid = False

        if current_time + calculate_distance(
            locations[current],
            locations[0],
        ) > locations[0]["due_date"]:
            return_valid = False

    visited_set = set(visited)
    return {
        "all_fulfilled_delivered": len(visited_set)
        == len(locations) - 1,
        "duplicate_deliveries": len(visited) != len(visited_set),
        "capacity_valid": capacity_valid,
        "time_windows_valid": time_windows_valid,
        "warehouse_return_valid": return_valid,
    }


def main():
    warehouse, customers = load_c101()
    _, inventory_customers = load_inventory_data()
    fulfilled, rejected, remaining_stock = allocate_inventory(
        INITIAL_STOCK,
        inventory_customers,
    )

    routes, unvisited, locations = route_fulfilled_customers(
        warehouse,
        [
            next(
                customer
                for customer in customers
                if customer["id"] == item["customer_id"]
            )
            for item in fulfilled
        ],
    )
    validation = validate_routes(routes, locations)
    total_distance = sum(route["distance"] for route in routes)

    print("INTEGRATED INVENTORY-ROUTING BASELINE")
    print("-------------------------------------")
    print("Warehouse stock:", INITIAL_STOCK)
    print("Total customers:", len(customers))
    print("Customers fulfilled:", len(fulfilled))
    print("Customers rejected:", len(rejected))
    print("Fulfilled demand:", sum(item["demand"] for item in fulfilled))
    print("Remaining stock:", remaining_stock)
    print("\nROUTING SUMMARY")
    print("Vehicles used:", len(routes))
    print("Fulfilled customers routed:", len(fulfilled) - len(unvisited))
    print("Fulfilled customers not routed:", len(unvisited))
    print("Total distance:", round(total_distance, 2))
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))

    print("\nINTEGRATED VALIDATION")
    print("---------------------")
    print("Inventory within stock:", sum(
        item["demand"] for item in fulfilled
    ) <= INITIAL_STOCK)
    print("All fulfilled customers routed:", validation["all_fulfilled_delivered"])
    print("Duplicate deliveries:", validation["duplicate_deliveries"])
    print("Capacity valid:", validation["capacity_valid"])
    print("Customer time windows valid:", validation["time_windows_valid"])
    print("Warehouse return time valid:", validation["warehouse_return_valid"])


if __name__ == "__main__":
    main()
