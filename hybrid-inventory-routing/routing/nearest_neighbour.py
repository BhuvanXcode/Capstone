import math

from customers import (
    load_c101,
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
)


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def nearest_neighbour():
    warehouse, customers = load_c101()
    locations = [warehouse] + customers
    unvisited = set(range(1, len(locations)))
    routes = []

    for vehicle in range(1, VEHICLE_COUNT + 1):
        current = 0
        current_time = warehouse["ready_time"]
        load = 0
        route = [0]
        route_distance = 0

        while unvisited:
            feasible_customers = []
            for customer_index in unvisited:
                customer = locations[customer_index]
                if load + customer["demand"] > VEHICLE_CAPACITY:
                    continue

                travel_time = distance(locations[current], customer)
                arrival_time = current_time + travel_time
                service_start = max(arrival_time, customer["ready_time"])
                if service_start <= customer["due_date"]:
                    feasible_customers.append(
                        (distance(locations[current], customer), customer_index)
                    )

            if not feasible_customers:
                break

            _, next_customer = min(feasible_customers)
            customer = locations[next_customer]
            travel_distance = distance(locations[current], customer)
            arrival_time = current_time + travel_distance
            current_time = max(arrival_time, customer["ready_time"])
            current_time += customer["service_time"]
            route_distance += travel_distance
            route.append(next_customer)
            load += customer["demand"]
            unvisited.remove(next_customer)
            current = next_customer

        if len(route) > 1:
            route_distance += distance(locations[current], warehouse)
            route.append(0)
            routes.append(
                {
                    "vehicle": vehicle,
                    "route": route,
                    "load": load,
                    "distance": route_distance,
                }
            )

        if not unvisited:
            break

    return routes, sorted(unvisited)


if __name__ == "__main__":
    routes, unvisited = nearest_neighbour()
    print("Nearest Neighbour Results")
    print("-------------------------")
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))
    print("\nCustomers not served:", len(unvisited))
