import math
import random

from customers import (
    load_c101,
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
)


ANT_COUNT = 30
ITERATIONS = 80
ALPHA = 1.0
BETA = 3.0
EVAPORATION_RATE = 0.5
PHEROMONE_DEPOSIT = 100.0
UNSERVED_PENALTY = 1_000_000
VEHICLE_PENALTY = 1000


def distance(location_a, location_b):
    dx = location_a["x"] - location_b["x"]
    dy = location_a["y"] - location_b["y"]
    return math.sqrt(dx * dx + dy * dy)


def decode_ant_route(order, locations):
    warehouse = locations[0]
    remaining = list(order)
    routes = []

    for vehicle in range(1, VEHICLE_COUNT + 1):
        if not remaining:
            break

        current = 0
        current_time = warehouse["ready_time"]
        route_load = 0
        route_distance = 0
        route = [0]
        served = []

        for customer_index in list(remaining):
            customer = locations[customer_index]
            if route_load + customer["demand"] > VEHICLE_CAPACITY:
                continue

            travel_time = distance(locations[current], customer)
            service_start = max(
                current_time + travel_time,
                customer["ready_time"],
            )
            if service_start > customer["due_date"]:
                continue

            next_time = service_start + customer["service_time"]
            return_time = next_time + distance(customer, warehouse)
            if return_time > warehouse["due_date"]:
                continue

            route_distance += travel_time
            current_time = next_time
            route_load += customer["demand"]
            route.append(customer_index)
            served.append(customer_index)
            current = customer_index

        if not served:
            continue

        route_distance += distance(locations[current], warehouse)
        route.append(0)
        routes.append(
            {
                "vehicle": vehicle,
                "route": route,
                "load": route_load,
                "distance": route_distance,
            }
        )
        for customer_index in served:
            remaining.remove(customer_index)

    return routes, sorted(remaining)


def route_fitness(routes, unserved):
    total_distance = sum(route["distance"] for route in routes)
    return (
        len(unserved) * UNSERVED_PENALTY
        + len(routes) * VEHICLE_PENALTY
        + total_distance
    )


def select_next_customer(current, candidates, pheromones, locations):
    weights = []
    for candidate in candidates:
        trail = pheromones[current][candidate] ** ALPHA
        visibility = (1 / max(distance(
            locations[current],
            locations[candidate],
        ), 1e-12)) ** BETA
        weights.append(trail * visibility)

    return random.choices(candidates, weights=weights, k=1)[0]


def build_ant_order(locations, pheromones):
    order = []
    candidates = list(range(1, len(locations)))
    current = 0

    while candidates:
        next_customer = select_next_customer(
            current,
            candidates,
            pheromones,
            locations,
        )
        order.append(next_customer)
        candidates.remove(next_customer)
        current = next_customer

    return order


def update_pheromones(pheromones, ant_solutions, best_solution):
    for row in pheromones:
        for index in range(len(row)):
            row[index] *= 1 - EVAPORATION_RATE

    solutions = ant_solutions + [best_solution]
    for routes, unserved in solutions:
        fitness = route_fitness(routes, unserved)
        if fitness <= 0:
            continue
        deposit = PHEROMONE_DEPOSIT / fitness
        for route in routes:
            for first, second in zip(
                route["route"],
                route["route"][1:],
            ):
                pheromones[first][second] += deposit
                pheromones[second][first] += deposit


def validate_routes(routes, locations):
    expected = set(range(1, len(locations)))
    visited = []
    capacity_valid = True
    time_windows_valid = True
    warehouse_return_valid = True
    route_structure_valid = True

    for route_data in routes:
        route = route_data["route"]
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            route_structure_valid = False
            continue

        load = 0
        current = 0
        current_time = locations[0]["ready_time"]
        for customer_index in route[1:-1]:
            customer = locations[customer_index]
            visited.append(customer_index)
            load += customer["demand"]
            travel_time = distance(locations[current], customer)
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
        return_time = current_time + distance(
            locations[current],
            locations[0],
        )
        if return_time > locations[0]["due_date"]:
            warehouse_return_valid = False

    visited_set = set(visited)
    return {
        "all_customers_visited": visited_set == expected,
        "duplicate_visits": len(visited) != len(visited_set),
        "capacity_valid": capacity_valid,
        "time_windows_valid": time_windows_valid,
        "warehouse_return_valid": warehouse_return_valid,
        "route_structure_valid": route_structure_valid,
    }


def solve_ant_colony():
    warehouse, customers = load_c101()
    locations = [warehouse] + customers
    size = len(locations)
    pheromones = [[1.0] * size for _ in range(size)]
    best_solution = ([], list(range(1, size)))
    best_fitness = route_fitness(*best_solution)

    print("ANT COLONY OPTIMIZATION STARTED")
    for iteration in range(1, ITERATIONS + 1):
        ant_solutions = []
        for _ in range(ANT_COUNT):
            order = build_ant_order(locations, pheromones)
            solution = decode_ant_route(order, locations)
            ant_solutions.append(solution)
            solution_fitness = route_fitness(*solution)
            if solution_fitness < best_fitness:
                best_solution = solution
                best_fitness = solution_fitness

        update_pheromones(pheromones, ant_solutions, best_solution)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {best_fitness:.2f}")

    routes, unserved = best_solution
    total_distance = sum(route["distance"] for route in routes)
    print("\nANT COLONY OPTIMIZATION RESULTS")
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))

    validation = validate_routes(routes, locations)
    print("\nFINAL ACO SUMMARY")
    print("Customers served:", len(customers) - len(unserved))
    print("Customers unserved:", len(unserved))
    print("Vehicles used:", len(routes))
    print("Total distance:", round(total_distance, 2))
    print("\nINDEPENDENT VALIDATION")
    print("----------------------")
    print("All customers visited:", validation["all_customers_visited"])
    print("Duplicate customer visits:", validation["duplicate_visits"])
    print("Capacity valid:", validation["capacity_valid"])
    print("Customer time windows valid:", validation["time_windows_valid"])
    print("Warehouse return time valid:", validation["warehouse_return_valid"])
    print("Route structure valid:", validation["route_structure_valid"])
    validation_passed = (
        validation["all_customers_visited"]
        and not validation["duplicate_visits"]
        and validation["capacity_valid"]
        and validation["time_windows_valid"]
        and validation["warehouse_return_valid"]
        and validation["route_structure_valid"]
    )
    print("ACO VALIDATION PASSED:", validation_passed)

    return routes, unserved


if __name__ == "__main__":
    random.seed(42)
    solve_ant_colony()
