import math
import random

from customers import (
    load_c101,
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
)


GENERATIONS = 60
POPULATION_SIZE = 50
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def calculate_route_details(route, locations):
    if len(route) < 3 or route[0] != 0 or route[-1] != 0:
        return None

    warehouse = locations[0]
    current = 0
    current_time = warehouse["ready_time"]
    total_distance = 0
    total_load = 0

    for customer_index in route[1:-1]:
        customer = locations[customer_index]
        if total_load + customer["demand"] > VEHICLE_CAPACITY:
            return None

        travel_distance = distance(locations[current], customer)
        arrival_time = current_time + travel_distance
        service_start = max(arrival_time, customer["ready_time"])
        if service_start > customer["due_date"]:
            return None

        total_distance += travel_distance
        current_time = service_start + customer["service_time"]
        total_load += customer["demand"]
        current = customer_index

    return_distance = distance(locations[current], warehouse)
    total_distance += return_distance
    return_time = current_time + return_distance
    if return_time > warehouse["due_date"]:
        return None

    return {
        "distance": total_distance,
        "load": total_load,
    }


def improve_route_2opt(route, locations):
    best_route = route.copy()
    best_details = calculate_route_details(best_route, locations)
    if best_details is None:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = (
                    best_route[:i]
                    + best_route[i:j + 1][::-1]
                    + best_route[j + 1:]
                )
                candidate_details = calculate_route_details(
                    candidate,
                    locations,
                )
                if (
                    candidate_details is not None
                    and candidate_details["distance"]
                    < best_details["distance"]
                ):
                    best_route = candidate
                    best_details = candidate_details
                    improved = True
                    break
            if improved:
                break

    return best_route


def decode_chromosome(chromosome, locations):
    warehouse = locations[0]
    remaining = list(chromosome)
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
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, customer["ready_time"])
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

    for route_data in routes:
        improved_route = improve_route_2opt(
            route_data["route"],
            locations,
        )
        route_data["route"] = improved_route
        details = calculate_route_details(improved_route, locations)
        if details is not None:
            route_data["distance"] = details["distance"]
            route_data["load"] = details["load"]

    return routes, sorted(remaining)


def fitness(chromosome, locations):
    routes, unserved = decode_chromosome(chromosome, locations)
    total_distance = sum(route["distance"] for route in routes)
    penalty = len(unserved) * 1_000_000
    vehicle_cost = len(routes) * 1000
    return penalty + vehicle_cost + total_distance


def tournament_selection(population, locations):
    candidates = random.sample(
        population,
        min(TOURNAMENT_SIZE, len(population)),
    )
    return min(
        candidates,
        key=lambda chromosome: fitness(chromosome, locations),
    )


def ordered_crossover(parent_a, parent_b):
    start, end = sorted(random.sample(range(len(parent_a)), 2))
    child = [None] * len(parent_a)
    child[start:end] = parent_a[start:end]
    remaining = [
        customer for customer in parent_b if customer not in child
    ]

    remaining_index = 0
    for index, value in enumerate(child):
        if value is None:
            child[index] = remaining[remaining_index]
            remaining_index += 1

    return child


def mutate(chromosome):
    operation = random.choice([
        "swap",
        "inversion",
        "insertion",
    ])

    if operation == "swap":
        first, second = random.sample(range(len(chromosome)), 2)
        chromosome[first], chromosome[second] = (
            chromosome[second],
            chromosome[first],
        )
    elif operation == "inversion":
        first, second = sorted(
            random.sample(range(len(chromosome)), 2)
        )
        chromosome[first:second + 1] = (
            chromosome[first:second + 1][::-1]
        )
    elif operation == "insertion":
        first, second = random.sample(range(len(chromosome)), 2)
        customer = chromosome.pop(first)
        chromosome.insert(second, customer)


def validate_routes(routes, customers, locations):
    expected_customers = set(range(1, len(locations)))
    visited_customers = []
    capacity_valid = True
    time_windows_valid = True
    warehouse_return_valid = True
    route_structure_valid = True

    for route_data in routes:
        route = route_data["route"]
        load = route_data["load"]

        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            route_structure_valid = False
            continue

        if load > VEHICLE_CAPACITY:
            capacity_valid = False

        current = 0
        current_time = locations[0]["ready_time"]
        calculated_load = 0

        for customer_index in route[1:-1]:
            customer = locations[customer_index]
            visited_customers.append(customer_index)
            calculated_load += customer["demand"]

            travel_time = distance(locations[current], customer)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, customer["ready_time"])
            if service_start > customer["due_date"]:
                time_windows_valid = False

            current_time = service_start + customer["service_time"]
            current = customer_index

        if calculated_load != load:
            capacity_valid = False

        return_time = current_time + distance(
            locations[current],
            locations[0],
        )
        if return_time > locations[0]["due_date"]:
            warehouse_return_valid = False

    visited_set = set(visited_customers)
    all_customers_visited = visited_set == expected_customers
    duplicate_visits = len(visited_customers) != len(visited_set)

    return {
        "all_customers_visited": all_customers_visited,
        "duplicate_visits": duplicate_visits,
        "capacity_valid": capacity_valid,
        "time_windows_valid": time_windows_valid,
        "warehouse_return_valid": warehouse_return_valid,
        "route_structure_valid": route_structure_valid,
    }


def solve_genetic_algorithm():
    warehouse, customers = load_c101()
    locations = [warehouse] + customers
    customer_indices = list(range(1, len(locations)))
    population = []

    for _ in range(POPULATION_SIZE):
        chromosome = customer_indices.copy()
        random.shuffle(chromosome)
        population.append(chromosome)

    best_chromosome = min(
        population,
        key=lambda chromosome: fitness(chromosome, locations),
    )

    print("GENETIC ALGORITHM STARTED")
    for generation in range(1, GENERATIONS + 1):
        next_population = [best_chromosome.copy()]

        while len(next_population) < POPULATION_SIZE:
            parent_a = tournament_selection(population, locations)
            parent_b = tournament_selection(population, locations)
            child = ordered_crossover(parent_a, parent_b)
            if random.random() < MUTATION_RATE:
                mutate(child)
            next_population.append(child)

        population = next_population
        generation_best = min(
            population,
            key=lambda chromosome: fitness(chromosome, locations),
        )
        if fitness(generation_best, locations) < fitness(
            best_chromosome,
            locations,
        ):
            best_chromosome = generation_best.copy()

        if generation % 10 == 0:
            print(
                f"Generation {generation}: "
                f"Best fitness = "
                f"{fitness(best_chromosome, locations):.2f}"
            )

    routes, unserved = decode_chromosome(best_chromosome, locations)
    total_distance = sum(route["distance"] for route in routes)
    best_fitness = fitness(best_chromosome, locations)

    print("\nGENETIC ALGORITHM RESULTS")
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])
        print("Distance:", round(route["distance"], 2))

    print("\nFINAL GA SUMMARY")
    print("Customers served:", len(customers) - len(unserved))
    print("Customers unserved:", len(unserved))
    print("Vehicles used:", len(routes))
    print("Total distance:", round(total_distance, 2))

    validation = validate_routes(routes, customers, locations)
    print("\nINDEPENDENT VALIDATION")
    print("----------------------")
    print("All customers visited:", validation["all_customers_visited"])
    print("Duplicate customer visits:", validation["duplicate_visits"])
    print("Capacity valid:", validation["capacity_valid"])
    print("Customer time windows valid:", validation["time_windows_valid"])
    print("Warehouse return time valid:", validation["warehouse_return_valid"])
    print("Route structure valid:", validation["route_structure_valid"])

    expected_fitness = (
        len(unserved) * 1_000_000
        + len(routes) * 1000
        + total_distance
    )
    print("\nFITNESS VERIFICATION")
    print("Recalculated fitness:", round(expected_fitness, 2))
    print("Stored best fitness:", round(best_fitness, 2))

    return routes, unserved


if __name__ == "__main__":
    random.seed(42)
    solve_genetic_algorithm()
