import math
import sys
from pathlib import Path

import pulp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from routing.customers import load_c101


warehouse, all_customers = load_c101()
customers = all_customers[:8]

STOCK = 120
VEHICLE_CAPACITY = 70
VEHICLE_COUNT = 3
REJECTION_PENALTY = 1000
VEHICLE_FIXED_COST = 50
BIG_M = 10_000

N = [customer["id"] for customer in customers]
V = [0] + N
K = range(VEHICLE_COUNT)
locations = {0: warehouse}
locations.update({customer["id"]: customer for customer in customers})


def calculate_distance(first, second):
    return math.hypot(
        first["x"] - second["x"],
        first["y"] - second["y"],
    )


def build_model():
    model = pulp.LpProblem(
        "C101_Integrated_Inventory_Routing",
        pulp.LpMinimize,
    )

    arcs = [
        (i, j, vehicle)
        for i in V
        for j in V
        if i != j
        for vehicle in K
    ]
    x = {
        arc: pulp.LpVariable(
            f"x_{arc[0]}_{arc[1]}_{arc[2]}",
            cat=pulp.LpBinary,
        )
        for arc in arcs
    }
    fulfilled = {
        customer_id: pulp.LpVariable(
            f"fulfilled_{customer_id}",
            cat=pulp.LpBinary,
        )
        for customer_id in N
    }
    assigned = {
        (customer_id, vehicle): pulp.LpVariable(
            f"assigned_{customer_id}_{vehicle}",
            cat=pulp.LpBinary,
        )
        for customer_id in N
        for vehicle in K
    }
    used = {
        vehicle: pulp.LpVariable(
            f"used_{vehicle}",
            cat=pulp.LpBinary,
        )
        for vehicle in K
    }
    service_start = {
        (customer_id, vehicle): pulp.LpVariable(
            f"service_start_{customer_id}_{vehicle}",
            lowBound=0,
        )
        for customer_id in N
        for vehicle in K
    }
    departure = {
        vehicle: pulp.LpVariable(
            f"departure_{vehicle}",
            lowBound=0,
            upBound=warehouse["due_date"],
        )
        for vehicle in K
    }
    return_time = {
        vehicle: pulp.LpVariable(
            f"return_{vehicle}",
            lowBound=0,
            upBound=warehouse["due_date"],
        )
        for vehicle in K
    }
    order = {
        (customer_id, vehicle): pulp.LpVariable(
            f"order_{customer_id}_{vehicle}",
            lowBound=0,
            upBound=len(N),
        )
        for customer_id in N
        for vehicle in K
    }

    transport_cost = pulp.lpSum(
            calculate_distance(locations[i], locations[j])
            * x[i, j, vehicle]
            for i, j, vehicle in arcs
        )
    rejection_cost = pulp.lpSum(
            REJECTION_PENALTY * (1 - fulfilled[customer_id])
            for customer_id in N
    )
    vehicle_cost = pulp.lpSum(
        VEHICLE_FIXED_COST * used[vehicle] for vehicle in K
    )
    time_tie_breaker = 0.001
    model += (
        transport_cost
        + rejection_cost
        + vehicle_cost
        + time_tie_breaker * pulp.lpSum(return_time[vehicle] for vehicle in K)
    )

    model += pulp.lpSum(
        locations[customer_id]["demand"] * fulfilled[customer_id]
        for customer_id in N
    ) <= STOCK, "InventoryLimit"

    for customer_id in N:
        model += pulp.lpSum(
            assigned[customer_id, vehicle] for vehicle in K
        ) == fulfilled[customer_id], f"CustomerAssignment_{customer_id}"

    for vehicle in K:
        model += pulp.lpSum(
            locations[customer_id]["demand"]
            * assigned[customer_id, vehicle]
            for customer_id in N
        ) <= VEHICLE_CAPACITY * used[vehicle], f"Capacity_{vehicle}"

        model += pulp.lpSum(
            x[0, customer_id, vehicle] for customer_id in N
        ) == used[vehicle], f"DepotDeparture_{vehicle}"
        model += pulp.lpSum(
            x[customer_id, 0, vehicle] for customer_id in N
        ) == used[vehicle], f"DepotReturn_{vehicle}"

        model += departure[vehicle] >= warehouse["ready_time"] * used[vehicle]
        model += departure[vehicle] <= (
            warehouse["due_date"] * used[vehicle]
            + BIG_M * (1 - used[vehicle])
        )
        model += return_time[vehicle] <= (
            warehouse["due_date"] * used[vehicle]
            + BIG_M * (1 - used[vehicle])
        )
        model += return_time[vehicle] >= departure[vehicle]

        for customer_id in N:
            incoming = pulp.lpSum(
                x[origin, customer_id, vehicle]
                for origin in V
                if origin != customer_id
            )
            outgoing = pulp.lpSum(
                x[customer_id, destination, vehicle]
                for destination in V
                if destination != customer_id
            )
            model += (
                incoming == assigned[customer_id, vehicle]
            ), f"Incoming_{customer_id}_{vehicle}"
            model += (
                outgoing == assigned[customer_id, vehicle]
            ), f"Outgoing_{customer_id}_{vehicle}"

            customer = locations[customer_id]
            model += service_start[customer_id, vehicle] >= (
                customer["ready_time"]
                - BIG_M * (1 - assigned[customer_id, vehicle])
            )
            model += service_start[customer_id, vehicle] <= (
                customer["due_date"]
                + BIG_M * (1 - assigned[customer_id, vehicle])
            )
            model += (
                order[customer_id, vehicle]
                >= assigned[customer_id, vehicle]
            )
            model += (
                order[customer_id, vehicle]
                <= len(N) * assigned[customer_id, vehicle]
            )

            model += service_start[customer_id, vehicle] >= (
                departure[vehicle]
                + calculate_distance(locations[0], customer)
                - BIG_M * (1 - x[0, customer_id, vehicle])
            )

        for origin in N:
            for destination in N:
                if origin == destination:
                    continue
                model += service_start[destination, vehicle] >= (
                    service_start[origin, vehicle]
                    + locations[origin]["service_time"]
                    + calculate_distance(
                        locations[origin],
                        locations[destination],
                    )
                    - BIG_M * (1 - x[origin, destination, vehicle])
                )
                model += order[destination, vehicle] >= (
                    order[origin, vehicle]
                    + 1
                    - len(N) * (1 - x[origin, destination, vehicle])
                )

        for customer_id in N:
            model += return_time[vehicle] >= (
                service_start[customer_id, vehicle]
                + locations[customer_id]["service_time"]
                + calculate_distance(
                    locations[customer_id],
                    locations[0],
                )
                - BIG_M * (1 - x[customer_id, 0, vehicle])
            )

    return model, x, fulfilled, assigned, used, service_start, return_time


def extract_routes(x, used):
    routes = []
    for vehicle in K:
        if pulp.value(used[vehicle]) < 0.5:
            continue

        successors = {
            origin: destination
            for origin in V
            for destination in V
            if origin != destination
            and pulp.value(x[origin, destination, vehicle]) > 0.5
        }
        route = [0]
        current = 0
        while current in successors:
            current = successors[current]
            route.append(current)
            if current == 0:
                break

        load = sum(
            locations[customer_id]["demand"]
            for customer_id in route
            if customer_id != 0
        )
        routes.append(
            {
                "vehicle": vehicle + 1,
                "route": route,
                "load": load,
            }
        )
    return routes


def main():
    model, x, fulfilled, assigned, used, service_start, return_time = (
        build_model()
    )
    solver_status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[solver_status]

    print("C101 INTEGRATED MILP RESULTS")
    print("---------------------------")
    print("Customers in test:", len(customers))
    print("Solver status:", status)

    if status != "Optimal":
        return

    routes = extract_routes(x, used)
    fulfilled_customers = [
        customer_id
        for customer_id in N
        if pulp.value(fulfilled[customer_id]) > 0.5
    ]
    fulfilled_demand = sum(
        locations[customer_id]["demand"]
        for customer_id in fulfilled_customers
    )

    print("\nINVENTORY RESULTS")
    print("-----------------")
    print("Stock limit:", STOCK)
    print("Fulfilled customers:", len(fulfilled_customers))
    print("Fulfilled demand:", fulfilled_demand)
    print("Rejected customers:", len(N) - len(fulfilled_customers))

    print("\nROUTING RESULTS")
    print("----------------")
    print("Vehicles used:", len(routes))
    for route in routes:
        print(f"\nVehicle {route['vehicle']}")
        print("Route:", route["route"])
        print("Load:", route["load"])

    print("\nCUSTOMER SERVICE TIMES")
    print("----------------------")
    for customer_id in fulfilled_customers:
        vehicle = next(
            vehicle
            for vehicle in K
            if pulp.value(assigned[customer_id, vehicle]) > 0.5
        )
        print(
            f"Customer {customer_id}: "
            f"Vehicle {vehicle + 1}, "
            f"Service start {pulp.value(service_start[customer_id, vehicle]):.2f}, "
            f"Window "
            f"{locations[customer_id]['ready_time']}-"
            f"{locations[customer_id]['due_date']}"
        )

    print("\nWAREHOUSE RETURN TIMES")
    print("----------------------")
    for vehicle in K:
        if pulp.value(used[vehicle]) > 0.5:
            print(
                f"Vehicle {vehicle + 1} solver value: "
                f"{pulp.value(return_time[vehicle]):.2f}"
            )

    visited = [
        customer_id
        for route in routes
        for customer_id in route["route"]
        if customer_id != 0
    ]
    time_windows_valid = True
    warehouse_return_valid = True
    actual_return_times = {}
    for route_data in routes:
        route = route_data["route"]
        current_time = warehouse["ready_time"]
        for position in range(1, len(route) - 1):
            previous = route[position - 1]
            customer_id = route[position]
            arrival = (
                current_time
                + calculate_distance(
                    locations[previous],
                    locations[customer_id],
                )
            )
            actual_service_start = max(
                arrival,
                locations[customer_id]["ready_time"],
            )
            if actual_service_start > locations[customer_id]["due_date"]:
                time_windows_valid = False
            current_time = (
                actual_service_start
                + locations[customer_id]["service_time"]
            )

        if len(route) > 2:
            last_customer = route[-2]
            actual_return = (
                current_time
                + calculate_distance(
                    locations[last_customer],
                    locations[0],
                )
            )
            actual_return_times[route_data["vehicle"]] = actual_return
            if actual_return > warehouse["due_date"]:
                warehouse_return_valid = False

    print("\nINDEPENDENT RETURN-TIME VALIDATION")
    print("----------------------------------")
    for vehicle, actual_return in actual_return_times.items():
        print(f"Vehicle {vehicle} actual return: {actual_return:.2f}")

    print("\nMILP VALIDATION")
    print("----------------")
    print("Inventory valid:", fulfilled_demand <= STOCK)
    print(
        "All fulfilled customers routed:",
        set(visited) == set(fulfilled_customers),
    )
    print("No duplicate deliveries:", len(visited) == len(set(visited)))
    print(
        "Capacity valid:",
        all(route["load"] <= VEHICLE_CAPACITY for route in routes),
    )
    print(
        "Routes start and end at warehouse:",
        all(
            route["route"][0] == 0 and route["route"][-1] == 0
            for route in routes
        ),
    )
    print("Customer time windows valid:", time_windows_valid)
    print("Warehouse return time valid:", warehouse_return_valid)


if __name__ == "__main__":
    main()
