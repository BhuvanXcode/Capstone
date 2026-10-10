import math
import matplotlib.pyplot as plt


# Load Solomon VRPTW dataset
file_path = "data/c101.txt"

customers = []

with open(file_path, "r") as file:
    for line in file:
        parts = line.split()

        if len(parts) == 7 and parts[0].isdigit():

            customer = {
                "id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "demand": float(parts[3]),
                "ready_time": float(parts[4]),
                "due_time": float(parts[5]),
                "service_time": float(parts[6])
            }

            customers.append(customer)


print("Dataset loaded successfully!")
print("Number of customers:", len(customers))


# Separate depot and customers
depot = customers[0]
customer_points = customers[1:]


# Calculate distance between two locations
def calculate_distance(point1, point2):
    return math.sqrt(
        (point1["x"] - point2["x"]) ** 2 +
        (point1["y"] - point2["y"]) ** 2
    )


# Test distance between depot and Customer 1
distance = calculate_distance(depot, customer_points[0])

print("\nDistance from Depot to Customer 1:")
print(round(distance, 2))


# -----------------------------------
# Step 4: Create distance matrix
# -----------------------------------

distance_matrix = []

for i in customers:
    row = []

    for j in customers:
        distance = calculate_distance(i, j)
        row.append(round(distance, 2))

    distance_matrix.append(row)


# Display first 5 × 5 part of the matrix
print("\nDistance Matrix (first 5 locations):")

for row in distance_matrix[:5]:
    print(row[:5])

# -----------------------------------
# Step 5: Vehicle capacity check
# -----------------------------------

vehicle_capacity = 200


def check_capacity(route):
    total_demand = sum(customers[i]["demand"] for i in route)

    if total_demand <= vehicle_capacity:
        return True, total_demand
    else:
        return False, total_demand


# Example route
test_route = [1, 2, 3, 4]

valid, demand = check_capacity(test_route)

print("\nTest Route:", test_route)
print("Total Demand:", demand)
print("Capacity:", vehicle_capacity)
print("Capacity Valid:", valid)

# -----------------------------------
# Step 6: Time window check
# -----------------------------------

def check_time_window(customer_id, arrival_time):
    customer = customers[customer_id]

    ready_time = customer["ready_time"]
    due_time = customer["due_time"]

    if arrival_time < ready_time:
        return False, "Too early"

    elif arrival_time > due_time:
        return False, "Too late"

    else:
        return True, "Within time window"


# Test Customer 1
arrival_time = 930

valid, message = check_time_window(1, arrival_time)

print("\nTime Window Test")
print("Customer:", 1)
print("Arrival Time:", arrival_time)
print("Valid:", valid)
print("Message:", message)
   
# -----------------------------------
# Step 7: Simulate a small route
# -----------------------------------

def simulate_route(route):
    current_time = 0
    total_distance = 0
    total_demand = 0

    for i in range(len(route) - 1):

        current_customer = customers[route[i]]
        next_customer = customers[route[i + 1]]

        # Add customer demand
        if next_customer["id"] != 0:
            total_demand += next_customer["demand"]

            # Check vehicle capacity
            if total_demand > vehicle_capacity:
                return {
                    "route": route,
                    "distance": round(total_distance, 2),
                    "feasible": False,
                    "reason": (
                        f"Vehicle capacity exceeded "
                        f"({total_demand} > {vehicle_capacity})"
                    ),
                    "total_demand": total_demand
                }

        # Travel to next customer
        travel_time = calculate_distance(
            current_customer,
            next_customer
        )

        total_distance += travel_time
        current_time += travel_time

        # Check time window
        if next_customer["id"] != 0:

            ready_time = next_customer["ready_time"]
            due_time = next_customer["due_time"]

            # Wait if we arrive too early
            if current_time < ready_time:
                current_time = ready_time

            # Check if too late
            if current_time > due_time:
                return {
                    "route": route,
                    "distance": round(total_distance, 2),
                    "feasible": False,
                    "reason": (
                        f"Customer {next_customer['id']} "
                        f"time window violated"
                    ),
                    "total_demand": total_demand
                }

            # Service customer
            current_time += next_customer["service_time"]

    # Route is feasible
    return {
        "route": route,
        "distance": round(total_distance, 2),
        "feasible": True,
        "reason": "Route is feasible",
        "total_demand": total_demand
    }
import matplotlib.pyplot as plt


def visualize_route(route):

    plt.figure(figsize=(10, 7))

    # Plot all customers
    for customer in customers:

        if customer["id"] == 0:
            # Depot
            plt.scatter(
                customer["x"],
                customer["y"],
                marker="*",
                s=250,
                label="Depot"
            )

        else:
            # Customer
            plt.scatter(
                customer["x"],
                customer["y"],
                s=60
            )

            # Customer ID
            plt.text(
                customer["x"] + 0.5,
                customer["y"] + 0.5,
                str(customer["id"]),
                fontsize=8
            )

    # Draw route
    for i in range(len(route) - 1):

        current = customers[route[i]]
        next_customer = customers[route[i + 1]]

        plt.plot(
            [current["x"], next_customer["x"]],
            [current["y"], next_customer["y"]],
            linewidth=2
        )

    # Route information
    result = simulate_route(route)

    if result["feasible"]:
        status = "FEASIBLE"
    else:
        status = "INFEASIBLE"

    plt.title(
        f"VRPTW Route Visualization\n"
        f"Route: {route} | "
        f"Distance: {result['distance']} | "
        f"Status: {status}"
    )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.grid(True)
    plt.legend()

    plt.show()




route_input = input("\nEnter route (example: 0,3,7,12,0): ")

test_route = [
    int(x.strip())
    for x in route_input.split(",")
]

result = simulate_route(test_route)

print("\nRoute Result")
print("------------")
print("Route:", result["route"])
print("Distance:", result["distance"])
print("Feasible:", result["feasible"])
print("Reason:", result["reason"])
print("Total Demand:", result["total_demand"])

visualize_route(test_route)