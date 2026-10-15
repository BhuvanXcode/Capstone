import itertools
import math

# Same synthetic coordinates used by quantum_tsp_qaoa.py.
# Node 0 is the depot; nodes 1-3 are customers.
coordinates = [
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 1.0],
    [0.0, 2.0],
]


def distance(a, b):
    return math.hypot(
        coordinates[a][0] - coordinates[b][0],
        coordinates[a][1] - coordinates[b][1],
    )


def route_distance(route):
    return sum(
        distance(route[i], route[i + 1])
        for i in range(len(route) - 1)
    )


routes = []
for ordering in itertools.permutations(range(1, len(coordinates))):
    route = (0, *ordering, 0)
    routes.append((route, route_distance(route)))

routes.sort(key=lambda item: item[1])
best_distance = routes[0][1]

print("====== EXACT CLASSICAL TSP RESULTS ======")
print("Routes evaluated:", len(routes))
for route, total_distance in routes:
    print(f"{list(route)} -> distance {total_distance:.4f}")

print("\nBest exact route:", list(routes[0][0]))
print(f"Optimal distance: {best_distance:.4f}")
