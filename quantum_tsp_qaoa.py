
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

# ==================================================
# 1. DEFINE A SMALL ROUTING PROBLEM
# ==================================================

# Node 0 = depot
# Nodes 1, 2, 3 = customers
coordinates = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 1.0],
    [0.0, 2.0],
])

n_customers = 3
n_qubits = n_customers * n_customers
penalty = 20.0

distances = np.linalg.norm(
    coordinates[:, None, :] - coordinates[None, :, :],
    axis=2,
)

print("Distance matrix:")
print(np.round(distances, 2))
print(f"\nQubits: {n_qubits}")


# ==================================================
# 2. COST FUNCTION
# ==================================================

def energy(bitstring):
    # x[city, position] indicates whether a customer
    # occupies a particular position in the route.
    x = np.array([
        [
            (bitstring >> (city * n_customers + pos)) & 1
            for pos in range(n_customers)
        ]
        for city in range(n_customers)
    ])

    # Every customer must appear exactly once.
    city_penalty = sum(
        (int(row.sum()) - 1) ** 2
        for row in x
    )

    # Every route position must contain exactly one customer.
    position_penalty = sum(
        (int(x[:, pos].sum()) - 1) ** 2
        for pos in range(n_customers)
    )

    # Depot -> first customer.
    cost = sum(
        x[city, 0] * distances[0, city + 1]
        for city in range(n_customers)
    )

    # Consecutive customers.
    for pos in range(n_customers - 1):
        for i in range(n_customers):
            for j in range(n_customers):
                cost += (
                    x[i, pos]
                    * x[j, pos + 1]
                    * distances[i + 1, j + 1]
                )

    # Last customer -> depot.
    cost += sum(
        x[city, n_customers - 1]
        * distances[city + 1, 0]
        for city in range(n_customers)
    )

    return float(
        cost + penalty * (city_penalty + position_penalty)
    )


# Precompute the cost of every possible bitstring.
energies = np.array([
    energy(state)
    for state in range(2 ** n_qubits)
])


# ==================================================
# 3. BUILD THE QAOA CIRCUIT
# ==================================================

def create_qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(n_qubits)

    # Initial uniform superposition.
    qc.h(range(n_qubits))

    # Cost-dependent phase separator.
    phases = np.exp(-1j * gamma * energies)
    qc.append(
        DiagonalGate(phases),
        list(range(n_qubits)),
    )

    # X-mixer.
    for qubit in range(n_qubits):
        qc.rx(2 * beta, qubit)

    return qc


# ==================================================
# 4. OPTIMIZE THE PARAMETERS
# ==================================================

def expectation(parameters):
    gamma, beta = parameters

    circuit = create_qaoa_circuit(gamma, beta)
    probabilities = Statevector.from_instruction(
        circuit
    ).probabilities()

    return float(np.dot(probabilities, energies))


print("\nOptimizing QAOA parameters...")

result = minimize(
    expectation,
    x0=[0.02, 0.3],
    method="COBYLA",
    options={
        "maxiter": 100,
        "rhobeg": 0.1,
    },
)

gamma, beta = result.x

print("Optimization finished.")
print("Optimizer success:", result.success)
print("Best expected energy:", round(result.fun, 4))
print("Gamma:", round(gamma, 4))
print("Beta:", round(beta, 4))


# ==================================================
# 5. SAMPLE AND CHECK CANDIDATE ROUTES
# ==================================================

circuit = create_qaoa_circuit(gamma, beta)
probabilities = Statevector.from_instruction(
    circuit
).probabilities()

rng = np.random.default_rng(42)

n_shots = 5000
samples = rng.choice(
    2 ** n_qubits,
    size=n_shots,
    p=probabilities,
)

feasible_routes = {}

for sample in samples:
    x = np.array([
        [
            (int(sample) >> (city * n_customers + pos)) & 1
            for pos in range(n_customers)
        ]
        for city in range(n_customers)
    ])

    # Reject assignments violating either constraint.
    if (
        np.all(x.sum(axis=0) == 1)
        and np.all(x.sum(axis=1) == 1)
    ):
        route = [0]

        for pos in range(n_customers):
            city = int(np.argmax(x[:, pos]))
            route.append(city + 1)

        route.append(0)
        route = tuple(route)

        route_distance = sum(
            distances[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        )

        feasible_routes[route] = route_distance


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

print("\n========== QAOA ROUTING RESULTS ==========")
print("Shots:", n_shots)
print("Unique feasible routes sampled:", len(feasible_routes))

if feasible_routes:
    best_route = min(
        feasible_routes,
        key=feasible_routes.get,
    )

    print("Best sampled route:", list(best_route))
    print(
        "Total distance:",
        round(feasible_routes[best_route], 4),
    )

    print("\nAll feasible routes found:")
    for route, distance in sorted(
        feasible_routes.items(),
        key=lambda item: item[1],
    ):
        print(f"{list(route)} -> distance {distance:.4f}")
else:
    print("No feasible routes were sampled.")
    print("Try changing the initial parameters or iteration count.")

print("\nNote: This is a small TSP prototype.")
print("Vehicle capacity and time windows are not implemented yet.")
