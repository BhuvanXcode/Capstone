import itertools
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector


def run_qaoa_experiment():
    """Run the synthetic three-customer QAOA TSP experiment."""


    # ==================================================
    # 1. DEFINE A SMALL ROUTING PROBLEM
    # ==================================================

    # Node 0 = depot; nodes 1, 2, 3 = customers.
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
        # occupies a particular route position.
        x = np.array([
            [
                (bitstring >> (city * n_customers + pos)) & 1
                for pos in range(n_customers)
            ]
            for city in range(n_customers)
        ])

        city_penalty = sum(
            (int(row.sum()) - 1) ** 2
            for row in x
        )
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
            x[city, n_customers - 1] * distances[city + 1, 0]
            for city in range(n_customers)
        )

        return float(cost + penalty * (city_penalty + position_penalty))


    # Precompute cost for all 2^9 computational basis states.
    energies = np.array([
        energy(state)
        for state in range(2 ** n_qubits)
    ])


    # ==================================================
    # 3. BUILD THE QAOA CIRCUIT
    # ==================================================

    def create_qaoa_circuit(gamma, beta):
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        phases = np.exp(-1j * gamma * energies)
        qc.append(DiagonalGate(phases), list(range(n_qubits)))

        for qubit in range(n_qubits):
            qc.rx(2 * beta, qubit)

        return qc


    # ==================================================
    # 4. OPTIMIZE THE PARAMETERS
    # ==================================================

    def expectation(parameters):
        gamma, beta = parameters
        circuit = create_qaoa_circuit(gamma, beta)
        probabilities = Statevector.from_instruction(circuit).probabilities()
        return float(np.dot(probabilities, energies))


    print("\nOptimizing QAOA parameters...")
    result = minimize(
        expectation,
        x0=[0.02, 0.3],
        method="COBYLA",
        options={"maxiter": 100, "rhobeg": 0.1},
    )

    gamma, beta = result.x
    print("Optimization finished.")
    print("Optimizer success:", result.success)
    print("Best expected energy:", round(result.fun, 4))
    print("Gamma:", round(gamma, 4))
    print("Beta:", round(beta, 4))


    # ==================================================
    # 5. SAMPLE AND COUNT FEASIBLE ROUTES
    # ==================================================

    circuit = create_qaoa_circuit(gamma, beta)
    probabilities = Statevector.from_instruction(circuit).probabilities()

    rng = np.random.default_rng(42)
    n_shots = 5000
    samples = rng.choice(2 ** n_qubits, size=n_shots, p=probabilities)

    feasible_route_counts = Counter()
    feasible_route_distances = {}

    for sample in samples:
        x = np.array([
            [
                (int(sample) >> (city * n_customers + pos)) & 1
                for pos in range(n_customers)
            ]
            for city in range(n_customers)
        ])

        # Require every customer exactly once and every position filled.
        if np.all(x.sum(axis=0) == 1) and np.all(x.sum(axis=1) == 1):
            route = [0]
            for pos in range(n_customers):
                route.append(int(np.argmax(x[:, pos])) + 1)
            route.append(0)
            route = tuple(route)

            route_distance = sum(
                distances[route[i], route[i + 1]]
                for i in range(len(route) - 1)
            )
            feasible_route_counts[route] += 1
            feasible_route_distances[route] = float(route_distance)


    # ==================================================
    # 6. EXACT CLASSICAL BENCHMARK
    # ==================================================

    exact_routes = []
    for ordering in itertools.permutations(range(1, n_customers + 1)):
        route = (0, *ordering, 0)
        route_distance = sum(
            distances[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        )
        exact_routes.append((route, float(route_distance)))

    optimal_distance = min(distance for _, distance in exact_routes)
    optimal_routes = {
        route for route, distance in exact_routes
        if np.isclose(distance, optimal_distance)
    }

    feasible_samples = sum(feasible_route_counts.values())
    optimal_samples = sum(
        count for route, count in feasible_route_counts.items()
        if route in optimal_routes
    )

    # Save machine-readable experiment metrics for later comparisons.
    results_data = {
        "experiment": "QAOA versus exact classical enumeration on a 3-customer TSP",
        "problem": {
            "type": "synthetic TSP",
            "depot_id": 0,
            "customer_ids": list(range(1, n_customers + 1)),
            "coordinates": coordinates.tolist(),
            "customers": n_customers,
            "qubits": n_qubits,
            "penalty": penalty,
        },
        "qaoa": {
            "simulator": "Qiskit Statevector",
            "optimizer": "COBYLA",
            "optimizer_success": bool(result.success),
            "max_iterations": 100,
            "gamma": float(gamma),
            "beta": float(beta),
            "best_expected_energy": float(result.fun),
            "shots": n_shots,
            "random_seed": 42,
            "feasible_samples": feasible_samples,
            "feasible_sample_rate_percent": 100 * feasible_samples / n_shots,
            "unique_feasible_routes": len(feasible_route_counts),
            "optimal_samples": optimal_samples,
            "optimal_sample_rate_percent_of_all_shots": 100 * optimal_samples / n_shots,
            "optimal_sample_rate_percent_of_feasible_samples": (
                100 * optimal_samples / feasible_samples if feasible_samples else 0.0
            ),
            "best_route": (
                list(min(feasible_route_distances, key=feasible_route_distances.get))
                if feasible_route_distances else None
            ),
            "best_route_distance": (
                min(feasible_route_distances.values())
                if feasible_route_distances else None
            ),
            "route_sample_counts": [
                {
                    "route": list(route),
                    "distance": feasible_route_distances[route],
                    "samples": count,
                }
                for route, count in feasible_route_counts.most_common()
            ],
        },
        "exact_classical": {
            "method": "exhaustive enumeration of all customer permutations",
            "routes_evaluated": len(exact_routes),
            "optimal_distance": float(optimal_distance),
            "optimal_routes": [list(route) for route in sorted(optimal_routes)],
            "all_routes": [
                {"route": list(route), "distance": distance}
                for route, distance in sorted(exact_routes, key=lambda item: item[1])
            ],
        },
        "interpretation": [
            "QAOA sampled a route matching the exact optimum on this small synthetic TSP.",
            "This is a classical statevector simulation and is not evidence of quantum speedup.",
            "Vehicle capacity and time-window constraints are not implemented in this prototype.",
        ],
    }


    # ==================================================
    # 7. DISPLAY RESULTS
    # ==================================================

    print("\n========== QAOA EVALUATION ==========")
    print("Total shots:", n_shots)
    print("Feasible samples:", feasible_samples)
    print("Unique feasible routes:", len(feasible_route_counts))
    print("Exact optimal distance:", round(optimal_distance, 4))
    print("Optimal samples:", optimal_samples)
    print(f"Optimal-sample rate: {100 * optimal_samples / n_shots:.2f}%")
    print(f"Feasible-sample rate: {100 * feasible_samples / n_shots:.2f}%")

    if feasible_route_distances:
        best_route = min(feasible_route_distances, key=feasible_route_distances.get)
        print("Best QAOA route:", list(best_route))
        print("Best QAOA distance:", round(feasible_route_distances[best_route], 4))
    else:
        print("No feasible routes were sampled.")

    print("\nExact optimal routes:")
    for route in sorted(optimal_routes):
        print(f"{list(route)} -> distance {optimal_distance:.4f}")

    print("\nSample counts for feasible routes:")
    for route, count in feasible_route_counts.most_common():
        print(
            f"{list(route)} -> distance {feasible_route_distances[route]:.4f}, "
            f"samples {count}"
        )

    print("\nNote: This is a small synthetic TSP prototype.")
    print("Vehicle capacity and time windows are not implemented yet.")


if __name__ == "__main__":
    run_qaoa_experiment()
