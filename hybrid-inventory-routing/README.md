# Hybrid Quantum-Classical Optimization for Integrated Warehouse Inventory Planning and Last-Mile Vehicle Routing

## 1. Project Overview

This project explores a hybrid quantum-classical approach for integrated warehouse inventory planning and last-mile vehicle routing.

The system combines:

- Classical optimization for vehicle routing.
- QUBO (Quadratic Unconstrained Binary Optimization) for customer selection under inventory constraints.
- QAOA (Quantum Approximate Optimization Algorithm) using a classical statevector simulator.
- OR-Tools for route optimization.
- MILP for small-instance mathematical optimization and validation.

The objective is to select customers that can be fulfilled within available warehouse inventory while considering delivery-related costs, and then optimize the routes for the selected customers.

**Note:** The current implementation uses a two-stage hybrid approach: QAOA selects customers, and OR-Tools generates delivery routes. It is not a fully joint quantum optimization of inventory and routing.

## 2. Problem Statement

A warehouse has limited inventory and must fulfil customer orders. Each customer has a demand quantity and delivery location.

The system must:

1. Select customers whose complete orders can be fulfilled.
2. Ensure total selected demand does not exceed warehouse stock.
3. Consider transportation costs when selecting customers.
4. Assign selected customers to delivery vehicles.
5. Minimize delivery distance while respecting vehicle capacity and applicable constraints.

## 3. Project Objectives

- Develop a classical inventory allocation and routing baseline.
- Formulate the inventory selection problem as a QUBO model.
- Implement QAOA using statevector simulation.
- Incorporate routing-aware cost information into customer selection.
- Integrate QAOA selection with classical OR-Tools routing.
- Compare classical and QAOA solutions.
- Evaluate feasibility, solution quality and sampling performance as the problem size increases.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical calculations |
| SciPy | Classical parameter optimization |
| Matplotlib | Experimental graphs |
| PuLP | Mixed Integer Linear Programming |
| OR-Tools | Vehicle routing optimization |
| QUBO | Binary optimization formulation |
| QAOA | Quantum-inspired optimization algorithm implemented through simulation |

## 5. System Architecture

```text
Customer and Warehouse Data
            |
            v
     Inventory Planning
            |
            v
       QUBO Model
            |
            v
       QAOA Simulator
            |
            v
   Customer Selection
            |
            v
     OR-Tools Routing
            |
            v
     Optimized Routes
            |
            v
    Results and Evaluation
```

Classical optimization is also used for baseline comparisons and small-instance validation.

## 6. Dataset and Assumptions

The routing experiments use the Solomon C101 benchmark dataset.

Dataset characteristics:

- One depot.
- 100 customers.
- 25 available vehicles.
- Vehicle capacity: 200.
- Customer demand and delivery time-window information.

Project assumptions:

- One warehouse.
- One product type.
- Full-order fulfilment only.
- One order per customer.
- Known customer demands and coordinates.
- Deterministic Euclidean travel distances.
- Initial warehouse inventory is a configurable project parameter.

The 8-customer and 10-customer QUBO experiments use small experimental instances with illustrative coordinates and configured inventory and vehicle capacity. Customers 9 and 10 are synthetic additions to those experiments, not additional C101 records.

## 7. Optimization Approaches

### Classical optimization

The project implements:

- Nearest Neighbour.
- OR-Tools Vehicle Routing Problem with Time Windows.
- Genetic Algorithm.
- Ant Colony Optimization.
- Greedy inventory allocation.
- Mixed Integer Linear Programming.

### QUBO formulation

Binary decision variables represent whether a customer is selected.

The inventory constraint is incorporated using binary slack variables and a penalty term.

The routing-aware formulation includes:

- Customer selection rewards.
- Independent warehouse round-trip costs.
- Pairwise routing-savings estimates.
- Inventory constraint penalties.

Pairwise savings are an approximation. They may overlap and do not represent the exact cost of a complete multi-stop route.

### QAOA simulation

QAOA is implemented using a statevector simulator.

A classical optimizer searches for QAOA parameters, and the simulator calculates measurement probabilities.

No physical quantum computer or quantum hardware is used.

## 8. Experimental Results

### Classical routing benchmark

| Algorithm | Vehicles | Distance |
|---|---:|---:|
| Nearest Neighbour | 21 | 1870.69 |
| OR-Tools | 10 | **854.38** |
| Genetic Algorithm | 17 | 2860.79 |
| Ant Colony Optimization | 17 | 2510.91 |

OR-Tools produced the lowest distance among these reported experiments.

### Inventory allocation

With initial warehouse stock of 1,000 units:

- Total customers: 100.
- Fulfilled customers: 76.
- Rejected customers: 24.
- Fulfilled demand: 1,000.
- Remaining inventory: 0.

### Integrated classical baseline

The sequential greedy-inventory and routing baseline produced:

- Fulfilled customers: 76.
- Vehicles used: 19.
- Delivery distance: 1594.41.
- Inventory and routing validation: passed.

### QAOA scaling experiment

| Metric | 8 customers | 10 customers |
|---|---:|---:|
| Qubits | 15 | 17 |
| State space | 32,768 | 131,072 |
| Feasible states | 238 | 773 |
| QAOA found classical optimum | No | Yes |

For the 10-customer experiment:

- Classical optimum selection: `[3, 4, 5, 6, 8, 9, 10]`.
- Total demand: 100.
- Slack: 0.
- Classical optimum QUBO energy: -774.378753.
- QAOA best feasible energy: -774.378753.

### Repeated QAOA evaluation

Using 10 runs and 10,000 shots per run:

| Metric | Result |
|---|---:|
| Total measurements | 100,000 |
| Feasible measurements | 984 |
| Optimal measurements | 5 |
| Observed feasible probability | 0.984% |
| Observed optimal probability | 0.005% |
| Runs finding classical optimum | 4/10 |

The QAOA simulator recovered the classical optimum in four of the ten repeated sampling runs.

These measurements use the same optimized parameters and different sampling seeds; they are not ten independent parameter optimizations.

## 9. Generated Results

The `results/` directory contains:

- `classical_vs_qaoa.png`
- `feasible_measurements.png`
- `optimal_measurements.png`
- `qaoa_probabilities.png`
- `qaoa_run_results.csv`
- `final_comparison.csv`

These files provide visual and numerical summaries of the final 10-customer experiment.

## 10. Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd hybrid-inventory-routing
```

Install the required packages:

```bash
python -m pip install numpy scipy matplotlib pulp ortools
```

Run an experiment from the project root:

```bash
python evaluation/scaling_classical_10_customers.py
```

Run the 10-customer QAOA experiment:

```bash
python quantum/qaoa_scaling_10_customers.py
```

Run repeated QAOA evaluation:

```bash
python evaluation/qaoa_repeated_10_customers.py
```

Generate final result graphs and CSV files:

```bash
python evaluation/generate_final_results.py
```

Run the final comparison:

```bash
python evaluation/final_classical_qaoa_comparison.py
```

## 11. Limitations

- QAOA experiments use classical statevector simulation, not quantum hardware.
- The QAOA state space grows exponentially with the number of qubits.
- The 8-customer and 10-customer QUBO experiments are small experimental instances.
- The 10-customer instance includes synthetic customer coordinates.
- Pairwise routing savings are approximate and can double-count savings.
- QAOA customer selection and OR-Tools routing are sequential rather than jointly optimized.
- The full MILP implementation was evaluated on a small instance, not all 100 C101 customers.
- Results do not demonstrate quantum advantage over classical algorithms.

## 12. Future Enhancements

- Develop a more accurate integrated inventory-routing QUBO.
- Improve constraint handling and QAOA mixer design.
- Evaluate deeper QAOA circuits.
- Test larger and more realistic customer datasets.
- Incorporate multiple warehouses.
- Support multiple products and replenishment decisions.
- Compare execution time and solution quality under controlled conditions.
- Evaluate on quantum hardware when practical access becomes available.

## 13. Conclusion

This project demonstrates a prototype hybrid quantum-classical framework for inventory-aware last-mile delivery planning.

Classical optimization provides routing and mathematical-programming baselines, while QUBO and QAOA simulation explore inventory-constrained customer selection with approximate routing-aware costs.

The 10-customer experiment demonstrates that the QAOA simulator can recover the classical optimum, although its sampling probability is low. Further work is required to improve scalability and develop a more tightly integrated inventory-routing formulation.

---

**Project:** Hybrid Quantum-Classical Optimization for Integrated Warehouse Inventory Planning and Last-Mile Vehicle Routing  
**Implementation:** Python, QUBO, QAOA simulation, MILP and OR-Tools
