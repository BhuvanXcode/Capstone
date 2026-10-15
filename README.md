# Hybrid Quantum-Classical Optimization for Vehicle Routing

A capstone project exploring classical and quantum approaches to
small-scale vehicle-routing problems. The repository includes a
classical Vehicle Routing Problem with Time Windows (VRPTW)
implementation using Google OR-Tools, a small QAOA-based Traveling
Salesperson Problem (TSP) prototype using Qiskit, and scripts for result
comparison and visualization.

> **Scope note:** The current quantum prototype solves a small synthetic
> TSP. It does not yet combine warehouse inventory planning,
> vehicle-capacity constraints, and delivery time windows into one
> integrated quantum model. The QAOA experiment uses Qiskit's classical
> statevector simulator, so it does not demonstrate quantum hardware
> advantage or quantum speedup.

## Project Goals

-   Model last-mile routing as an optimization problem.
-   Generate a classical baseline using Google OR-Tools.
-   Explore QAOA for a small TSP instance using Qiskit.
-   Compare candidate routes and distances with classical results.
-   Visualize routes and save experiment outputs for analysis.

## Repository Structure

``` text
Capstone/
├── data/                    # Input benchmark dataset(s), including c101.txt
├── outputs/                 # Generated results and route data
├── screenshots/             # Project screenshots
├── src/                     # Additional source code
├── classical_routing.py     # OR-Tools VRPTW solver
├── exact_classical_tsp.py   # Exact classical TSP benchmark
├── quantum_tsp_qaoa.py      # Small QAOA TSP experiment
├── compare_results.py       # Compare experiment results
├── visualization.py         # Visualize routing results
├── main.py                  # Dataset parsing, constraint checks and plotting
└── .gitignore
```

## Technologies

-   **Python** --- core implementation
-   **Google OR-Tools** --- classical vehicle-routing optimization
-   **Qiskit** --- quantum circuit construction and statevector
    simulation
-   **NumPy** --- numerical calculations
-   **SciPy** --- classical parameter optimization (COBYLA)
-   **Matplotlib** --- route and result visualization

## Problem Types

### Classical VRPTW

`classical_routing.py` loads a Solomon-style C101 dataset from
`data/c101.txt`, builds a distance matrix, and uses OR-Tools to search
for routes subject to vehicle capacity and customer time-window
constraints. Route data is configured to be written to
`outputs/ortools_routes.json`.

### QAOA TSP Prototype

`quantum_tsp_qaoa.py` defines a small synthetic problem with one depot
and three customers. It encodes route-ordering constraints with penalty
terms, builds a QAOA-style circuit, tunes parameters with COBYLA,
samples the Qiskit statevector distribution, and compares feasible
routes against exhaustive classical enumeration.

The current quantum prototype does **not** implement vehicle capacity,
customer time windows, or warehouse inventory constraints.

## Getting Started

### Prerequisites

-   Python 3.10 or newer recommended
-   Git
-   A virtual environment recommended

### 1. Clone the repository

``` bash
git clone https://github.com/BhuvanXcode/Capstone.git
cd Capstone
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install numpy scipy matplotlib ortools qiskit
```

If installation fails or a Qiskit API differs in your environment, check
the imports in the relevant script and install compatible package
versions.

### 4. Check the dataset

Confirm that `data/c101.txt` exists before running the classical routing
script. The script expects a Solomon-format file with vehicle
information and customer records.

## Running the Project

Run commands from the repository root.

### Classical VRPTW solver

``` bash
python classical_routing.py
```

### QAOA TSP experiment

``` bash
python quantum_tsp_qaoa.py
```

This runs the small synthetic TSP experiment and prints QAOA sampling
statistics and the exact classical benchmark. It uses Qiskit's
`Statevector` simulator and does not require a physical quantum
computer.

### Exact classical TSP benchmark

``` bash
python exact_classical_tsp.py
```

### Compare results

``` bash
python compare_results.py
```

Run the comparison after generating the files it expects. If a file is
missing, inspect the paths in `compare_results.py` and run the
corresponding experiment first.

### Route visualization and checks

``` bash
python main.py
```

This script loads the dataset, performs distance and constraint checks,
asks for a route in comma-separated node IDs (for example,
`0,3,7,12,0`), and plots the route. Use node IDs that exist in the
loaded dataset.

## Workflow

``` text
Solomon C101 dataset
        |
        v
Data loading and distance matrix
        |
        +----------------------------+
        |                            |
        v                            v
Classical VRPTW                  Small synthetic TSP
Google OR-Tools                  QUBO-style cost + QAOA
        |                            |
        v                            v
Classical route output          Qiskit Statevector samples
        |                            |
        +-------------+--------------+
                      v
             Compare route quality
             and visualize results
```

## Evaluation Metrics

Depending on the script and output file, the project can examine:

-   Route distance or objective cost
-   Route feasibility
-   Vehicle capacity and time-window compliance for the classical VRPTW
    model
-   QAOA feasible-sample rate
-   Frequency of optimal routes in the small TSP experiment
-   Runtime and other saved metrics, where available

The classical VRPTW dataset and the small synthetic QAOA TSP are
different problem setups. Do not compare their metrics as if they were
the same instance.

## Current Limitations

-   The quantum experiment is a small synthetic TSP prototype.
-   The quantum experiment uses a classical statevector simulator, not
    quantum hardware.
-   Vehicle capacity and delivery time windows are not yet encoded in
    the QAOA prototype.
-   Integrated warehouse inventory planning is not yet implemented in
    the current quantum experiment.
-   Small-instance results do not establish quantum speedup or
    superiority over classical solvers.
-   Results depend on the dataset, solver settings, penalty values, and
    environment.

## Future Work

-   Benchmark classical and quantum approaches on the same problem
    instance.
-   Evaluate how QUBO penalty weights affect feasibility and solution
    quality.
-   Integrate inventory availability and demand fulfilment into the
    optimization model.
-   Validate solutions against the same constraints.
-   Report solution quality, feasibility, and runtime across multiple
    problem sizes.
-   Explore noisy simulation or quantum hardware when appropriate and
    available.

## Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Make a focused change and test the relevant script.
4.  Open a pull request describing the change and any experimental
    results.


## Acknowledgements

-   Google OR-Tools for classical optimization tools.
-   Qiskit for quantum circuit and simulation tooling.
-   Solomon benchmark instances for vehicle-routing research, where
    applicable.
