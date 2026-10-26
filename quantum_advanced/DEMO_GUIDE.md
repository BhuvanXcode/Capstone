# Mentor Demonstration Guide: Quantum VRPTW Capstone

> **Capstone Project:** Vehicle Routing Problem with Time Windows (VRPTW)  
> **Directory:** `D:\vrptw\quantum_advanced`  
> **Purpose:** Step-by-step presentation guide for your mentor meeting.  
> **Approach:** Practical, beginner-friendly, and scientifically honest.

---

## 1. Quick Project Overview (Beginner-Friendly Concepts)

### What is the Objective?
The goal of this capstone is to solve the **Vehicle Routing Problem with Time Windows (VRPTW)** using both **classical optimization** and **quantum computing (QAOA)**, and compare their performance honestly on standard benchmark data.

### The Dataset: Solomon C101 (100 Customers)
- We use the standard benchmark dataset from academic operations research: **Solomon C101 (`c101.txt`)**.
- It consists of **1 Central Depot (Node 0)** and **100 Customers (Nodes 1 to 100)** located on a 2D grid.
- Each customer has:
  1. **Demand:** Quantities of goods to deliver (Total demand = 1,810 units).
  2. **Ready Time & Due Time:** A specific time window during which delivery must begin. In C101, windows are narrow (~55 time units wide).
  3. **Service Duration:** 90 time units spent serving each customer.
- Each vehicle has a **Capacity Limit of 200.0 units**. Minimum theoretical fleet size = $\lceil 1810 / 200 \rceil = 10$ vehicles.

### What is QUBO?
- **QUBO** stands for **Quadratic Unconstrained Binary Optimization**.
- To give an optimization problem to a quantum computer, we represent routing decisions as binary variables:
  $$x_{i, p} \in \{0, 1\}$$
  $x_{i, p} = 1$ if customer $i$ is visited at position $p$ on the route; $0$ otherwise.
- Permutation rules ("every customer visited once", "every position filled once") are added to the objective as quadratic penalty multipliers.

### What is QAOA?
- **QAOA** stands for **Quantum Approximate Optimization Algorithm**.
- It is a hybrid quantum-classical algorithm designed for near-term (NISQ) quantum hardware:
  1. Starts in an equal superposition of all states using **Hadamard gates ($H$)**.
  2. Applies a **Problem Cost Unitary** using single-qubit rotations ($R_Z$) and two-qubit entangling gates ($R_{ZZ}$) parameterized by angle $\gamma$.
  3. Applies a **Mixer Unitary** using single-qubit rotations ($R_X$) parameterized by angle $\beta$.
  4. A classical optimizer (COBYLA) tunes $\gamma$ and $\beta$ to minimize the expected tour distance.
  5. Measures the quantum circuit to extract candidate routes.

### Why Does the Quantum Circuit Use a Small Subproblem Rather than All 100 Customers?
- **The $N^2$ Qubit Barrier:** In order-based QUBO routing, $N$ customers requires $N^2$ qubits.
  - 3 customers $= 3^2 = 9$ qubits $\implies$ Statevector RAM: **8 KB** (Runs in 2 seconds).
  - 4 customers $= 4^2 = 16$ qubits $\implies$ Statevector RAM: **1 MB** (Runs in 6 seconds).
  - 6 customers $= 6^2 = 36$ qubits $\implies$ Statevector RAM: **1,024 GB (1.0 TB)** (Exhausts laptop memory).
  - 100 customers $= 100^2 = 10,000$ qubits $\implies$ Statevector RAM: **$> 10^{2998}\text{ TB}$** (Physically impossible to simulate on classical computers, and no 10,000-qubit entangling quantum computer exists today).
- **The Solution:** We implement a **practical hybrid decomposition**:
  - Classical algorithms partition the 100 customers into vehicle routes.
  - QAOA is deployed on small, genuine subproblems of 3 to 4 customers (9 to 16 qubits).
  - The quantum-optimized pieces are synthesized back into complete routes.

---

## 2. Executive Results Summary (Honest Scientific Verdict)

| Metric | Classical Baseline (OR-Tools) | Practical Hybrid QAOA Solver | Mentor Status / Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Dataset Size** | **100 Customers + 1 Depot** | **100 Customers + 1 Depot** | Full Solomon C101 dataset (Never truncated) |
| **Customer Coverage** | **100 / 100 (100%)** | **100 / 100 (100%)** | **Both cover all 100 customers with zero omissions** |
| **Fleet Size** | **10 Vehicles** | **11 Vehicles** | Meets theoretical lower bound ($\lceil 1810 / 200 \rceil = 10$) |
| **Total Route Distance** | **828.94** | **1353.90** (+63.33% gap) | Classical metaheuristic achieves the benchmark optimum |
| **Total Delivered Demand** | **1,810 / 1,810** | **1,810 / 1,810** | 100% of physical customer demands satisfied |
| **Vehicle Capacity Check** | **PASSED (0 overloads)** | **PASSED (0 overloads)** | All routes respect vehicle capacity $\le 200.0$ |
| **Customer Time Windows** | **PASSED (0 violations)** | **VIOLATED (89 violations)** | **Transparently reported (see explanation below)** |
| **QAOA Subproblems** | N/A (Monolithic Classical) | **31 Genuine QAOA Runs** | 3-4 customers each (9 or 16 qubits) |
| **Quantum Runtime** | N/A | **30.58 seconds** (`AerSimulator`) | Real gate-level simulation on Qiskit Aer |
| **Total Solve Runtime** | **5.05 seconds** | **35.97 seconds** | Classical solver is faster on classical hardware |
| **Overall Feasibility** | **100% FEASIBLE** | **REPORTED INFEASIBLE** | **No fabricated results; no false supremacy claims** |

### Why Does the Hybrid Solution Have 89 Time Window Violations?
- Solomon C101 customer time windows are very narrow (~55 time units), while customer service duration is 90 units.
- Classical clustering groups customers geographically. The intra-cluster QAOA quantum circuits optimize pure Euclidean routing distance.
- Because time windows are not encoded in the 2-local Ising Hamiltonian (which would require time-expanded discretization with thousands of ancilla qubits), QAOA reorders stops based on distance alone.
- A delay at one stop cascades to subsequent stops down a 10-customer route, resulting in 89 late deliveries.
- **Academic Lesson for Defense:** Pure distance-minimizing quantum subroutines must be coupled with classical temporal repair operators in operational logistics.

---

## 3. Recommended Live Demo Order (3 Simple Steps)

Open a terminal in `D:\vrptw\quantum_advanced` and run these three commands in order:

### Step 1: Show the Hardware Resource Estimator (0.1 Seconds)
```bash
python run_quantum_vrptw.py --estimate
```
- **What this demonstrates:** Pre-flight math calculating statevector RAM and qubit scaling.
- **What to tell your mentor:**
  > *"Before running any quantum code, our system calculates memory requirements. For 3 customers, 9 qubits require 8 KB RAM. But 100 customers would require 10,000 qubits and $>10^{2998}\text{ TB}$ of RAM, which is physically impossible. This mathematically proves why full-dataset routing must use a hybrid decomposition approach."*

---

### Step 2: Show the Small-Scale QAOA Baseline (2.2 Seconds)
```bash
python run_quantum_vrptw.py
```
- **What this demonstrates:** A genuine gate-level QAOA circuit ($H, R_Z, R_{ZZ}, R_X$) running on Qiskit `AerSimulator` for 3 customers (9 qubits).
- **What to tell your mentor:**
  > *"Here is our core quantum algorithm. It runs on Qiskit AerSimulator, optimizes variational parameters $(\gamma, \beta)$ with COBYLA, and measures 5,000 shots. The best measured bitstring matches the exact classical optimum with 0.00% gap and 0 time window violations."*

---

### Step 3: Run the Full 100-Customer Workflow (~35 Seconds)
```bash
python run_full_dataset.py --shots 500 --maxiter 10
```
- **What this demonstrates:**
  1. Loads all 100 customers from `c101.txt` (never truncated).
  2. Runs Google OR-Tools on the full 100 customers (10 vehicles, distance 828.94, 0 violations).
  3. Runs practical hybrid QAOA: decomposes into 31 small subproblems of 3–4 customers (9–16 qubits), executes them on `AerSimulator`, and reassembles them into 11 vehicle routes.
  4. Audits customer coverage (100/100 served), vehicle capacity, and time windows.
- **What to tell your mentor:**
  > *"Now we scale to the complete 100-customer Solomon C101 dataset. Classical OR-Tools solves it in 5 seconds with 10 vehicles and zero violations. Our hybrid solver partitions the 100 customers into 31 quantum subproblems of 9 to 16 qubits. It successfully visits all 100 customers with zero capacity overloads. However, it incurs 89 time window violations because unconstrained QAOA minimizes distance rather than time windows. We report this honestly without fabricating results."*

---

### Step 4 (Optional): Run the Automated Test Suite (6 Seconds)
```bash
python -m unittest test_full_dataset
```
- **What this demonstrates:** 7 automated unit tests verifying 100-customer ingestion, coverage checks, capacity math, and time-window auditing. All 7 tests pass.

---

## 4. Visual Files to Open During Presentation

All presentation figures are organized in [`mentor_demo_gallery/`](file:///d:/vrptw/quantum_advanced/mentor_demo_gallery):

```
mentor_demo_gallery/
├── 01_full_100_customer_classical_routes.png       <-- Full 100-customer OR-Tools baseline (10 vehicles)
├── 02_qaoa_quantum_circuit_architecture.png        <-- Gate breakdown (Rz, Rzz entangling, Rx mixer)
├── 03_qaoa_circuit_ascii_diagram.txt              <-- Full Qiskit quantum circuit diagram in text
├── 04_small_instance_qaoa_convergence.png         <-- Classical optimizer (COBYLA) energy descent
├── 05_small_instance_bitstring_sampling.png        <-- Quantum measurement shot distribution
├── 06_small_instance_route_verification.png       <-- 3-customer spatial map and metrics card
├── 07_quantum_hardware_scaling_limits.png          <-- RAM explosion and feasible subspace collapse
├── 08_hybrid_qaoa_100_customer_routes.png         <-- Hybrid QAOA 100-customer fleet route map
├── 09_classical_vs_hybrid_comparison_dashboard.png <-- Side-by-side comparison cards and charts
└── 10_qaoa_subproblem_telemetry.png               <-- Shot rates & depths for all 31 QAOA subproblems
```

---

## 5. Short Mentor Presentation Script

Use this script during your meeting:

### Slide 1: Problem Definition & Context
> *"Good morning/afternoon. For my capstone project, I investigated solving the Vehicle Routing Problem with Time Windows (VRPTW) using quantum computing and classical algorithms. We used the standard Solomon C101 benchmark, which has 100 customers with tight delivery time windows and a vehicle capacity of 200 units."*

### Slide 2: The Classical Benchmark (Open `01_full_100_customer_classical_routes.png`)
> *"First, we established a classical ground truth using Google OR-Tools. As you can see in this figure, OR-Tools uses Guided Local Search to solve the full 100-customer fleet in 5 seconds. It uses 10 vehicles, covers a total distance of 828.94, and satisfies 100% of customer time windows with zero violations."*

### Slide 3: Why Quantum Cannot Run All 100 Customers Directly (Open `07_quantum_hardware_scaling_limits.png`)
> *"Next, we investigated applying QAOA. When mapping routing to a QUBO, order-based encoding requires $N^2$ qubits. For 3 customers, that is 9 qubits (8 KB RAM). But for 100 customers, it would require 10,000 qubits. Simulating 10,000 qubits would take more RAM than there are atoms in the universe. Even a 6-customer instance requires 1 TB of RAM. So monolithic 100-customer quantum simulation is impossible on today's hardware."*

### Slide 4: The Quantum Circuit & Small Instance Verification (Open `02_qaoa_quantum_circuit_architecture.png` and `06_small_instance_route_verification.png`)
> *"To prove that our quantum algorithm works, we built a genuine gate-level QAOA circuit in Qiskit using Hadamard gates, single-qubit Rz rotations, two-qubit Rzz entangling couplers, and Rx mixers. On a 3-customer instance (9 qubits), QAOA optimized with COBYLA found the exact classical optimum with a 0.00% gap and zero violations."*

### Slide 5: The Full-Dataset Hybrid Workflow (Open `08_hybrid_qaoa_100_customer_routes.png` and `09_classical_vs_hybrid_comparison_dashboard.png`)
> *"To address the full 100-customer dataset, we developed a practical hybrid quantum-classical decomposition. We partitioned the 100 customers into vehicle clusters, and then divided each cluster into 31 small subproblems of 3 to 4 customers (9 to 16 qubits each). We ran real QAOA circuits on Qiskit AerSimulator for all 31 subproblems and synthesized the routes.*
> 
> *As shown in our comparison dashboard: all 100 customers are served (100% coverage), and all 11 vehicles respect the 200-unit capacity limit. However, the hybrid solution incurs 89 time window violations. This happens because unconstrained QAOA minimizes geometric distance without time-window penalties, causing cascading delays along 10-stop routes.*
> 
> *We report this result honestly: quantum subroutines can optimize route geometries, but classical coordination is essential for temporal feasibility. We do not claim quantum supremacy."*

---

## 6. Verification Checklist

- [x] All 100 customers from Solomon C101 are loaded and processed.
- [x] Classical OR-Tools baseline runs and is 100% feasible (10 vehicles, 828.94 distance).
- [x] Genuine QAOA quantum circuits executed on Qiskit `AerSimulator`.
- [x] 31 QAOA subproblems executed with complete telemetry (qubits, depths, gates).
- [x] All 100 customers covered with zero omissions and zero duplicates.
- [x] Automated test suite passes 7 out of 7 unit tests (`test_full_dataset.py`).
- [x] High-resolution plots organized into `mentor_demo_gallery/`.
- [x] Zero fabricated data; honest presentation of limitations.
