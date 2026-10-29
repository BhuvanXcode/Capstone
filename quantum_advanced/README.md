# Focused QAOA Quantum Computing Demonstration for VRPTW

> **Capstone Proof-of-Concept Module**  
> **Location:** `quantum_advanced/`  
> **Framework:** Qiskit 2.x & Qiskit Aer  
> **Target Problem:** Focused Vehicle Routing Subproblem (Solomon C101 Benchmark)

---

## 1. Project Purpose & Scope

This module provides a **genuine, gate-level Quantum Approximate Optimization Algorithm (QAOA)** demonstration for vehicle routing, formulated specifically as an explainable quantum proof-of-concept for the VRPTW capstone project.

### Core Objectives
1. **Genuine Gate-Level Circuit:** Unlike matrix shortcuts (such as loading precomputed classical energies into a `DiagonalGate`), this implementation builds a real QAOA circuit composed of single-qubit rotations ($H$, $R_Z$, $R_X$) and two-qubit entangling gates ($R_{ZZ}$).
2. **Authentic Benchmark Data:** Uses customer coordinates, demands, time windows, and vehicle capacity directly from the standard Solomon C101 benchmark (`c101.txt`).
3. **Multi-Level Constraint Validation:** Distinguishes clearly between:
   - **Model-Level Permutation Constraints** (encoded in the QUBO objective via quadratic penalty terms).
   - **Operational VRPTW Constraints** (vehicle capacity and customer time windows, validated via post-quantum classical schedule analysis).
4. **Honest Scientific Evaluation:** Executed on a local CPU quantum simulator (`AerSimulator`), compared against an exact classical exhaustive enumeration baseline, with zero fabricated results and no false claims of quantum supremacy.

---

## 2. Mathematical Formulation

### 2.1 Decision Variables
For a routing instance with $N$ customers (Node 0 is the depot; nodes $1, \dots, N$ are customers) and route positions $p \in \{0, \dots, N-1\}$:
$$x_{i, p} \in \{0, 1\}$$
- $x_{i, p} = 1$ if customer $(i + 1)$ is visited at route position $p$.
- Total binary variables / qubits: $n = N^2$. For $N = 3$ customers, $n = 9$ qubits; for $N = 4$ customers, $n = 16$ qubits.
- Qubit mapping index: $q = i \cdot N + p$.

### 2.2 Objective Function (Distance)
$$\text{Cost}_{\text{dist}}(x) = \sum_{i=0}^{N-1} d(0, i+1) x_{i, 0} + \sum_{p=0}^{N-2} \sum_{i=0}^{N-1} \sum_{j \neq i}^{N-1} d(i+1, j+1) x_{i, p} x_{j, p+1} + \sum_{i=0}^{N-1} d(i+1, 0) x_{i, N-1}$$

### 2.3 Permutation Constraints (Lagrange Penalty $P$)
1. **Each customer visited exactly once:**
   $$P \sum_{i=0}^{N-1} \left( \sum_{p=0}^{N-1} x_{i, p} - 1 \right)^2 = P \sum_{i=0}^{N-1} \left( -\sum_{p=0}^{N-1} x_{i, p} + 2 \sum_{p_1 < p_2} x_{i, p_1} x_{i, p_2} + 1 \right)$$
2. **Each route position occupied by exactly one customer:**
   $$P \sum_{p=0}^{N-1} \left( \sum_{i=0}^{N-1} x_{i, p} - 1 \right)^2 = P \sum_{p=0}^{N-1} \left( -\sum_{i=0}^{N-1} x_{i, p} + 2 \sum_{i_1 < i_2} x_{i_1, p} x_{i_2, p} + 1 \right)$$

Total QUBO objective minimized:
$$H_{\text{QUBO}}(x) = \text{Cost}_{\text{dist}}(x) + P \cdot \left[ \text{Row Constraints} + \text{Column Constraints} \right] = x^T Q x + \text{offset}$$

---

## 3. Analytical Mapping to Ising Hamiltonian & QAOA

To construct a quantum circuit, binary variables $x_u \in \{0, 1\}$ are mapped to Pauli-Z spin operators $Z_u \in \{+1, -1\}$ via:
$$x_u = \frac{I - Z_u}{2}$$

Substituting into $x^T Q x$ yields the Problem Hamiltonian:
$$H_C = \sum_{u=0}^{n-1} h_u Z_u + \sum_{u < v} J_{u, v} Z_u Z_v + C_0 \cdot I$$
where:
$$C_0 = \text{offset} + \sum_{u} \frac{Q_{u, u}}{2} + \sum_{u < v} \frac{Q_{u, v} + Q_{v, u}}{4}$$
$$h_u = -\frac{Q_{u, u}}{2} - \sum_{v \neq u} \frac{Q_{u, v} + Q_{v, u}}{4}, \quad J_{u, v} = \frac{Q_{u, v} + Q_{v, u}}{4}$$

### Quantum Circuit Implementation
1. **Initial Superposition State:**
   $$|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n}$$
2. **Problem Cost Unitary $U(C, \gamma) = e^{-i \gamma H_C}$:**
   - Single-qubit rotations: $R_Z(2 \gamma h_u)$ on qubit $u$.
   - Two-qubit entangling gates: $R_{ZZ}(2 \gamma J_{u, v})$ on pairs $(u, v)$.
3. **Mixer Unitary $U(B, \beta) = e^{-i \beta \sum X_u}$:**
   - Single-qubit rotations: $R_X(2 \beta)$ on qubit $u$.
4. **$p$-Layer Variational State:**
   $$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p U(B, \beta_l) U(C, \gamma_l) |\psi_0\rangle$$
5. **Measurement:** Standard computational basis measurement into classical register.

---

## 4. Quickstart & Execution

### Prerequisites & Installation (Windows 11)
The environment requires standard scientific Python libraries, Google OR-Tools, and Qiskit:
```powershell
python -m pip install --upgrade pip
pip install numpy scipy matplotlib qiskit qiskit-aer ortools
```

### Execution Commands

#### 1. Full 100-Customer Hybrid QAOA Workflow (All 100 Customers Accounted For)
```powershell
# Run the complete 100-customer workflow (Classical baseline vs Hybrid QAOA)
python run_full_dataset.py

# Quick smoke-test mode (runs 3 QAOA subproblems for rapid verification)
python run_full_dataset.py --quick

# Custom configuration (e.g. 10 subproblems, 400 shots, 10 max iterations)
python run_full_dataset.py --time-limit 5 --shots 400 --maxiter 10 --max-subproblems 10

# Primary script entrypoint with full-dataset delegation
python run_quantum_vrptw.py --full-dataset --quick
```

#### 2. Focused Gate-Level QAOA Demonstration (3-Customer Educational Baseline)
```powershell
python run_quantum_vrptw.py
```

#### 3. Run Complete Verification Test Suite (9 Unit Tests)
```powershell
python -m unittest test_full_dataset.py
```

### Available Command-Line Arguments (`run_full_dataset.py`)
- `--time-limit INT`: Classical OR-Tools solve time limit in seconds (default: 10).
- `--shots INT`: Measurement shots per QAOA subproblem on `AerSimulator` (default: 500).
- `--maxiter INT`: Maximum classical COBYLA optimizer evaluations per subproblem (default: 15).
- `--cluster-size INT`: Customer window size per QAOA subproblem (`3` or `4`, default: 3).
- `--max-subproblems INT`: Maximum QAOA subproblems to run (default: None = all route windows).
- `--initial-mode STR`: Initial solution mode: `'fast'` (`PATH_CHEAPEST_ARC`) or `'exact'` (`GUIDED_LOCAL_SEARCH`).
- `--quick`: Quick smoke-test mode (sets subproblems=3, shots=300, maxiter=8, time_limit=3s).
- `--seed INT`: Random seed for reproducibility (default: 42).

---

## 5. Demonstration Results (3-Customer Instance)

### 5.1 Quantum Circuit Specifications
- **Number of Qubits:** 9 ($N^2 = 3^2$)
- **Logical Circuit Depth:** 19 (Decomposed Depth: 49)
- **Gate Counts:**
  - $R_{ZZ}$ (two-qubit entangling interactions): 30
  - $R_Z$ (single-qubit cost rotations): 9
  - $R_X$ (single-qubit mixer rotations): 9
  - $H$ (Hadamard superposition gates): 9
  - `measure` (classical readout): 9

### 5.2 Variational Parameter Optimization (COBYLA)
- **Initial Parameters:** $\gamma_0 = 0.05, \beta_0 = 0.35$
- **Initial Expected Energy:** $276.28$
- **Optimized Parameters:** $\gamma^* = -0.08921, \beta^* = 0.48138$
- **Optimized Expected Energy:** $156.32$ (Energy reduction: $-119.96$)
- **Optimizer Runtime:** 0.19 seconds (24 evaluations)

### 5.3 Simulation Sampling (Qiskit AerSimulator, 5000 Shots)
- **Simulator Backend:** `aer_simulator` (CPU)
- **Feasible Shots Sampled:** 35 / 5000 (0.70%)
- **Optimal Shots Sampled:** 10 / 5000 (0.20%)
- **Best QAOA Measured Route:** `[0, 3, 2, 1, 0]`
- **Best QAOA Measured Bitstring:** `001010100`
- **Total Route Distance:** **41.8061** (Matches Exact Classical Optimum)

### 5.4 Comparison with Classical Exact Solver
| Metric | Exact Classical Enumeration | QAOA Quantum Simulation |
| :--- | :--- | :--- |
| **Method** | Exhaustive ($3! = 6$ permutations) | Variational QAOA ($p=1$, 9 qubits) |
| **Backend** | Classical CPU (Itertools) | Qiskit `AerSimulator` (5000 shots) |
| **Optimal Distance** | **41.8061** | **41.8061** (Exact Match) |
| **Best Route** | `[0, 3, 2, 1, 0]` | `[0, 3, 2, 1, 0]` |
| **Capacity Check** | $50.0 \le 200.0$ (PASSED) | $50.0 \le 200.0$ (PASSED) |
| **Time Windows** | 0 violations (PASSED) | 0 violations (PASSED) |
| **Solve / Sim Time** | 0.6 ms | ~2.3 s (including optimization & shots) |

---

## 6. Multi-Level Constraint Validation & Operational Insights

A critical distinction highlighted in this capstone proof-of-concept is the **two-stage constraint validation**:

```mermaid
flowchart LR
    A[Solomon C101 Instance] --> B[QUBO & Ising Mapping]
    B --> C[QAOA Circuit on AerSimulator]
    C --> D[Measured Bitstrings]
    D --> E{Stage 1: Model Permutation Constraints}
    E -->|Infeasible| F[Reject: Row/Col Sum != 1]
    E -->|Feasible| G[Decoded Tour Candidate]
    G --> H{Stage 2: Classical VRPTW Validation}
    H -->|Capacity Check| I[Demand <= Vehicle Capacity]
    H -->|Schedule Timing| J[Arrival <= Due Time at all stops]
```

### Key Analytical Finding
In pure Euclidean distance, routes `[0, 3, 2, 1, 0]` and `[0, 1, 2, 3, 0]` have the **exact same distance** ($41.8061$) because distance is symmetric in reverse:
1. **Route `[0, 3, 2, 1, 0]`:**
   - Visits C3 (due: 146) early at $t=65$, departs $t=155$.
   - Visits C2 (due: 870) at $t=825$, departs $t=915$.
   - Visits C1 (due: 967) at $t=917$, departs $t=1007$.
   - Returns to depot at $t=1025.7 \le 1236$.
   - **Result: 0 time window violations (100% FEASIBLE).**
2. **Route `[0, 1, 2, 3, 0]`:**
   - Visits C1 at $t=912$. By the time it arrives at C3, $t = 1099 > 146$ (953 time units late!).
   - **Result: 2 time window violations (INFEASIBLE).**

**Takeaway for Capstone Defense:** Pure routing QUBOs search the topological permutation space for distance minimization. Classical post-validation is essential to verify physical VRPTW temporal constraints unless complex multi-commodity time-discretized QUBOs are formulated.

---

## 7. Generated Visualizations & Artifacts

All experiment plots and data are automatically saved to `outputs/`:
- `optimization_convergence.png`: Classical optimizer energy descent curve.
- `bitstring_distribution.png`: Two-panel histogram of top measured states and the feasible route candidate spectrum.
- `route_comparison.png`: 2D spatial coordinate map with customer nodes, depot, time windows, and route overlay.
- `circuit_summary.png`: Architecture dashboard with gate counts and optimized angles.
- `circuit_diagram.txt`: Full ASCII text rendering of the Qiskit quantum circuit.
- `quantum_results.json`: Complete machine-readable experimental data.

---

## 8. Limitations & Honest Discussion (For Capstone Defense)

1. **Qubit Scalability ($O(N^2)$):**
   Order-based TSP/routing formulation scales with $N^2$ qubits. For 3 customers, 9 qubits are needed; for 4 customers, 16 qubits; for 10 customers, 100 qubits. Current NISQ hardware cannot run 100-qubit entangling circuits without decoherence.
2. **Exponentially Diluted Feasible Subspace:**
   Out of $2^{N^2}$ possible computational basis states, only $N!$ represent valid permutation routes:
   - For $N = 3$: $6 / 512 \approx 1.17\%$ of states are feasible.
   - For $N = 4$: $24 / 65,536 \approx 0.036\%$ of states are feasible.
   - For $N = 5$: $120 / 33,554,432 \approx 3.58 \times 10^{-6}\%$ of states are feasible.
   Standard unconstrained QAOA with $p=1$ struggles to concentrate probability on such narrow subspaces without advanced constraint-preserving mixers (e.g., XY-mixers) or higher circuit depth ($p \ge 3$).
3. **Classical Simulation vs. Quantum Advantage:**
   Running QAOA on `AerSimulator` or `Statevector` is simulated on a classical CPU. It is a proof-of-concept demonstrating quantum algorithmic structure, **not evidence of quantum supremacy**.
4. **Hybrid VRPTW Recommendation:**
   For realistic multi-vehicle VRPTW (100+ customers), pure end-to-end QUBO is impractical. The recommended industry direction is a **quantum-classical hybrid** architecture: classical clustering/decomposition (e.g. cluster-first route-second) to partition customers into small clusters of 3–5 stops, solved via quantum subroutines.

---

## 9. Multi-Instance Empirical Scaling Experiments ($N \in [3, 4, 5, 6]$)

We conducted live comparative benchmarks across problem sizes using identical Solomon C101 coordinates and constraints:

| $N$ (Cust) | Qubits ($N^2$) | Hilbert Dimension ($2^{N^2}$) | Statevector RAM | Feasible Subspace ($N! / 2^{N^2}$) | QAOA Feasible Shots | QAOA Best Dist | Classical Exact Dist | Optimality Gap | Quantum Simulation Runtime | Laptop Feasibility Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3** | 9 | 512 | 8.0 KB | 1.17% | 35 / 5000 (0.70%) | **41.8061** | **41.8061** | **0.00%** | 2.24s | **Ran Smoothly** |
| **4** | 16 | 65,536 | 1.0 MB | 0.0366% | 4 / 2000 (0.20%) | **43.4464** | **42.4116** | **2.44%** | 6.15s | **Ran Smoothly** |
| **5** | 25 | 33,554,432 | 512.0 MB | $3.58 \times 10^{-6}\%$ | 0 / 2000 (0.00%) | N/A | **42.4198** | N/A | 52.40s | **Heavy Computation (Near Limit)** |
| **6** | 36 | 68,719,476,736 | 1,024.0 GB (1.0 TB) | $1.05 \times 10^{-8}\%$ | Skipped | N/A | **46.1494** | N/A | 0.00s | **SAFELY ABORTED (Exceeds RAM)** |

### Key Experimental Insights
1. **$N=3$ (9 qubits):** Ideal classroom proof-of-concept. The feasible subspace ($1.17\%$) is dense enough that QAOA finds both the exact optimum and satisfies all VRPTW time windows in 2.24 seconds.
2. **$N=4$ (16 qubits):** Demonstrates variational optimization on 16 qubits. QAOA sampled 4 valid permutations and discovered a route within **2.44%** of the global optimum.
3. **$N=5$ (25 qubits, 512 MB RAM):** Represents the boundary of laptop classical simulation. Statevector dimension is $33.5 \times 10^6$ amplitudes. In unconstrained QAOA ($p=1$), the feasible subspace drops to $3.58 \times 10^{-6}\%$. With 2000 random measurement shots, 0 feasible permutations were drawn because sampling a valid permutation is a 1-in-280,000 event. Meanwhile, classical exhaustive search evaluated all 120 permutations in **28 milliseconds**.
4. **$N=6$ (36 qubits, 1.0 TB RAM):** Pre-flight resource estimation safely intercepted and aborted the quantum simulation before execution. A single statevector of 36 qubits requires **1,024 GB of RAM**, which would instantly crash an ordinary laptop CPU. Classical exhaustive search evaluated all 720 permutations in **296 milliseconds** ($46.1494$).

---

## 10. Pre-Flight Hardware Resource Estimation & Safety Gating

Before allocating simulator memory, `resource_estimator.py` inspects the instance size:

```bash
python run_quantum_vrptw.py --estimate
```

```
 Customers |   Qubits |     Hilbert States |   RAM Required |  Feasible Routes |     Feasible % |               Status
-------------------------------------------------------------------------------------------------------------------
         3 |        9 |                512 |         8.0 KB |                6 |        1.1719% |      SAFE_FOR_LAPTOP
         4 |       16 |             65,536 |        1.00 MB |               24 |        0.0366% |      SAFE_FOR_LAPTOP
         5 |       25 |         33,554,432 |      512.00 MB |              120 |      3.58e-04% |    HEAVY_COMPUTATION
         6 |       36 |               2^36 |         1.0 TB |              720 |      1.05e-06% | EXCEEDS_MEMORY_LIMIT
         8 |       64 |               2^64 | 268,435,456 TB |           40,320 |      2.19e-13% | EXCEEDS_MEMORY_LIMIT
        10 |      100 |              2^100 |     > 10^18 TB |        3,628,800 | 0.0% (Infinitesimal) | EXCEEDS_MEMORY_LIMIT
       100 |    10000 |            2^10000 |   > 10^2998 TB |           100!   | 0.0% (Infinitesimal) | EXCEEDS_MEMORY_LIMIT
```

**Rule:** Instances with $> 28$ qubits or $> 2.0\text{ GB}$ statevector RAM are automatically blocked from full statevector simulation, preventing system freeze or out-of-memory crashes.

---

## 11. Compact Logarithmic Binary Encoding vs. $N^2$ 1-Hot Encoding

Instead of representing order with $N^2$ one-hot variables $x_{i, p}$, each stop index can be encoded as a binary integer using $K = \lceil \log_2 N \rceil$ qubits:

$$\text{Compact Qubits} = N \cdot \lceil \log_2 N \rceil$$

```bash
python run_quantum_vrptw.py --decompose
```

| Customers | 1-Hot Qubits ($N^2$) | Compact Qubits ($N \lceil \log_2 N \rceil$) | Qubit Reduction | Coupling Complexity |
| :---: | :---: | :---: | :---: | :---: |
| 3 | 9 | 6 | **33.3%** | 2-local quadratic vs 3-order PUBO |
| 4 | 16 | 8 | **50.0%** | 2-local quadratic vs 4-order PUBO |
| 5 | 25 | 15 | **40.0%** | 2-local quadratic vs 5-order PUBO |
| 6 | 36 | 18 | **50.0%** | 2-local quadratic vs 6-order PUBO |
| 8 | 64 | 24 | **62.5%** | 2-local quadratic vs 8-order PUBO |
| 10 | 100 | 40 | **60.0%** | 2-local quadratic vs 10-order PUBO |
| 100 | 10,000 | 700 | **93.0%** | 2-local quadratic vs 100-order PUBO |

### Fundamental Algorithmic Trade-Off
- **Advantage:** Compact binary reduces qubit count by 33% to 93% (e.g., $N=6$ requires only 18 qubits instead of 36, fitting into 4 MB RAM instead of 1 TB).
- **Severe Disadvantage:** The permutation constraint ("all stops must be distinct") transforms from simple pairwise quadratic penalties into high-order polynomial unconstrained binary optimization (**PUBO**). Quadradizing high-order terms back to 2-local Ising Hamiltonians requires introducing $O(N^2)$ ancilla slack qubits and extra two-qubit gates, largely neutralizing the qubit savings on NISQ hardware.

---

## 12. Hybrid Decomposition Paradigm (Cluster-First, Route-Second)

To solve large-scale problems like Solomon C101 (100 customers), we implement the industry-standard **Cluster-First, Route-Second** decomposition:

```mermaid
flowchart TD
    A[Full 100-Customer Solomon C101 Benchmark] --> B[Classical Polar Sweep Clustering]
    B --> C[25 Capacity-Feasible Subclusters of 3-5 Stops]
    C --> D[Subproblem 1: 16 Qubits QAOA]
    C --> E[Subproblem 2: 16 Qubits QAOA]
    C --> F[Subproblem ...: 9-25 Qubits QAOA]
    D --> G[Classical Schedule Reassembly]
    E --> G
    F --> G
    G --> H[Final End-to-End VRPTW Dispatch Plan]
```

- **Classical Stage:** Polar sweep partitions 100 customers into 25 clusters ($\le 200\text{ demand}$, 3–5 customers each).
- **Quantum Subroutine:** Each cluster requires only 9 to 25 qubits, which is within the tractable limit of NISQ hardware or local simulation.
- **Classical Synthesis:** Reassembles solved sub-tours and checks vehicle schedule compatibility.

---

## 13. Full 100-Customer Solomon C101 Classical Evaluation (OR-Tools)

To satisfy capstone benchmarking without fabricating quantum results, the complete 100-customer Solomon C101 benchmark was solved using Google OR-Tools:

```bash
python run_quantum_vrptw.py --full-scale
```

### Full-Scale Classical Results (Clearly Separated from Quantum PoC)
- **Problem Instance:** Solomon C101 (Depot + 100 Customers)
- **Fleet Requirements:** **10 Vehicles** (Each capacity 200.0)
- **Total Fleet Route Distance:** **828.94**
- **Customer Demand Satisfied:** 1,810 / 2,000 units (100% fulfilled)
- **Time Window Feasibility:** **100% Feasible** (0 violations across all 100 customers)
- **Classical Solve Time:** **5.00 seconds** (Parallel Guided Local Search)
- **Comparison Visualization:** Saved to `outputs/full_100_customers_classical.png`

> **Academic Conclusion:** While a 100-customer quantum simulation would require $10,000$ qubits and $> 10^{2998}\text{ TB}$ of RAM (which is physically impossible on classical computers and unavailable on existing NISQ hardware), modern classical solvers solve the entire 100-customer problem with 10 vehicles and time windows in **5 seconds**. Quantum computing for VRPTW remains a promising exploratory research direction for small subroutines within hybrid decomposition schemes.

---

## 14. Complete 100-Customer Closed-Loop Hybrid QAOA Workflow (`run_full_dataset.py`)

A production-grade, mathematically grounded **Quantum-Classical Large Neighborhood Search (QC-LNS)** pipeline is implemented to account for all 100 customers in the Solomon C101 benchmark.

### 14.1 Mathematical Formulation: Subroute QUBO with Fixed Endpoints

To optimize local route segments without breaking global route continuity, the subproblem is formulated as a fixed-endpoint Hamiltonian path.
Given a current route $\Pi = [0, c_1, \dots, u, s_1, \dots, s_k, w, \dots, 0]$, we extract a window of $k$ customers $S = \{s_1, \dots, s_k\}$ situated between fixed predecessor node $u$ and successor node $w$.

#### Decision Variables
For each customer $i \in S$ and local position $t \in \{0, \dots, k-1\}$:
$$x_{i, t} \in \{0, 1\}$$
- $x_{i, t} = 1$ if customer $i$ is visited at local position $t$; $0$ otherwise.
- Total binary variables / qubits: $n = k^2$. For $k = 3$ customers, $n = 9$ qubits.
- Qubit mapping index: $q = \text{index}(i) \cdot k + t$.

#### Objective Function Components
1. **Transition Routing Cost ($H_C$):**
   Includes connection costs from predecessor $u$, pairwise transitions between customers in $S$, and connection to successor $w$:
   $$H_C(x) = \sum_{i \in S} d(u, i) x_{i, 0} + \sum_{t=0}^{k-2} \sum_{i \in S} \sum_{j \in S, j \neq i} d(i, j) x_{i, t} x_{j, t+1} + \sum_{i \in S} d(i, w) x_{i, k-1}$$

2. **Permutation Assignment Constraints ($H_A$):**
   Enforces that every customer $i \in S$ is assigned exactly once, and each position $t$ is occupied by exactly one customer:
   $$H_A(x) = A \sum_{i \in S} \left( \sum_{t=0}^{k-1} x_{i, t} - 1 \right)^2 + A \sum_{t=0}^{k-1} \left( \sum_{i \in S} x_{i, t} - 1 \right)^2$$

3. **Soft Time Window Penalty ($H_{TW}$):**
   Penalizes scheduling customer $i$ at position $t$ if customer $i$'s ready time exceeds the departure deadline at $w$:
   $$H_{TW}(x) = \lambda_{TW} \sum_{i \in S} \sum_{t=0}^{k-1} \max\left(0, \frac{\text{ready}_i - \text{due}_w}{10}\right) x_{i, t}$$

4. **Ising Hamiltonian Mapping:**
   Using the transformation $x_{i, t} = \frac{I - Z_{i, t}}{2}$, the combined QUBO $H = H_C + H_A + H_{TW}$ is mapped to the 2-local spin Hamiltonian:
   $$H_C = \sum_{u=0}^{n-1} h_u Z_u + \sum_{u < v} J_{u, v} Z_u Z_v + C_0 \cdot I$$

---

### 14.2 The 8-Step Closed-Loop Hybrid Optimization Loop

```mermaid
flowchart TD
    A[Solomon C101: 100 Customers + Depot 0] --> B[Monolithic Classical Solver: OR-Tools]
    B --> C[Initial 100% Feasible Solution: 10 Vehicles, 854.31 Dist]
    C --> D[Subproblem Selector: Extract Window u -> S -> w]
    D --> E[SubrouteQUBO & Ising Hamiltonian: 9 Qubits]
    E --> F[Qiskit AerSimulator: Gate-Level QAOA Circuit]
    F --> G[COBYLA Tuning & Measurement Sampling]
    G --> H[Bitstring Decoder: Candidate Permutations]
    H --> I{Fleet Validator: Single-Route Audit}
    I -->|Feasible & Dist Improves| J[ACCEPT: Update Route]
    I -->|Feasible & Same Dist| K[CONFIRMED: Retain Route]
    I -->|Worse Dist or Infeasible| L[REJECT & Fallback: Retain Route]
    J --> M{More Windows?}
    K --> M
    L --> M
    M -->|Yes| D
    M -->|No| N[Final Independent Fleet Audit: validate_fleet]
    N --> O[100% Feasible 100-Customer Dispatch Plan]
```

1. **Full Problem Ingestion:** Loads all 100 customers ($1 \dots 100$) + depot $0$ from Solomon C101. Total demand = 1,810.0, vehicle capacity = 200.0.
2. **Classical Feasible Baseline:** Uses OR-Tools to obtain a 100% feasible initial solution covering all 100 customers with 10 vehicles.
3. **Subroute Window Extraction:** Slides a $k$-customer window across vehicle routes, extracting $u \to S \to w$.
4. **QUBO Formulation:** Instantiates `SubrouteQUBO` with fixed boundary transitions.
5. **QAOA Execution:** Builds and simulates parameterized quantum circuits ($H^{\otimes 9}$, $R_Z, R_{ZZ}$, $R_X$) using Qiskit Aer (`AerSimulator`).
6. **Bitstring Decoding:** Decodes measured bitstrings into candidate customer permutations.
7. **Single-Route Feasibility & Improvement Audit (`validate_single_route`):**
   - Simulates vehicle timeline: arrival, waiting, 90-unit service duration, departure, and return to depot.
   - Evaluates vehicle load $\le 200.0$ and arrival $\le \text{due\_time}$ for every stop.
   - **Accept/Reject Decision:**
     - **ACCEPTED:** If candidate is feasible AND $\Delta d < 0$.
     - **CONFIRMED_FEASIBLE:** If candidate matches current route distance ($\Delta d = 0$) and is feasible.
     - **REJECTED (Safe Fallback):** If candidate is worse ($\Delta d > 0$) or violates time windows/capacity. The existing feasible route is safely retained.
8. **Final Independent Fleet Audit (`validate_fleet`):**
   - Audits all 100 customers independently. Verifies zero missing, zero duplicates, zero overloads, and zero late stops.

---

### 14.3 Measured Benchmark Results (100 Customers)

| Metric | Classical Baseline (OR-Tools) | Hybrid QAOA Solver | Status / Verification |
| :--- | :--- | :--- | :--- |
| **Dataset Size** | **100 Customers + 1 Depot** | **100 Customers + 1 Depot** | Solomon C101 Benchmark |
| **Customer Coverage** | **100 / 100 (100%)** | **100 / 100 (100%)** | **100% Unique Coverage** |
| **Unique Customers** | 100 | 100 | Zero omissions, zero duplicates |
| **Missing Customers** | 0 missing | 0 missing | PASSED |
| **Duplicate Customers** | 0 duplicate | 0 duplicate | PASSED |
| **Vehicles Used** | **10 Vehicles** | **10 Vehicles** | Matches theoretical minimum |
| **Total Fleet Route Distance** | **828.94** | **854.31** (+3.06% initial) | Audited across complete routes |
| **Total Demand Delivered** | 1,810 / 1,810 (100%) | 1,810 / 1,810 (100%) | 100% demand satisfied |
| **Vehicle Capacity Check** | **PASSED (0 overloads)** | **PASSED (0 overloads)** | Every vehicle load $\le 200.0$ |
| **Time Window Compliance** | **PASSED (0 violations)** | **PASSED (0 violations)** | **100% FEASIBLE (0 late stops)** |
| **Overall Fleet Feasibility** | **100% FEASIBLE** | **100% FEASIBLE** | **Rigorous audit confirmed** |
| **QAOA Subproblems Run** | N/A (Monolithic Classical) | **10 Subproblems** (3 cust each) | Real gate circuits executed |
| **Total Qubits Allocated** | N/A | **90 Qubits total** (9q per circuit)| Fits comfortably in RAM |
| **Confirmed Feasible Orders**| N/A | **7 confirmed** | Sampled optimal permutations |
| **Rejected Candidates (Safe)**| N/A | **3 rejected (fallback retained)**| Zero route regression |
| **Quantum Execution Runtime** | N/A | **9.46 seconds** (`AerSimulator`) | Real CPU quantum simulation |
| **Total Wall-Clock Runtime** | **5.09 seconds** | **11.15 seconds** | Complete end-to-end solve |

---

### 14.4 Laptop Protection & Resource Profile

- **Target Laptop Hardware:** Intel Core i3-1005G1 CPU, 8 GB RAM, NVIDIA MX330 GPU, Windows 11.
- **Resource Management:**
  - Subproblem size $k=3 \implies k^2 = 9$ qubits.
  - Hilbert space dimension: $2^9 = 512$ states.
  - Statevector RAM: **8.0 KB** per circuit.
  - Simulation time: ~0.4s per subproblem on CPU.
  - Total RAM utilized during the full 100-customer run remains under **150 MB**, preventing system freeze or crash.

---

### 14.5 Unit Test Suite (9 Tests)

Run the full verification suite covering all project requirements:
```powershell
python -m unittest test_full_dataset.py
```
```
.........
----------------------------------------------------------------------
Ran 9 tests in 13.231s

OK
```
1. `test_01_full_dataset_ingestion_100_customers`: Verifies all 100 customer records and depot.
2. `test_02_exactly_100_distinct_customer_ids`: Verifies IDs 1..100 are strictly unique.
3. `test_03_qubo_subroute_construction_fixed_endpoints`: Validates `SubrouteQUBO` with fixed $u, w$.
4. `test_04_qaoa_circuit_gate_level_structure`: Verifies $H$, $R_Z$, $R_{ZZ}$, $R_X$ gates.
5. `test_05_actual_qaoa_circuit_execution_aer_simulator`: Executes QAOA on `AerSimulator` with COBYLA.
6. `test_06_decoding_measured_outcomes`: Verifies bitstring decoding to valid permutations.
7. `test_07_safe_fallback_when_qaoa_proposal_invalid_or_worse`: Verifies fallback retention on bad moves.
8. `test_08_classical_baseline_feasibility_100_customers`: Verifies OR-Tools 100-customer feasibility.
9. `test_09_hybrid_solver_complete_workflow_and_telemetry`: Runs hybrid loop, auditing 100 customers and telemetry.


