# Mathematical Formulation
## Hybrid Quantum-Classical Optimization for Integrated Warehouse Inventory Planning and Last-Mile Vehicle Routing

## 1. Problem Definition

The problem considers a single warehouse serving multiple customers using a fleet of vehicles.

The warehouse has limited inventory. Each customer places an order with a specific demand.

The model must decide:

1. Which customer orders should be fulfilled completely.
2. Which customers should be rejected because of limited inventory.
3. Which vehicle should serve each fulfilled customer.
4. The order in which each vehicle visits its assigned customers.
5. How to minimize transportation and unfulfilled-order costs.

Each customer order must either be fulfilled completely or not fulfilled.

Vehicles start and finish at the warehouse.

The model considers vehicle capacity, customer time windows, and warehouse operating hours.

---

## 2. Sets and Indices

| Symbol | Description |
|---|---|
| N | Set of customers |
| K | Set of available vehicles |
| V | Set of all locations, including warehouse |
| 0 | Warehouse (depot) |
| i, j | Customer or warehouse indices |
| k | Vehicle index |

The location set is:

V = {0} ∪ N

For our initial experiment:

- One warehouse
- 100 customers
- 25 available vehicles
- One product type

---

## 3. Input Parameters

| Symbol | Description |
|---|---|
| q_i | Demand of customer i |
| I | Initial warehouse inventory |
| Q_k | Capacity of vehicle k |
| c_ij | Transportation cost from location i to j |
| t_ij | Travel time from location i to j |
| a_i | Earliest service start time at customer i |
| b_i | Latest service start time at customer i |
| s_i | Service duration at customer i |
| P_i | Penalty for not fulfilling customer i |
| F_k | Fixed cost of using vehicle k |
| M | A sufficiently large positive constant |

For the initial experiment:

- Warehouse inventory I = 1000 units.
- Vehicle capacity Q_k = 200 units.
- Available vehicles = 25.
- Transportation cost is initially represented by Euclidean distance.
- Travel time is initially assumed equal to Euclidean distance.
- Customer demands and time windows come from the C101 dataset.

The warehouse inventory is an experimental project parameter, not a value provided by C101.

---

## 4. Decision Variables

### 4.1 Inventory decision

y_i = 1 if customer i's complete order is fulfilled.

y_i = 0 if customer i's order is rejected.

### 4.2 Routing decision

x_ijk = 1 if vehicle k travels directly from location i to location j.

x_ijk = 0 otherwise.

### 4.3 Vehicle usage decision

u_k = 1 if vehicle k is used.

u_k = 0 otherwise.

### 4.4 Customer assignment decision

v_ik = 1 if vehicle k visits customer i.

v_ik = 0 otherwise.

### 4.5 Timing decision

T_ik = service start time at customer i by vehicle k.

T_0k = departure time of vehicle k from the warehouse.

R_k = return time of vehicle k to the warehouse.

---

## 5. Objective Function

We want to minimize the total operational cost.

The objective contains three components:

1. Transportation cost.
2. Penalty for unfulfilled customer orders.
3. Fixed vehicle usage cost.

The objective function is:

Minimize Z:

Z = Σ(k∈K) Σ(i∈V) Σ(j∈V, j≠i) c_ij x_ijk

    + Σ(i∈N) P_i(1-y_i)

    + Σ(k∈K) F_k u_k

Explanation:

- The first term minimizes total transportation cost.
- The second term penalizes rejected customer orders.
- The third term discourages unnecessary vehicle usage.

The penalty values must be calibrated so that transportation savings do not encourage unreasonable rejection of customer orders.

---

## 6. Constraints

### Constraint 1: Warehouse inventory

The total demand of fulfilled customers cannot exceed available warehouse stock.

Σ(i∈N) q_i y_i ≤ I

Explanation:

The warehouse can only fulfil orders when sufficient inventory is available.

### Constraint 2: Each fulfilled customer is assigned exactly once

Σ(k∈K) v_ik = y_i

For every customer i.

Explanation:

- If y_i = 1, exactly one vehicle must serve the customer.
- If y_i = 0, no vehicle can serve the customer.

### Constraint 3: Incoming route connection

Σ(j∈V, j≠i) x_jik = v_ik

For every customer i and vehicle k.

Explanation:

If a vehicle visits a customer, it must enter that customer exactly once.

### Constraint 4: Outgoing route connection

Σ(j∈V, j≠i) x_ijk = v_ik

For every customer i and vehicle k.

Explanation:

If a vehicle visits a customer, it must leave that customer exactly once.

### Constraint 5: Vehicle departure

Σ(j∈N) x_0jk = u_k

For every vehicle k.

Explanation:

A vehicle that is used must leave the warehouse once.

An unused vehicle must not leave the warehouse.

### Constraint 6: Vehicle return

Σ(i∈N) x_i0k = u_k

For every vehicle k.

Explanation:

Every used vehicle must return to the warehouse exactly once.

### Constraint 7: Vehicle capacity

Σ(i∈N) q_i v_ik ≤ Q_k u_k

For every vehicle k.

Explanation:

The total quantity delivered by a vehicle cannot exceed its capacity.

### Constraint 8: Vehicle assignment

v_ik ≤ u_k

For every customer i and vehicle k.

Explanation:

An unused vehicle cannot serve customers.

### Constraint 9: Customer time windows

When vehicle k serves customer i:

T_ik ≥ a_i - M(1-v_ik)

T_ik ≤ b_i + M(1-v_ik)

These constraints ensure that the time-window restriction applies only when vehicle k visits customer i.

Explanation:

The vehicle must begin service within the customer's allowed time window.

### Constraint 10: Travel-time precedence

For every selected arc between two customers:

T_jk ≥ T_ik + s_i + t_ij - M(1-x_ijk)

For i, j ∈ N, i ≠ j.

Explanation:

If vehicle k travels from customer i to customer j, it cannot begin service at j before completing service at i and travelling to j.

### Constraint 11: Departure-time precedence

For a selected route from the warehouse to customer j:

T_jk ≥ T_0k + t_0j - M(1-x_0jk)

Explanation:

A customer cannot be served before the vehicle leaves the warehouse and reaches that customer.

### Constraint 12: Warehouse return time

For a selected route from customer i to the warehouse:

R_k ≥ T_ik + s_i + t_i0 - M(1-x_i0k)

The vehicle must return within the warehouse operating horizon:

R_k ≤ b_0 + M(1-u_k)

### Constraint 13: Variable domains

y_i ∈ {0,1}

x_ijk ∈ {0,1}

v_ik ∈ {0,1}

u_k ∈ {0,1}

T_ik ≥ 0

R_k ≥ 0

Subtour elimination is required to prevent vehicles from forming disconnected customer loops. Positive travel-time sequencing constraints can eliminate customer-only subtours when all travel times are strictly positive. We will verify this in the implementation.

---

## 7. Integrated Model Summary

The model jointly decides:

- Which customer orders to fulfil.
- How much warehouse inventory to allocate.
- Which vehicle serves each fulfilled customer.
- The delivery sequence of each vehicle.
- How to minimize transportation and rejection costs.

The model must satisfy:

1. Warehouse inventory availability.
2. Complete-order fulfilment.
3. Unique customer assignment.
4. Vehicle capacity.
5. Route continuity.
6. Warehouse departure and return.
7. Customer time windows.
8. Vehicle operating constraints.

---

## 8. Planned Quantum Formulation

The mathematical model above is a mixed-integer optimization formulation.

Our next step is to formulate a smaller binary optimization problem using QUBO.

QUBO means Quadratic Unconstrained Binary Optimization.

The QUBO objective has the form:

H(x) = xᵀ Q x

where:

- x is a binary decision vector.
- Q is the QUBO coefficient matrix.
- H(x) is the objective energy.

Constraints will be represented using penalty terms or handled through a suitable feasibility-preserving encoding.

The initial quantum simulation will use a small customer instance because the number of binary variables grows rapidly with the number of customers and vehicles.

The full C101 problem will continue to be evaluated using classical optimization methods.

---

## 9. Important Modelling Assumptions

1. One warehouse is considered initially.
2. One product type is considered initially.
3. Each customer has one order.
4. Customer orders cannot be partially fulfilled.
5. All vehicles begin and end at the warehouse.
6. Vehicles have known capacities.
7. Customer demands are deterministic.
8. Customer time windows are known.
9. Travel times are initially based on Euclidean distances.
10. Initial warehouse inventory is an experimental parameter.

Future extensions may include multiple warehouses, multiple products, replenishment, and uncertain customer demand.
