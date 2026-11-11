# Project Definition

## 1. Problem Statement

Warehouse inventory planning and last-mile vehicle routing are two important activities in supply-chain management.

A warehouse must determine whether its available inventory can satisfy customer orders and how the available stock should be allocated. At the same time, delivery vehicles must visit multiple customers while respecting vehicle capacity limitations.

When inventory allocation and vehicle routing are optimized independently, the resulting decisions may not minimize the overall operational cost.

Therefore, this project aims to develop a hybrid quantum-classical optimization framework that integrates warehouse inventory planning and last-mile vehicle routing for a single warehouse serving multiple customers.

The proposed framework will investigate classical optimization and QUBO-based QAOA simulation to find feasible and cost-effective inventory and delivery decisions.

## 2. Project Objectives

1. Develop an inventory planning model for a single warehouse storing one product.
2. Manage multiple customer orders with different quantities.
3. Implement full-order-only inventory allocation.
4. Determine which customer orders can be fulfilled using available warehouse stock.
5. Optimize delivery routes while satisfying vehicle capacity constraints.
6. Formulate the integrated inventory-routing problem mathematically.
7. Convert the optimization problem into a QUBO formulation.
8. Implement QAOA using quantum circuit simulation.
9. Integrate classical and quantum optimization components.
10. Compare classical and hybrid solutions using appropriate performance metrics.

## 3. Project Scope

### Initial Implementation

- One central warehouse.
- One product type.
- Multiple customers.
- One order per customer.
- Different order quantities.
- Full-order-only fulfilment policy.
- An order is either completely fulfilled or not fulfilled.
- No partial order allocation.
- Limited warehouse inventory.
- Multiple delivery vehicles.
- Fixed vehicle capacities.
- Delivery routes beginning and ending at the warehouse.
- Single-period inventory planning.
- QUBO formulation and QAOA simulation.

### Future Extensions

- Multiple product types.
- Multiple orders per customer.
- Partial order fulfilment.
- Inventory replenishment.
- Multiple warehouses.
- Customer-to-warehouse assignment.
- Multi-depot vehicle routing.

## 4. Initial Assumptions

1. The system operates with one warehouse.
2. The warehouse stores one product type.
3. Each customer places one order.
4. Each order specifies a quantity of the product.
5. An order must be fulfilled completely or rejected.
6. Partial order fulfilment is not permitted.
7. Warehouse inventory is known before delivery planning.
8. Vehicle capacities are known.
9. Customer locations are available.
10. Delivery distances are calculated from location coordinates.
11. Each fulfilled customer is assigned to exactly one delivery vehicle.
12. Unfulfilled customers are not included in delivery routes.
13. Vehicles start from and return to the warehouse.
14. Inventory planning is initially performed for a single planning period.
15. Travel distances and costs are deterministic in the initial implementation.
16. Unfulfilled orders incur a penalty in the optimization objective.
17. All fulfilled orders must satisfy warehouse inventory and vehicle capacity constraints.

## 5. Initial Mathematical Direction

Let:

- I = available warehouse inventory.
- d_i = quantity requested by customer i.
- y_i = binary order fulfilment decision.

The order fulfilment variable is:

y_i = 1 if customer i's order is fulfilled; otherwise y_i = 0.

The inventory constraint is:

Sum(d_i * y_i) <= I

Only fulfilled orders can be assigned to delivery vehicles.

The integrated objective initially considers:

Minimize total transportation cost + unfulfilled-order penalty.

The detailed mathematical formulation and QUBO conversion will be developed in the mathematical modelling phase.

## 6. Expected Outcomes

The project is expected to produce:

- A warehouse inventory allocation plan.
- A list of fulfilled and unfulfilled customer orders.
- Optimized delivery routes.
- Total inventory and transportation cost estimates.
- A classical optimization solution.
- A QUBO-based optimization formulation.
- A QAOA simulation solution.
- A comparison of classical and hybrid optimization performance.

## 7. Research Direction

The research will investigate the feasibility and performance of integrating warehouse inventory planning and last-mile vehicle routing using a hybrid quantum-classical optimization approach.

The contribution and research gap will be finalized after reviewing relevant published research and evaluating the proposed methodology.