import itertools
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from routing.customers import load_c101


warehouse, all_customers = load_c101()
customers = all_customers[:4]
customer_ids = [customer["id"] for customer in customers]
demands = [int(customer["demand"]) for customer in customers]
STOCK = 60
PENALTY = 100


def calculate_energy(selected, slack):
    fulfilled_count = sum(selected)
    inventory_balance = (
        sum(demand * decision for demand, decision in zip(demands, selected))
        + slack
        - STOCK
    )
    return -fulfilled_count + PENALTY * inventory_balance ** 2


def solve_inventory_qubo():
    slack_bits = [1, 2, 4, 8, 16, 32, 64]
    best_solution = None

    for selected in itertools.product((0, 1), repeat=len(customers)):
        for slack_selection in itertools.product(
            (0, 1),
            repeat=len(slack_bits),
        ):
            slack = sum(
                bit * value
                for bit, value in zip(slack_bits, slack_selection)
            )
            energy = calculate_energy(selected, slack)
            if best_solution is None or energy < best_solution["energy"]:
                best_solution = {
                    "selected": selected,
                    "slack": slack,
                    "energy": energy,
                }

    return best_solution


if __name__ == "__main__":
    print("\nINVENTORY QUBO MODEL")
    print("--------------------")
    print("Customers:", customer_ids)
    print("Demands:", demands)
    print("Available stock:", STOCK)
    print("Penalty coefficient:", PENALTY)

    solution = solve_inventory_qubo()
    selected = solution["selected"]
    fulfilled_ids = [
        customer_id
        for customer_id, decision in zip(customer_ids, selected)
        if decision
    ]
    fulfilled_demand = sum(
        demand for demand, decision in zip(demands, selected) if decision
    )

    print("\nQUBO SOLUTION")
    print("-------------")
    print("Selected variables:", list(selected))
    print("Fulfilled customer IDs:", fulfilled_ids)
    print("Fulfilled customers:", sum(selected))
    print("Fulfilled demand:", fulfilled_demand)
    print("Slack value:", solution["slack"])
    print("Energy:", solution["energy"])

    print("\nQUBO VALIDATION")
    print("----------------")
    print("Inventory valid:", fulfilled_demand <= STOCK)
    print(
        "Constraint balance valid:",
        fulfilled_demand + solution["slack"] == STOCK,
    )
