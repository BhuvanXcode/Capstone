from inventory_data import INITIAL_STOCK, load_inventory_data


def greedy_inventory_allocation():
    _, customers = load_inventory_data()

    # Sort customers by demand (smallest first)
    sorted_customers = sorted(
        customers,
        key=lambda customer: (
            customer["demand"],
            customer["customer_id"],
        ),
    )

    remaining_stock = INITIAL_STOCK
    fulfilled_customers = []
    rejected_customers = []

    # Allocate inventory
    for customer in sorted_customers:
        customer_id = customer["customer_id"]
        demand = customer["demand"]
        if demand <= remaining_stock:
            fulfilled_customers.append(customer_id)
            remaining_stock -= demand
        else:
            rejected_customers.append(customer_id)

    fulfilled_demand = sum(
        customer["demand"]
        for customer in customers
        if customer["customer_id"] in fulfilled_customers
    )

    return {
        "fulfilled_customers": fulfilled_customers,
        "rejected_customers": rejected_customers,
        "fulfilled_demand": fulfilled_demand,
        "remaining_stock": remaining_stock,
    }


if __name__ == "__main__":
    result = greedy_inventory_allocation()
    fulfilled_customers = result["fulfilled_customers"]
    rejected_customers = result["rejected_customers"]
    fulfilled_demand = result["fulfilled_demand"]
    remaining_stock = result["remaining_stock"]

    print("GREEDY INVENTORY ALLOCATION")
    print("---------------------------")
    print("Customers fulfilled:", len(fulfilled_customers))
    print("Customers rejected:", len(rejected_customers))
    print("Total fulfilled demand:", fulfilled_demand)
    print("Remaining warehouse stock:", remaining_stock)
    print("Fulfilled customer IDs:", fulfilled_customers)
    print("Rejected customer IDs:", rejected_customers)

    print("\nINDEPENDENT INVENTORY VALIDATION")
    print("---------------------------------")
    print(
        "Allocation within initial stock:",
        fulfilled_demand <= INITIAL_STOCK,
    )
    print(
        "Stock accounting valid:",
        fulfilled_demand + remaining_stock == INITIAL_STOCK,
    )
    print(
        "No customer appears in both groups:",
        not set(fulfilled_customers) & set(rejected_customers),
    )
    print(
        "All customers classified:",
        len(fulfilled_customers) + len(rejected_customers) == 100,
    )
