from inventory_data import load_inventory_data


def allocate_inventory(stock, customers):
    sorted_customers = sorted(
        customers,
        key=lambda customer: (
            customer["demand"],
            customer["customer_id"],
        ),
    )
    remaining_stock = stock
    fulfilled = []

    for customer in sorted_customers:
        demand = customer["demand"]
        if demand <= remaining_stock:
            fulfilled.append(customer)
            remaining_stock -= demand

    return fulfilled, remaining_stock


def run_scenario_analysis():
    _, customers = load_inventory_data()
    stock_levels = [500, 750, 1000, 1250, 1500, 1810]

    print("INVENTORY SCENARIO ANALYSIS")
    print("---------------------------")

    for stock in stock_levels:
        fulfilled, remaining_stock = allocate_inventory(stock, customers)
        fulfilled_demand = sum(
            customer["demand"] for customer in fulfilled
        )
        print(f"\nStock level: {stock}")
        print("Customers fulfilled:", len(fulfilled))
        print("Customers rejected:", len(customers) - len(fulfilled))
        print("Fulfilled demand:", fulfilled_demand)
        print("Remaining stock:", remaining_stock)


if __name__ == "__main__":
    run_scenario_analysis()
