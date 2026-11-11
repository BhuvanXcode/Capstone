from pathlib import Path
import sys


# Allow importing the routing dataset parser
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTING_FOLDER = PROJECT_ROOT / "routing"
sys.path.append(str(ROUTING_FOLDER))

from customers import load_c101


# Experimental starting stock.
# This is a project assumption, not C101 dataset data.
INITIAL_STOCK = 1000


def load_inventory_data():
    warehouse, customers = load_c101()
    inventory_data = []

    for customer in customers:
        inventory_data.append(
            {
                "customer_id": customer["id"],
                "demand": customer["demand"],
                "fulfilled": False,
            }
        )

    return warehouse, inventory_data


if __name__ == "__main__":
    warehouse, inventory_data = load_inventory_data()
    total_demand = sum(item["demand"] for item in inventory_data)
    stock_shortage = max(total_demand - INITIAL_STOCK, 0)

    print("WAREHOUSE INVENTORY SUMMARY")
    print("----------------------------")
    print("Initial warehouse stock:", INITIAL_STOCK)
    print("Total customers:", len(inventory_data))
    print("Total customer demand:", total_demand)
    print("Available stock:", min(INITIAL_STOCK, total_demand))
    print("Stock shortage:", stock_shortage)
