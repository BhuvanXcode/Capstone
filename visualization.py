
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "c101.txt"
RESULTS_FILE = BASE_DIR / "outputs" / "ortools_routes.json"
IMAGE_FILE = BASE_DIR / "outputs" / "classical_routes.png"


def load_customers(file_path):
    """Load customer coordinates from a Solomon VRPTW dataset."""
    customers = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) == 7 and parts[0].isdigit():
                customer_id = int(parts[0])
                customers[customer_id] = {
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                }

    if not customers:
        raise ValueError(f"No customer coordinates found in {file_path}")

    return customers


def visualize_routes():
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    customers = load_customers(DATA_FILE)
    routes = results["routes"]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot customer locations
    xs = [point["x"] for point in customers.values()]
    ys = [point["y"] for point in customers.values()]

    ax.scatter(
        xs, ys,
        s=24,
        color="steelblue",
        label="Customers",
        zorder=3,
    )

    # Plot depot (customer ID 0)
    if 0 not in customers:
        raise ValueError("Depot ID 0 was not found in the dataset.")

    depot = customers[0]
    ax.scatter(
        depot["x"], depot["y"],
        marker="*",
        s=260,
        color="red",
        edgecolors="black",
        label="Depot",
        zorder=5,
    )

    # Draw each vehicle route
    color_map = plt.get_cmap("tab20")

    for index, vehicle in enumerate(routes):
        route = vehicle["route"]

        try:
            route_x = [customers[node]["x"] for node in route]
            route_y = [customers[node]["y"] for node in route]
        except KeyError as error:
            raise ValueError(
                f"Route contains customer ID {error.args[0]}, "
                "which is missing from the dataset."
            ) from error

        ax.plot(
            route_x,
            route_y,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color_map(index % 20),
            label=f"Vehicle {vehicle['vehicle']} "
                  f"(distance {vehicle['distance']})",
            alpha=0.85,
        )

    ax.set_title(
        f"Classical VRPTW Solution — {results['dataset']}\n"
        f"{results['vehicles_used']} vehicles | "
        f"Total distance: {results['total_distance']}"
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )

    fig.tight_layout()
    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_FILE, dpi=180, bbox_inches="tight")

    print(f"Saved route visualization to: {IMAGE_FILE}")
    plt.show()


if __name__ == "__main__":
    visualize_routes()
