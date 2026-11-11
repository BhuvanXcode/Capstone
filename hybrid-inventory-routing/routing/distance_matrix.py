import math

from customers import load_c101


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two locations.
    """
    dx = point1["x"] - point2["x"]
    dy = point1["y"] - point2["y"]
    return math.sqrt(dx ** 2 + dy ** 2)


def create_distance_matrix():
    warehouse, customers = load_c101()

    # Warehouse is the first location
    locations = [warehouse] + customers
    size = len(locations)
    distance_matrix = []

    for i in range(size):
        row = []
        for j in range(size):
            distance = calculate_distance(locations[i], locations[j])
            row.append(round(distance, 2))
        distance_matrix.append(row)

    return distance_matrix


if __name__ == "__main__":
    distance_matrix = create_distance_matrix()
    print("Distance Matrix Created Successfully")
    print(f"Total locations: {len(distance_matrix)}")
    print(
        f"Matrix size: {len(distance_matrix)} x {len(distance_matrix[0])}"
    )
