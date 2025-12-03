# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5), prng=42):
    """
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    rng = (
        prng
        if isinstance(prng, np.random.Generator)
        else np.random.default_rng(prng)
    )

    # Initialize with a random point
    best_x = rng.uniform(bounds[0], bounds[1])
    best_y = rng.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Simple random search
        x = rng.uniform(bounds[0], bounds[1])
        y = rng.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)

print("WE IMPORTED THE RIGHT GUY")


def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def main():
    x, y, _ = search_algorithm()
    return x, y


if __name__ == "__main__":
    x, y = main()
    print(f"Found minimum at ({x}, {y}) with value {evaluate_function(x,y)}")
