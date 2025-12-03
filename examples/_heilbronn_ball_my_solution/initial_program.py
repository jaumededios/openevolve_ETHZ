# EVOLVE-BLOCK-START 

"""HeilBronn maximizer for OpenEvolve"""
import numpy as np
from itertools import combinations
from scipy.optimize import minimize


def min_triangle_area(x: np.ndarray) -> float:
    """
    Takes an n x 2 ndarray of floats and returns the area of the smallest
    triangle formed by any 3 distinct points in the list.
    """
    x = np.asarray(x, dtype=float)

    n = len(x)
    if n < 3:
        raise ValueError(f"Need at least three points to form a triangle but got {n} points.")
    
    min_area = np.inf
    witness = (1,1,1)
    
    for i, j, k in combinations(range(n), 3):
        a, b, c = x[i], x[j], x[k]
        # Area of triangle = 0.5 * |cross(b - a, c - a)| in 2D
        area = 0.5 * abs(np.cross(b - a, c - a))
        if area < min_area:
            min_area = area
            witness = (a,b,c)
    
    return (float(min_area),witness)

def optimize_helbronn_disk(N) -> np.ndarray:
    """
    Construct an arrangement of N points on or inside the unit ball in order to maximize the area of the
    smallest triangle formed by these points. For us N = 11.

    Returns:
        points: np.ndarray of shape (N,2) with the x,y coordinates of the points.
    """
    points = np.zeros((N, 2))
    return points


# EVOLVE-BLOCK-END



# This part remains fixed (not evolved)



NUM_POINTS = 11

def main():
    points= optimize_helbronn_disk(N = NUM_POINTS)
    return points


if __name__ == "__main__":
    x = main()
    area = min_triangle_area(x)
    OK = (np.max(x[:,0]**2+x[:,1]**2)<1+1E-6)& (np.array(x).shape == (NUM_POINTS,2))
    print(f"Found maximum area of value {area}")
    if not OK:
        print("But the given points are not valid")
