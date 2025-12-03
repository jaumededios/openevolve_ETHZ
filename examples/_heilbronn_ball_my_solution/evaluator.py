"""
Evaluator for the function minimization example
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import sys
import os
import base64
import io
from openevolve.evaluation_result import EvaluationResult
from itertools import combinations

import matplotlib

# Use non-interactive backend for worker threads
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon




def run_with_timeout(program_path: str, args=(), kwargs={}, timeout_seconds=5):

    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    sys.path.insert(0, program_dir)
    program = __import__(module_name)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(program.main, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            end_time = time.time()
            eval_time = end_time - start_time
            return result,eval_time
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")



def min_triangle_area(x: np.ndarray) -> float:
    """
    Takes an n x 2 ndarray of floats and returns the area of the smallest
    triangle formed by any 3 distinct points in the list.
    """
    x = np.asarray(x, dtype=float)

    n = len(x)
    if n < 3:
        raise ValueError("Need at least three points to form a triangle.")
    
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

def assert_points_ok(x:np.ndarray,expected_len,TOL = 1E-6):
    """Takes a n * 2 ndarray of floats and checks that they are all on the unit ball"""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("Input contains NaNs.")
    if expected_len is not None and x.shape != (expected_len,2):
        raise ValueError(f"Input must be an array of shape ({expected_len}, 2).")
    if np.max(x[:,0]**2+x[:,1]**2)>1+TOL:
        raise ValueError("All points must have norm <1")


def plot_points(points: np.ndarray, min_area: float, witness: tuple) -> str:
    """
    Plot points with the witness triangle highlighted and return the PNG as base64.
    """
    pts = np.asarray(points, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pts[:, 0], pts[:, 1], color="tab:blue", s=20, zorder=2)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=1, linestyle="--", zorder=1)

    if witness is not None and len(witness) == 3:
        tri = np.vstack(witness + (witness[0],))
        ax.plot(tri[:, 0], tri[:, 1], color="tab:red", linewidth=1.5, zorder=3)
        ax.scatter(tri[:3, 0], tri[:3, 1], color="tab:red", s=30, zorder=4)

    ax.set_title(f"Min triangle area: {min_area:.3e}")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    buf = io.BytesIO()
    fig.tight_layout(pad=0.1)
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


    

def evaluate(program_path):
    try:
        points,t = run_with_timeout(program_path=program_path)
        assert_points_ok(points, expected_len=None)
        area,witness = min_triangle_area(points)
        # Convert to scores (higher is better)
        return EvaluationResult(
                metrics={
                    "area_score": area,
                    "eval_tine": t,
                    "combined_score":  area/0.05
                },
                artifacts={
                    "suggestion": "The area is very small, there are probably three collinear points." if area<1E-4 else "",
                    "base64plot": plot_points(points, area, witness)
                }
            )
    except Exception as e:
        return EvaluationResult(
        metrics={
            "area_score": 0.0,
            "time_score": -10.0,
            "combined_score": 0.0,
            "error": "Trial failed",
            "error_type": type(e).__name__+str(e),
        },
        artifacts=  {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors, output format or missing imports in the generated code"
        }
        )
