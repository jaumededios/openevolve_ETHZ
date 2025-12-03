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
from openevolve.evaluation_result import EvaluationResult




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



def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def evaluate(program_path):
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.704
    GLOBAL_MIN_Y = 0.678
    GLOBAL_MIN_VALUE = -1.519
    try:
        (x,y),t = run_with_timeout(program_path=program_path)
        distance = ((x-GLOBAL_MIN_X)**2+(y-GLOBAL_MIN_Y)**2)**.5
        loss = evaluate_function(x,y)-GLOBAL_MIN_VALUE
        # Convert to scores (higher is better)
        return EvaluationResult(
                metrics={
                    "value_score": -loss,
                    "distance_score": -distance,
                    "combined_score": 6-loss-distance
                },
                artifacts={}
            )
    except Exception as e:
        return EvaluationResult(
        metrics={
            "value_score": -10.0,
            "distance_score": -10.0,
            "combined_score": 0.0,
            "error": "Trial failed",
            "error_type": type(e).__name__+str(e),
        },
        artifacts=  {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code"
        }
        )

