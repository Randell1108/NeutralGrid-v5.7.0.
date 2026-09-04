"""
Core utility functions.

Common utilities used across the application.
"""

from typing import Any
import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy type, or any value)

    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
