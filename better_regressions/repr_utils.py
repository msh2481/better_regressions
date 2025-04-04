import numpy as np
from numpy import ndarray as ND


def format_array(arr: ND | float | int, precision: int = 6) -> str:
    """Format numpy array with minimal precision needed."""
    if not isinstance(arr, np.ndarray):
        # Handle scalars or other types that might sneak in
        if isinstance(arr, (int, float)):
            return str(round(arr, precision))
        raise ValueError(f"Unsupported type: {type(arr)}")

    if arr.size == 0:
        return "np.array([])"
    # Format with specified precision for all numbers including scientific notation
    with np.printoptions(
        precision=precision,
        suppress=True,
        floatmode="maxprec",
        linewidth=np.inf,
        threshold=np.inf,
    ):
        arr_str = "np." + repr(arr)
    return arr_str
