import contextlib
import os

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


class Silencer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.null_file = None
        self.stdout_redirector = None
        self.stderr_redirector = None

    def __enter__(self):
        if not self.verbose:
            self.null_file = open(os.devnull, "w")
            self.stdout_redirector = contextlib.redirect_stdout(self.null_file)
            self.stderr_redirector = contextlib.redirect_stderr(self.null_file)
            self.stdout_redirector.__enter__()
            self.stderr_redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            self.stderr_redirector.__exit__(exc_type, exc_val, exc_tb)
            self.stdout_redirector.__exit__(exc_type, exc_val, exc_tb)
            self.null_file.close()
