"""Better Regressions - Advanced regression methods with sklearn-like interface."""

__version__ = "0.1.0"

from better_regressions.linear import Linear
from better_regressions.piecewise import Angle
from better_regressions.scaling import Scaler
from better_regressions.smoothing import Smooth
from better_regressions.utils import Silencer

__all__ = ["Linear", "Angle", "Scaler", "Smooth", "Silencer"]
