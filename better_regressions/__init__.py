"""Better Regressions - Advanced regression methods with sklearn-like interface."""

__version__ = "0.1.0"

from better_regressions.linear import Linear
from better_regressions.piecewise import Angle
from better_regressions.scaling import AutoScaler, PowerTransformer, QuantileTransformer, Scaler, SecondMomentScaler
from better_regressions.smoothing import Smooth
from better_regressions.utils import Silencer


def auto_angle(n_breakpoints: int = 1, max_epochs: int = 200, lr: float = 0.5):
    return AutoScaler(Smooth(method="angle", n_breakpoints=n_breakpoints, max_epochs=max_epochs, lr=lr))


def auto_linear(alpha: float | str = "bayes"):
    return AutoScaler(Linear(alpha=alpha))


__all__ = ["Linear", "Angle", "Scaler", "AutoScaler", "Smooth", "Silencer"]
