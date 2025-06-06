"""Better Regressions - Advanced regression methods with sklearn-like interface."""

from better_regressions.binned import AutoBinnedRegression, BinnedRegression
from better_regressions.eda import (
    plot_distribution,
    plot_trend,
    plot_trend_continuous,
    plot_trend_discrete,
)
from better_regressions.linear import AdaptiveLinear, Linear, Soft
from better_regressions.piecewise import Angle
from better_regressions.scaling import (
    AutoScaler,
    PowerTransformer,
    QuantileTransformer,
    Scaler,
    SecondMomentScaler,
    Stabilize,
)
from better_regressions.smoothing import Smooth
from better_regressions.utils import Silencer


def auto_angle(n_breakpoints: int = 1, max_epochs: int = 200, lr: float = 0.5):
    return AutoScaler(Smooth(method="angle", n_breakpoints=n_breakpoints, max_epochs=max_epochs, lr=lr))


def auto_linear(alpha: float | str = "bayes"):
    return AutoScaler(Linear(alpha=alpha))


__all__ = [
    "Linear",
    "AdaptiveLinear",
    "Soft",
    "Angle",
    "Scaler",
    "AutoScaler",
    "SecondMomentScaler",
    "PowerTransformer",
    "QuantileTransformer",
    "Stabilize",
    "Smooth",
    "BinnedRegression",
    "AutoBinnedRegression",
    "Silencer",
    "auto_angle",
    "auto_linear",
    "plot_distribution",
    "plot_trend",
    "plot_trend_continuous",
    "plot_trend_discrete",
]

__version__ = "0.7.0"
