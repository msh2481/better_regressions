# Better Regressions

Advanced regression methods with an sklearn-like interface.

## Current Features

- `Linear`:
  - Configurable regularization: Ridge with given `alpha` / BayesianRidge / ARD
  - "Better bias" option to properly regularize the intercept term
- `Scaler`:
  - Configurable preprocessing: Standard scaling (by second moment) / Quantile transformation with uniform/normal output / Power transformation
  - `AutoScaler` to automatically select the best scaling method based on validation split
- `Smooth`: Boosting-based regression using smooth functions for features
  - `SuperSmoother`: Adaptive-span smoother for arbitrary complex functions.
  - `Angle`: Bagging of piecewise-linear functions, it's less flexible but because of that it's more robust to overfitting.

## Installation

```bash
pip install better-regressions
```

## Basic Usage

```python
from better_regressions import auto_angle, auto_linear, Linear, Scaler
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
model = auto_angle(n_breakpoints=2)
model.fit(X, y)
y_pred = model.predict(X)
print(repr(model))
```
