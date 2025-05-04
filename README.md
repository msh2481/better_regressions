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
- `Soft`: Mixture of regressors based on quantile classification
- `Stabilize`: Robust scaling & clipping transformation for features/targets
- `AutoClassifier`: Classification with automatic model selection (LogisticRegression or XGBoost, with auto depth selection)

## Installation

```bash
pip install better-regressions
```

## Basic Usage

```python
from better_regressions import auto_angle, auto_linear, Linear, Scaler, AutoClassifier
from sklearn.datasets import make_regression, make_moons
import numpy as np

X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
model = auto_angle(n_breakpoints=2)
model.fit(X, y)
y_pred = model.predict(X)
print(repr(model))

# Classification example
dataset = make_moons(n_samples=200, noise=0.3)
Xc, yc = dataset
clf = AutoClassifier(depth="auto")
clf.fit(Xc, yc)
yc_pred = clf.predict(Xc)
```
