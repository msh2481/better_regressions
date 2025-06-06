# Better Regressions

Advanced regression methods with an sklearn-like interface.

## Current Features

- `Linear`:
  - Configurable regularization: Ridge with given `alpha` / BayesianRidge / ARD
  - "Better bias" option to properly regularize the intercept term
- `AdaptiveLinear`: Ridge regression with automatic shrinkage of features (like `ARDRegression`, but works in a different way and works better with correlated features)
- `Scaler`:
  - Configurable preprocessing: Standard scaling (by second moment) / Quantile transformation with uniform/normal output / Power transformation
  - `AutoScaler` to automatically select the best scaling method based on validation split
- `Smooth`: Boosting-based regression using smooth functions for features
  - `SuperSmoother`: Adaptive-span smoother for arbitrary complex functions.
  - `Angle`: Bagging of piecewise-linear functions, it's less flexible but because of that it's more robust to overfitting.
- `Soft`: Mixture of regressors based on quantile classification
- `Stabilize`: Robust scaling & clipping transformation for features/targets
- `AutoClassifier`: Classification with automatic model selection (LogisticRegression or XGBoost, with auto depth selection)
- `BinnedRegression`: Bins features and target, then trains a classifier. This way it can learn non-linear relationships and it also models the target distribution (not only its mean).
- `EDA`: Exploratory Data Analysis utilities
  - `plot_distribution`: Visualize sample distributions with fitted t-distribution parameters
  - `plot_trend`: Automatically detect and visualize relationships between variables + Pearson/Spearman correlation
    - For discrete features: Shows violin plots with distribution at each value
    - For continuous features: Fits trend lines with variance estimation and confidence intervals

## Installation

```bash
pip install better-regressions
```

## Basic Usage

```python
from better_regressions import auto_angle, auto_linear, Linear, Scaler, AutoClassifier
from better_regressions.eda import plot_distribution, plot_trend
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

# EDA example
plot_distribution(y, name="Target Distribution")
plot_trend(X[:, 0], y, name="Feature 0 vs Target")
```
