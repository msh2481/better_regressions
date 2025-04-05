# Better Regressions

Advanced regression methods with an sklearn-like interface.

## Current Features

- Linear regression with configurable regularization:
  - Ridge regression with alpha parameter
  - Automatic Relevance Determination (ARD) regression
  - Bayesian Ridge regression
  - "Better bias" option to properly regularize the intercept term
- Input/target scaling wrappers:
  - Standard scaling (based on second moment)
  - Quantile transformation with uniform output
  - Quantile transformation with normal output
  - Power transformation
  - AutoScaler to automatically select the best scaling method
- Smoothing-based regression:
  - Boosting-based regression using smooth functions for features
  - Two smoothing methods: SuperSmoother and piecewise-linear (Angle)

## Installation

```bash
pip install better-regressions
```

With uv:
```bash
uv pip install better-regressions
```

## Basic Usage

```python
import numpy as np
from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# Ridge regression with better bias handling
model = Linear(alpha=1e-6, better_bias=True)

# Wrap model with standard scaling for both inputs and targets
scaled_model = Scaler(model, x_method="standard", y_method="standard")
scaled_model.fit(X, y)
predictions = scaled_model.predict(X)

# Access model parameters
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# ARD regression with quantile-normal scaling
ard_model = Linear(alpha="ard", better_bias=True)
ard_scaled = Scaler(ard_model, x_method="quantile-normal", y_method="standard")
ard_scaled.fit(X, y)
```
