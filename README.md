# Better Regressions

Advanced regression methods with an sklearn-like interface.

## Current Features

- Linear regression with configurable regularization:
  - Ridge regression with alpha parameter
  - Automatic Relevance Determination (ARD) regression
  - "Better bias" option to properly regularize the intercept term
- Input/target scaling wrapper with multiple normalization methods:
  - Standard scaling (zero mean, unit variance)
  - Quantile transformation with uniform output
  - Quantile transformation with normal output

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

## Project Structure

```
better-regressions/
├── better_regressions/       # Main package
│   ├── __init__.py           # Package initialization
│   ├── linear.py             # Linear regression models
│   ├── scaling.py            # Data normalization wrappers
│   └── repr_utils.py         # Utility for model representations
└── tests/                    # Test directory
    └── test_linear.py        # Linear model tests
```

## Development

This project uses uv for dependency management:

```bash
# Clone the repository
git clone https://github.com/yourusername/better-regressions.git
cd better-regressions

# Create a virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Run tests
python -m tests.test_linear
```
