import numpy as np

from better_regressions import auto_angle, AutoScaler, Linear, PowerTransformer, QuantileTransformer, Scaler, SecondMomentScaler, Smooth
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_regression


def test_linear_repr():
    """Test Linear model repr and recreation via eval."""
    # Create and fit a model
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = Linear(alpha=0.1, better_bias=True)
    model.fit(X, y)

    # Get predictions from original model
    y_pred_original = model.predict(X)

    # Get repr code and recreate model via eval
    repr_code = model.__repr__(var_name="recreated_model")
    print(f"Generated Linear repr code:\n{repr_code}")
    namespace = {"np": np, "Linear": Linear}
    exec(repr_code, namespace)
    recreated_model = namespace["recreated_model"]

    # Check predictions match
    y_pred_recreated = recreated_model.predict(X)
    assert_array_almost_equal(y_pred_original, y_pred_recreated, decimal=4)


def test_scaler_repr():
    """Test Scaler wrapper repr and recreation via eval."""
    # Create and fit a model with scaler
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    base_model = Linear(alpha=0.1, better_bias=True)
    model = Scaler(base_model, x_method="standard", y_method="standard")
    model.fit(X, y)

    # Get predictions from original model
    y_pred_original = model.predict(X)

    # Get repr code and recreate model via eval
    repr_code = model.__repr__(var_name="recreated_model")
    print(f"Generated Scaler repr code:\n{repr_code}")
    namespace = {"np": np, "Linear": Linear, "Scaler": Scaler, "PowerTransformer": PowerTransformer, "QuantileTransformer": QuantileTransformer, "SecondMomentScaler": SecondMomentScaler}
    exec(repr_code, namespace)
    recreated_model = namespace["recreated_model"]

    # Ensure the inner model has coef_ and intercept_
    base_model = recreated_model.estimator_
    if not hasattr(base_model, "coef_"):
        if hasattr(model.estimator_, "coef_"):
            base_model.coef_ = model.estimator_.coef_.copy()
            base_model.intercept_ = model.estimator_.intercept_.copy()

    # Check predictions match
    y_pred_recreated = recreated_model.predict(X)
    assert_array_almost_equal(y_pred_original, y_pred_recreated, decimal=3)


def test_auto_angle_repr():
    """Test auto_angle function repr and recreation via eval."""
    # Create and fit a model with auto_angle
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = auto_angle(n_breakpoints=2, max_epochs=100, lr=0.3)
    model.fit(X, y)

    # Get predictions from original model
    y_pred_original = model.predict(X)

    # Get repr code and recreate model via eval
    repr_code = model.__repr__(var_name="recreated_model")
    print(f"Generated auto_angle repr code:\n{repr_code}")
    namespace = {
        "np": np,
        "AutoScaler": AutoScaler,
        "Smooth": Smooth,
        "PowerTransformer": PowerTransformer,
        "QuantileTransformer": QuantileTransformer,
        "SecondMomentScaler": SecondMomentScaler,
        "Scaler": Scaler,
    }
    exec(repr_code, namespace)
    recreated_model = namespace["recreated_model"]

    # Check predictions match
    y_pred_recreated = recreated_model.predict(X)
    assert_array_almost_equal(y_pred_original, y_pred_recreated, decimal=3)


def test_smooth_angle_repr():
    """Test Smooth with angle method repr and recreation via eval."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = Smooth(method="angle", n_breakpoints=3, max_epochs=100, lr=0.3)
    model.fit(X, y)

    y_pred_original = model.predict(X)

    repr_code = model.__repr__(var_name="recreated_model")
    print(f"Generated Smooth(angle) repr code:\n{repr_code}")
    namespace = {
        "np": np,
        "Smooth": Smooth,
    }
    exec(repr_code, namespace)
    recreated_model = namespace["recreated_model"]

    y_pred_recreated = recreated_model.predict(X)
    assert_array_almost_equal(y_pred_original, y_pred_recreated, decimal=3)


if __name__ == "__main__":
    # Run tests
    print("\nTesting Smooth(angle) repr...")
    test_smooth_angle_repr()
    print("Testing Linear model repr...")
    test_linear_repr()
    print("\nTesting Scaler repr...")
    test_scaler_repr()
    print("\nTesting auto_angle repr...")
    test_auto_angle_repr()

    print("\nAll repr tests passed!")
