import numpy as np
from sklearn.metrics import mean_squared_error

from better_regressions.binned import BinnedRegression


def test_binned_modes():
    np.random.seed(42)
    N = 10000

    # Test cases
    test_cases = [
        ("XOR", lambda a, b: (a.astype(int) ^ b.astype(int)).astype(float)),
        ("Weighted sum", lambda a, b: 2 * a + b),
        ("Simple sum", lambda a, b: a + b),
    ]

    modes = ["concat", "outer"]

    print("Testing BinnedRegression with different modes on edge cases")
    print("=" * 60)

    for case_name, y_func in test_cases:
        print(f"\n=== {case_name} ===")

        # Generate independent binary variables
        a = np.random.binomial(1, 0.5, N).astype(float)
        b = np.random.binomial(1, 0.5, N).astype(float)
        X = np.column_stack([a, b])
        y = y_func(a, b)

        print(f"Target distribution: {np.unique(y, return_counts=True)}")

        # Split data
        split_idx = N // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        results = {}

        for mode in modes:
            binned = BinnedRegression(X_bins=4, y_bins=4, mode=mode)
            binned.fit(X_train, y_train)

            # Calculate metrics
            log_loss = -binned.logpdf(X_test, y_test) / len(y_test)
            predictions = binned.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            # Calculate accuracy for discrete cases
            if case_name in ["XOR", "Weighted sum", "Simple sum"]:
                rounded_preds = np.round(predictions)
                accuracy = np.mean(rounded_preds == y_test)
            else:
                accuracy = None

            results[mode] = {"log_loss": log_loss, "mse": mse, "accuracy": accuracy, "predictions": predictions[:10]}

            print(f"\n{mode.upper()} mode:")
            print(f"  Log loss: {log_loss:.4f}")
            print(f"  MSE: {mse:.4f}")
            if accuracy is not None:
                print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Sample predictions: {predictions[:5]}")
            print(f"  Sample actual:      {y_test[:5]}")

        # Compare modes
        print(f"\nComparison:")
        if results["outer"]["log_loss"] < results["concat"]["log_loss"]:
            print(f"  OUTER mode has better log loss ({results['outer']['log_loss']:.4f} vs {results['concat']['log_loss']:.4f})")
        else:
            print(f"  CONCAT mode has better log loss ({results['concat']['log_loss']:.4f} vs {results['outer']['log_loss']:.4f})")

        if results["outer"]["mse"] < results["concat"]["mse"]:
            print(f"  OUTER mode has better MSE ({results['outer']['mse']:.4f} vs {results['concat']['mse']:.4f})")
        else:
            print(f"  CONCAT mode has better MSE ({results['concat']['mse']:.4f} vs {results['outer']['mse']:.4f})")


if __name__ == "__main__":
    test_binned_modes()
