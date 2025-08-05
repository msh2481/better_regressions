import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from tqdm import tqdm


class EMA(BaseEstimator, RegressorMixin):
    def __init__(self, regressor=None, span=1.0):
        if regressor is None:
            regressor = Ridge(alpha=1.0)
        if not (span >= 1.0):
            raise ValueError(f"span must be >= 1.0, got {span}")
        self.regressor = regressor
        self.span = span
        self.decay = 1 - 1.0 / span

    def fit(self, X, y, t=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D array")
        n_samples = X.shape[0]

        if t is None:
            t = np.arange(n_samples, dtype=float)
        else:
            t = np.asarray(t, dtype=float)
            if t.shape != (n_samples,):
                raise ValueError(f"t must have shape ({n_samples},), got {t.shape}")

        max_t = t.max()
        weights = self.decay ** (max_t - t)

        self.regressor.fit(X, y, sample_weight=weights)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    @property
    def coef_(self):
        return getattr(self.regressor, "coef_", None)

    @property
    def intercept_(self):
        return getattr(self.regressor, "intercept_", None)


class Roll(BaseEstimator, RegressorMixin):
    def __init__(self, regressor=None, span=1.0):
        if regressor is None:
            regressor = Ridge(alpha=1e-6)
        if not (span >= 1.0):
            raise ValueError(f"span must be >= 1.0, got {span}")
        self.regressor = regressor
        self.span = span

    def fit(self, X, y, t=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D array")
        n_samples = X.shape[0]

        if t is None:
            t = np.arange(n_samples, dtype=float)
        else:
            t = np.asarray(t, dtype=float)
            if t.shape != (n_samples,):
                raise ValueError(f"t must have shape ({n_samples},), got {t.shape}")

        t_max = t.max()
        valid_mask = t >= (t_max - self.span)

        if not np.any(valid_mask):
            raise ValueError(f"No samples within span {self.span} from t_max {t_max}")

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        self.regressor.fit(X_valid, y_valid)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    @property
    def coef_(self):
        return getattr(self.regressor, "coef_", None)

    @property
    def intercept_(self):
        return getattr(self.regressor, "intercept_", None)


def walk_forward_correlation(regressor, X, y, t=None, method: Literal["pearson", "spearman", "dcor"] = "pearson", batch_size=10):
    """Compute walk-forward correlation between predicted and actual values."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]

    if t is None:
        t = np.arange(n_samples, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
        if t.shape != (n_samples,):
            raise ValueError(f"t must have shape ({n_samples},), got {t.shape}")

    preds = []
    targets = []

    for i in range(batch_size, n_samples, batch_size):
        X_train = X[:i]
        y_train = y[:i]
        t_train = t[:i] if t is not None else None
        X_test = X[i : i + batch_size]
        y_test = y[i : i + batch_size]
        regressor.fit(X_train, y_train, t=t_train)
        y_pred = regressor.predict(X_test)
        preds.append(y_pred)
        targets.append(y_test)

    if not preds or len(preds) < 2:
        return 0.0

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if method == "pearson":
        corr, _ = pearsonr(preds, targets)
        return corr if not np.isnan(corr) else 0.0
    elif method == "spearman":
        corr, _ = spearmanr(preds, targets)
        return corr if not np.isnan(corr) else 0.0
    elif method == "dcor":
        return dcor.distance_correlation(preds, targets)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_span_correlations(args):
    """Helper function for parallel computation of correlations for a single span."""
    span, X, y, t, base_estimator, method, metrics, batch_size = args
    row = {"span": span}
    if method == "ema":
        regressor = EMA(regressor=copy.deepcopy(base_estimator), span=span)
    elif method == "roll":
        regressor = Roll(regressor=copy.deepcopy(base_estimator), span=span)
    else:
        raise ValueError(f"Unknown method: {method}")
    for metric in metrics:
        try:
            corr = walk_forward_correlation(regressor, X, y, t=t, method=metric, batch_size=batch_size)
            row[metric] = corr
        except Exception:
            row[metric] = 0.0
    return row


def plot_signal_decay(df, method, spans=None, name="-"):
    if spans is None:
        spans = df["span"].values
    plt.figure(figsize=(12, 6))
    colors = {"pearson": "blue", "spearman": "green", "dcor": "red"}
    metrics = [col for col in df.columns if col != "span"]
    for metric in metrics:
        if metric in df.columns:
            values = 100 * df[metric].values
            argmax = np.argmax(values)
            best_span = spans[argmax]
            best_corr = values[argmax]
            plt.plot(spans, values, lw=2, marker="o", markersize=4, color=colors.get(metric, "black"), label=f"{metric} (best: {best_span:.1f}, {best_corr:.2f})")
            plt.axvline(best_span, color=colors.get(metric, "black"), alpha=0.3, linestyle="--")
    plt.xlabel(f"{method.upper()} span")
    plt.xscale("log")
    plt.ylabel("Correlation")
    plt.title(f"Correlation vs span ({method.upper()} method) for {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def estimate_signal_decay(
    X,
    y,
    t=None,
    base_estimator=None,
    spans=None,
    method: Literal["ema", "roll"] = "ema",
    use_pearson=True,
    use_spearman=True,
    use_dcor=True,
    batch_size=10,
    show_plot=True,
    name="-",
    n_jobs=None,
):
    if base_estimator is None:
        base_estimator = Ridge(alpha=1e-6)
    if spans is None:
        spans = np.logspace(0, 3, 30)
    if n_jobs is None:
        import os

        n_jobs = os.cpu_count()

    metrics = []
    if use_pearson:
        metrics.append("pearson")
    if use_spearman:
        metrics.append("spearman")
    if use_dcor:
        metrics.append("dcor")

    args_list = [(span, X, y, t, base_estimator, method, metrics, batch_size) for span in spans]

    results = []

    if n_jobs == 1:
        # Sequential processing
        for args in tqdm(args_list, desc="Testing spans"):
            results.append(_compute_span_correlations(args))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_span = {executor.submit(_compute_span_correlations, args): args[0] for args in args_list}

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_span), total=len(spans), desc="Testing spans"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    span = future_to_span[future]
                    print(f"Error processing span {span}: {e}")
                    # Add fallback result
                    row = {"span": span}
                    for metric in metrics:
                        row[metric] = 0.0
                    results.append(row)

    # Sort results by span to maintain order
    results.sort(key=lambda x: x["span"])
    df = pd.DataFrame(results)

    if show_plot:
        plot_signal_decay(df, method, spans, name=name)
    return df


def test_ema_against_ridge():
    N = 1000
    X = np.random.randn(N, 1)
    y = X.dot(np.array([1.0])) + np.random.randn(N)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    print(f"Ridge coef: {ridge.coef_}")
    for span in [10, 100, 1000, 10000, 10**9]:
        ema = EMA(span=span)
        ema.fit(X, y)
        print(f"Span: {span}, EMA coef: {ema.coef_}")


def test_ema_as_ema():
    y = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0, 1.0, 2.0, 3, 4, 5])
    x = np.ones_like(y).reshape(-1, 1)
    span = 10
    for i in range(1, len(y)):
        ridge_no_intercept = Ridge(alpha=1e-6, fit_intercept=False)
        ema = EMA(regressor=ridge_no_intercept, span=span)
        ema.fit(x[:i], y[:i])
        print(f"Span: {i} | EMA coef: {ema.coef_} | EMA intercept: {ema.intercept_}")


def test_walk_forward_correlation():
    """Test the walk_forward_correlation function."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    random_signs = np.random.choice([-1, 1], size=n) + 0.1
    y = (X @ np.array([1, -0.5])) * random_signs + np.random.randn(n)

    ema = EMA(span=10)
    corr_pearson = walk_forward_correlation(ema, X, y, method="pearson")
    corr_spearman = walk_forward_correlation(ema, X, y, method="spearman")
    corr_dcor = walk_forward_correlation(ema, X, y, method="dcor")
    print(f"EMA correlations - Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f}, dcor: {corr_dcor:.3f}")
    roll = Roll(span=20)
    corr_pearson = walk_forward_correlation(roll, X, y, method="pearson")
    corr_spearman = walk_forward_correlation(roll, X, y, method="spearman")
    corr_dcor = walk_forward_correlation(roll, X, y, method="dcor")
    print(f"Roll correlations - Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f}, dcor: {corr_dcor:.3f}")


def test_estimate_signal_decay():
    np.random.seed(42)
    n = 1000
    d = 3
    xs = []
    ys = []
    w = np.random.randn(d)
    for i in range(n):
        if np.random.rand() < 0.01:
            w = np.random.randn(d)
        x = np.random.randn(d)
        xs.append(x)
        ys.append(x.dot(w) + np.random.randn())
    X = np.array(xs)
    y = np.array(ys)
    t = np.arange(n)

    df = estimate_signal_decay(
        X,
        y,
        t=t,
        method="roll",
        spans=np.logspace(0.5, 2.5, 10),
        use_pearson=True,
        use_spearman=True,
        use_dcor=True,
        batch_size=5,
        show_plot=True,
        name="X",
        n_jobs=2,  # Use 2 processes for testing
    )
    print(df.to_string())


if __name__ == "__main__":
    test_estimate_signal_decay()
