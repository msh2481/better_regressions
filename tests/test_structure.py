import numpy as np
import pandas as pd
from scipy import stats

from better_regressions.structure import mi_knn as mi_knn
from better_regressions.structure import mi_quantile, show_structure


def generate_very_simple_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    a = np.random.randn(n_samples)
    b = a + np.random.randn(n_samples)
    return pd.DataFrame({"a": a, "b": b, "target": a + b})


def generate_simple_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    a = np.random.randn(n_samples)
    a /= a.std()
    b = np.random.randn(n_samples)
    b /= b.std()
    c = np.random.randn(n_samples)
    c /= c.std()
    d = np.random.randn(n_samples)
    d /= d.std()
    e = np.random.randn(n_samples)
    e /= e.std()
    f = np.random.randn(n_samples)
    f /= f.std()
    target = a * b + c * d + e * f + np.tanh(a + f)
    # target = a + b + c + d + e + f
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "target": target})


def generate_trading_like_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    # Factor 1: Market condition (heavy-tailed)
    market_factor = stats.t.rvs(df=3, size=n_samples)
    market_factor /= market_factor.std()
    # Factor 2: Momentum
    momentum_factor = np.random.randn(n_samples)
    momentum_factor /= momentum_factor.std()
    # Factor 3: Volatility (chi-squared for positive values)
    volatility_factor = np.sqrt(stats.chi2.rvs(df=5, size=n_samples))
    volatility_factor /= volatility_factor.std()
    # Factor 4: Value (mixture of normals)
    value_factor = np.where(np.random.rand(n_samples) > 0.3, np.random.randn(n_samples), np.random.randn(n_samples) * 3 + 2)
    value_factor /= value_factor.std()
    # Factor 5: Size (log-normal)
    size_factor = np.random.lognormal(0, 0.5, n_samples)
    size_factor /= size_factor.std()
    # Factor 6: Quality (beta distribution)
    quality_factor = stats.beta.rvs(a=2, b=5, size=n_samples)
    quality_factor /= quality_factor.std()

    features = {}
    # Linear combinations (what standard factor analysis finds)
    features["momentum_pure"] = momentum_factor + 0.1 * np.random.randn(n_samples)
    features["momentum_mixed"] = 0.7 * momentum_factor + 0.3 * market_factor + 0.1 * np.random.randn(n_samples)
    features["value_pure"] = value_factor + 0.1 * np.random.randn(n_samples)
    features["value_quality_mix"] = 0.6 * value_factor + 0.4 * quality_factor + 0.1 * np.random.randn(n_samples)
    # Non-linear transformations (challenging for factor analysis, good for MI)
    features["volatility_squared"] = volatility_factor**2 + 0.5 * np.random.randn(n_samples)
    features["volatility_log"] = np.log(volatility_factor + 1) + 0.1 * np.random.randn(n_samples)
    features["momentum_tanh"] = np.tanh(2 * momentum_factor) + 0.1 * np.random.randn(n_samples)
    features["size_sqrt"] = np.sqrt(size_factor) + 0.1 * np.random.randn(n_samples)
    # Interaction terms (factor analysis struggles, MI captures)
    features["momentum_vol_interaction"] = momentum_factor * volatility_factor + 0.2 * np.random.randn(n_samples)
    features["market_regime_value"] = market_factor * np.sign(value_factor) + 0.1 * np.random.randn(n_samples)
    # Step functions / discrete transforms
    features["quality_bucket"] = (quality_factor * 5).astype(int) + 0.1 * np.random.randn(n_samples)
    features["momentum_signal"] = np.where(momentum_factor > 0, 1, -1) + 0.1 * np.random.randn(n_samples)
    # Complex non-monotonic relationships
    features["market_sine"] = np.sin(2 * market_factor) + 0.1 * np.random.randn(n_samples)
    features["value_polynomial"] = value_factor - 0.5 * value_factor**2 + 0.1 * value_factor**3 + 0.1 * np.random.randn(n_samples)
    # Correlated noise (should be filtered out)
    noise_base = np.random.randn(n_samples)
    features["noise_1"] = noise_base + 0.5 * np.random.randn(n_samples)
    features["noise_2"] = 0.8 * noise_base + 0.6 * np.random.randn(n_samples)
    features["noise_3"] = -0.7 * noise_base + 0.7 * np.random.randn(n_samples)
    # Pure independent noise
    features["pure_noise_1"] = np.random.randn(n_samples)
    features["pure_noise_2"] = np.random.exponential(1, n_samples)
    features["pure_noise_3"] = np.random.uniform(-2, 2, n_samples)

    # Redundant features (high MI with existing features)
    features["momentum_redundant"] = features["momentum_pure"] + 0.05 * np.random.randn(n_samples)
    features["volatility_redundant"] = 2 * features["volatility_squared"] + 1 + 0.1 * np.random.randn(n_samples)

    target = (
        # Linear effects
        0.5 * momentum_factor
        + 0.3 * value_factor
        - 0.4 * volatility_factor
        +
        # Non-linear effects
        0.2 * momentum_factor**2
        - 0.3 * np.tanh(market_factor)
        + 0.2 * np.log(size_factor + 1)
        +
        # Interaction effects
        0.15 * momentum_factor * (volatility_factor > np.median(volatility_factor))
        - 0.1 * value_factor * quality_factor
        +
        # Regime-dependent effects
        0.3 * momentum_factor * (market_factor > 0)
        +
        # Noise
        0.5 * np.random.randn(n_samples)
    )
    df = pd.DataFrame(features)
    df["target"] = target
    # Add hidden factors for validation (prefix with _ to indicate they're hidden)
    df["_factor_market"] = market_factor
    df["_factor_momentum"] = momentum_factor
    df["_factor_volatility"] = volatility_factor
    df["_factor_value"] = value_factor
    df["_factor_size"] = size_factor
    df["_factor_quality"] = quality_factor
    # Reorder columns
    feature_cols = [col for col in df.columns if not col.startswith("_") and col != "target"]
    hidden_cols = [col for col in df.columns if col.startswith("_")]
    df = df[feature_cols + ["target"] + hidden_cols]
    return df


def test_mi_simple():
    data = generate_trading_like_data(n_samples=10000)
    print(data.head())
    data = data.to_numpy()[:, :6]
    n, d = data.shape

    mi_knns = np.zeros((d, d))
    mi_quantiles = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            # mi_knns[i, j] = mi_knn(data[:, i], data[:, j])
            mi_quantiles[i, j] = mi_quantile(data[:, i], data[:, j])

    np.set_printoptions(formatter={"float": lambda x: f"{x:.3f}"})
    print(mi_knns)
    print(mi_quantiles)


def test_structure():
    data = generate_trading_like_data(n_samples=2000)
    X = data.drop(columns=["target"])
    y = data["target"]
    show_structure(
        X,
        y,
        "output",
        do_regional_mi=False,
        do_structure_matrices=True,
        do_factor_analysis=False,
    )


def simple_test_structure():
    data = generate_simple_data(n_samples=10000)
    X = data.drop(columns=["target"])
    y = data["target"]
    show_structure(
        X,
        y,
        "output",
        do_regional_mi=False,
        do_structure_matrices=True,
        do_factor_analysis=False,
    )


def very_simple_test_structure():
    data = generate_very_simple_data(n_samples=10000)
    X = data.drop(columns=["target"])
    y = data["target"]
    show_structure(
        X,
        y,
        "output",
        do_regional_mi=True,
        do_structure_matrices=True,
        do_factor_analysis=True,
    )


if __name__ == "__main__":
    test_structure()
