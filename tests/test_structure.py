import numpy as np
import pandas as pd
from scipy import stats

from better_regressions.structure import entropy, entropy1, entropy2, mi_quantile
from better_regressions.structure import sk_mi as mi_knn


def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generate synthetic dataset with known factor structure for testing MI and factor analysis.

    Returns:
        pd.DataFrame with features and target
    """
    np.random.seed(random_state)

    # Factor 1: Market condition (heavy-tailed)
    market_factor = stats.t.rvs(df=3, size=n_samples)
    # Factor 2: Momentum
    momentum_factor = np.random.randn(n_samples)
    # Factor 3: Volatility (chi-squared for positive values)
    volatility_factor = np.sqrt(stats.chi2.rvs(df=5, size=n_samples))
    # Factor 4: Value (mixture of normals)
    value_factor = np.where(np.random.rand(n_samples) > 0.3, np.random.randn(n_samples), np.random.randn(n_samples) * 3 + 2)
    # Factor 5: Size (log-normal)
    size_factor = np.random.lognormal(0, 0.5, n_samples)
    # Factor 6: Quality (beta distribution)
    quality_factor = stats.beta.rvs(a=2, b=5, size=n_samples)

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
    data = generate_synthetic_data(n_samples=100000)
    print(data.head())
    data = data.to_numpy()[:, :6]
    n, d = data.shape

    for i in range(d):
        series = data[:, i]
        print(f"series {i}: mean {series.mean():.2f}, std {series.std():.2f}")
        print("H =", entropy(data[:, i : i + 1]))
        print("H1 =", entropy1(series))
        print("H1_4 =", entropy1(series, q=4))
        print("H2 =", entropy2(series, series))
        print("I =", mi_quantile(series, series))
        print("I_knn =", mi_knn(series, series))
        print()


if __name__ == "__main__":
    test_mi_simple()
