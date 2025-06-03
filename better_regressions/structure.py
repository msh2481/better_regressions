import numpy as np
import pandas as pd
from beartype import beartype as typed
from jaxtyping import Float
from numpy import ndarray as ND
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

from better_regressions.entropy_estimators import entropy, mi


@typed
def sk_mi(x: Float[ND, "n"], y: Float[ND, "n"]) -> float:
    result = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3)
    assert result.shape == (1,)
    return float(result[0])


@typed
def entropy1(x: Float[ND, "n"], q: int = 10) -> float:
    x_clipped = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
    boundaries = np.quantile(x_clipped, np.linspace(0, 1, q + 1))
    counts = np.histogram(x_clipped, bins=boundaries)[0].astype(float)

    probabilities = counts / np.sum(counts)
    segment_widths = np.diff(boundaries)
    return -np.sum(probabilities * np.log2(probabilities / segment_widths + 1e-12))


@typed
def entropy2(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 4) -> float:
    x_clipped = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
    y_clipped = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

    x_boundaries = np.quantile(x_clipped, np.linspace(0, 1, q + 1))
    y_boundaries = np.quantile(y_clipped, np.linspace(0, 1, q + 1))

    counts, _, _ = np.histogram2d(x_clipped, y_clipped, bins=[x_boundaries, y_boundaries])
    probabilities = counts.flatten() / np.sum(counts)

    x_widths = np.diff(x_boundaries)
    y_widths = np.diff(y_boundaries)
    segment_areas = np.outer(x_widths, y_widths).flatten()

    return -np.sum(probabilities * np.log2(probabilities / segment_areas + 1e-12))


@typed
def entropy3(x: Float[ND, "n"], y: Float[ND, "n"], z: Float[ND, "n"], q: int = 4) -> float:
    x_clipped = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
    y_clipped = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
    z_clipped = np.clip(z, np.percentile(z, 1), np.percentile(z, 99))

    x_boundaries = np.quantile(x_clipped, np.linspace(0, 1, q + 1))
    y_boundaries = np.quantile(y_clipped, np.linspace(0, 1, q + 1))
    z_boundaries = np.quantile(z_clipped, np.linspace(0, 1, q + 1))

    counts, _ = np.histogramdd([x_clipped, y_clipped, z_clipped], bins=[x_boundaries, y_boundaries, z_boundaries])
    probabilities = counts.flatten() / np.sum(counts)

    x_widths = np.diff(x_boundaries)
    y_widths = np.diff(y_boundaries)
    z_widths = np.diff(z_boundaries)

    segment_volumes = np.zeros((q, q, q))
    for i in range(q):
        for j in range(q):
            for k in range(q):
                segment_volumes[i, j, k] = x_widths[i] * y_widths[j] * z_widths[k]
    segment_volumes = segment_volumes.flatten()

    return -np.sum(probabilities * np.log2(probabilities / segment_volumes + 1e-12))


@typed
def mi_quantile(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 4) -> float:
    """I(x, y) = H(x) + H(y) - H(x, y)"""
    return entropy1(x, q) + entropy1(y, q) - entropy2(x, y, q)


@typed
def cmi_quantile(x: Float[ND, "n"], y: Float[ND, "n"], z: Float[ND, "n"], q: int = 4) -> float:
    """I(x, y | z) = H(x, z) + H(y, z) - H(z) - H(x, y, z)"""
    return entropy2(x, z, q) + entropy2(y, z, q) - entropy1(z, q) - entropy3(x, y, z, q)


def test_entropy():
    x = np.random.randn(10000)
    y = np.random.randn(10000)
    z = np.random.randn(10000)
    print(f"H(x) = {entropy1(x)}")
    print(f"I(x, y) = {mi_quantile(x, y)}")
    print(f"I(x, y | z) = {cmi_quantile(x, y, z)}")


if __name__ == "__main__":
    test_entropy()
