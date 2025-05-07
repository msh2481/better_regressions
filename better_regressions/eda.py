from collections import defaultdict

import numpy as np
import seaborn as sns
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy import ndarray as ND
from scipy.stats import t as t_student
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from sklearn.linear_model import ARDRegression, BayesianRidge, LogisticRegression, Ridge
from sklearn.preprocessing import PowerTransformer

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from better_regressions.smoothing import Smooth


@typed
def plot_distribution(samples: Float[ND, "n_samples"]):
    min_value = np.min(samples)
    max_value = np.max(samples)
    mean = np.mean(samples)
    std = np.std(samples)
    df, loc, scale = t_student.fit(samples)
    plt.figure(figsize=(10, 6))
    plt.title(f"$\\mu={mean:.2f}$, $\\sigma={std:.2f}$ | $\\mu_t={loc:.2f}$, $\\sigma_t={scale:.2f}$, $\\nu={df:.2f}$\nrange: {min_value:.2f} to {max_value:.2f}")
    ql, qr = np.percentile(samples, [2, 98])
    samples = np.clip(samples, ql, qr)
    sns.histplot(
        samples,
        bins=100,
        kde=True,
        stat="density",
        kde_kws={"bw_adjust": 0.5},
        line_kws={"linewidth": 2, "color": "r"},
    )


@typed
def plot_trend(x: Float[ND, "n_samples"], y: Float[ND, "n_samples"], discrete_threshold: int = 50):
    if len(np.unique(x)) < discrete_threshold:
        plot_trend_discrete(x, y)
    else:
        plot_trend_continuous(x, y)


@typed
def extract_clusters(x: Float[ND, "n_samples"], max_clusters: int = 10) -> tuple[Float[ND, "n_clusters"], Int[ND, "n_samples"]]:
    x_2d = x.reshape(-1, 1)
    n_unique = len(np.unique(x))
    if n_unique == 1:
        return x[:1], np.zeros(len(x), dtype=int)
    n_clusters = min(n_unique, max_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(x_2d)
    clusters = kmeans.cluster_centers_.flatten()
    return clusters, labels


@typed
def prettify_sample(sample: Float[ND, "n_samples"]) -> Float[ND, "n_samples"]:
    """
    Applies PowerTransformer, inside fits and resamples via t-student distribution
    """
    pt = PowerTransformer()
    prepared = pt.fit_transform(sample.reshape(-1, 1)).flatten()
    df, loc, scale = t_student.fit(prepared)
    resampled = t_student.ppf(np.linspace(0.02, 0.98, 200, endpoint=True), df, loc, scale)
    return pt.inverse_transform(resampled.reshape(-1, 1)).flatten()


@typed
def plot_trend_discrete(x: Float[ND, "n_samples"], y: Float[ND, "n_samples"]):
    clusters, labels = extract_clusters(x)
    by_label = defaultdict(list)
    for value, label in zip(y, labels):
        by_label[label].append(value)

    new_x = []
    new_y = []
    for label, cluster in enumerate(clusters):
        samples = np.array(by_label[label])
        prettified = prettify_sample(samples)
        new_x.extend([cluster] * len(prettified))
        new_y.extend(prettified)
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    sns.violinplot(
        x=new_x,
        y=new_y,
        formatter=lambda x: f"{x:.2f}",
        inner="quart",
        fill=False,
    )
    plt.show()


@typed
def plot_trend_continuous(x: Float[ND, "n_samples"], y: Float[ND, "n_samples"]):
    pass


def test_plots():
    np.random.seed(42)
    x_discrete = np.concatenate([np.ones(50) * 1, np.ones(70) * 3, np.ones(40) * 5, np.ones(60) * 7, np.ones(30) * 9])
    y_values = np.concatenate([np.random.normal(10, 2, 50), np.random.standard_t(5, 70) * 2 + 15, np.random.gamma(2, 2, 40) + 5, np.random.normal(20, 4, 60), np.random.standard_cauchy(30) * 0.5 + 25])  # cluster 1: normal  # cluster 3: t-distribution  # cluster 5: gamma  # cluster 7: normal  # cluster 9: cauchy
    # x_discrete += np.random.normal(0, 0.1, len(x_discrete))

    plot_trend_discrete(x_discrete, y_values)
    plt.show()


if __name__ == "__main__":
    test_plots()
