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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, t as t_student
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from sklearn.linear_model import ARDRegression, BayesianRidge, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PowerTransformer

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from better_regressions.smoothing import Smooth


@typed
def plot_distribution(samples: Float[ND, "n_samples"], name: str):
    min_value = np.min(samples)
    max_value = np.max(samples)
    mean = np.mean(samples)
    std = np.std(samples)
    df, loc, scale = t_student.fit(samples)
    plt.figure(figsize=(10, 6))
    title = f"$\\mu={mean:.2f}$, $\\sigma={std:.2f}$ | $\\mu_t={loc:.2f}$, $\\sigma_t={scale:.2f}$, $\\nu={df:.2f}$\nrange: {min_value:.2f} to {max_value:.2f}"
    if name:
        title = f"{name}\n{title}"
    plt.title(title)
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
def plot_trend(x: Float[ND, "n_samples"], y: Float[ND, "n_samples"], discrete_threshold: int = 50, name: str = None):
    if len(np.unique(x)) < discrete_threshold:
        plot_trend_discrete(x, y)
    else:
        plot_trend_continuous(x, y)

    pearson_corr = pearsonr(x, y)[0]
    spearman_corr = spearmanr(x, y)[0]
    title = f"Pearson: ${pearson_corr*100:.1f}\%$, Spearman: ${spearman_corr*100:.1f}\%$"
    if name:
        title = f"{name}\n{title}"
    plt.title(title)


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


@typed
def plot_trend_continuous(x: Float[ND, "n_samples"], y: Float[ND, "n_samples"]):
    argsort = np.argsort(x)
    x = x[argsort]
    y = y[argsort]
    ql, qr = np.percentile(x, [2, 98])
    x = np.clip(x, ql, qr)
    ql, qr = np.percentile(y, [2, 98])
    y = np.clip(y, ql, qr)

    model = Scaler(Smooth(n_breakpoints=2))
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    deviations = y - y_pred
    squared_devs = deviations**2
    y_var = gaussian_filter1d(squared_devs, sigma=0.2 * np.std(np.arange(len(x))))
    variance_model = KNeighborsRegressor(n_neighbors=max(5, len(x) // 5))
    variance_model.fit(x.reshape(-1, 1), y_var)

    x_range = np.linspace(np.min(x), np.max(x), 200)
    y_var = variance_model.predict(x_range.reshape(-1, 1))
    y_std = np.sqrt(y_var)
    y_trend = model.predict(x_range.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "ok", ms=4, alpha=0.2)
    plt.plot(x_range, y_trend, color="k", linewidth=4, alpha=0.5)
    plt.fill_between(x_range, y_trend - y_std, y_trend + y_std, alpha=0.2, color="k")


def test_plots():
    x = np.random.standard_cauchy(1000)
    noise = np.random.standard_cauchy(len(x)) * x * 0.05
    y = np.sin(x) + noise
    plot_trend(x, y, name="Sine with noise")
    plt.show()


if __name__ == "__main__":
    test_plots()
