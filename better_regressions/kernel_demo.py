from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline

from better_regressions.kernel import SupervisedNystroem


def demo_moons_pca(random_state: int = 0) -> None:
    X, y = make_moons(n_samples=400, noise=0.2, random_state=random_state)
    pipeline = make_pipeline(
        SupervisedNystroem(
            forest_kind="rf",
            regression=False,
            n_estimators=200,
            min_samples_leaf=0.1,
            n_components=200,
            random_state=random_state,
        ),
        PCA(n_components=2),
    )
    embedding = pipeline.fit_transform(X, y)
    plt.figure(figsize=(4, 4))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=20)
    plt.title("SupNys + PCA on Moons")
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def demo_moons_kmeans(random_state: int = 1) -> None:
    X, y = make_moons(n_samples=400, noise=0.25, random_state=random_state)
    pipeline = make_pipeline(
        SupervisedNystroem(
            forest_kind="et",
            regression=False,
            n_estimators=300,
            min_samples_leaf=0.1,
            n_components=250,
            random_state=random_state,
        ),
        KMeans(n_clusters=20, n_init=20, random_state=random_state),
    )
    pipeline.fit(X, y)
    clusters = pipeline.named_steps["kmeans"].labels_
    ari = adjusted_rand_score(y, clusters)
    plt.figure(figsize=(4, 4))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="coolwarm", s=20)
    plt.title(f"SupNys + KMeans (ARI={ari:.3f})")
    plt.xlabel("x1")
    plt.ylabel("x2")


def demo_quadratic_gp(random_state: int = 2) -> None:
    rng = np.random.default_rng(random_state)
    X = np.linspace(-3, 3, 120)[:, None]
    y = X[:, 0] ** 2 + rng.normal(scale=0.5, size=X.shape[0])
    pipeline = make_pipeline(
        SupervisedNystroem(
            forest_kind="rf",
            regression=True,
            n_estimators=250,
            min_samples_leaf=0.1,
            n_components=200,
            random_state=random_state,
        ),
        GaussianProcessRegressor(
            kernel=DotProduct(sigma_0=1, sigma_0_bounds="fixed"),
            alpha=1e-3,
            normalize_y=True,
            random_state=random_state,
        ),
    )
    pipeline.fit(X, y)
    grid = np.linspace(-3.5, 3.5, 300)[:, None]
    mean, std = pipeline.predict(grid, return_std=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], y, color="black", s=20, alpha=0.6)
    plt.plot(grid[:, 0], mean, color="tab:blue")
    plt.fill_between(grid[:, 0], mean - std, mean + std, color="tab:blue", alpha=0.2)
    plt.title("SupNys + GP on Quadratic")
    plt.xlabel("x")
    plt.ylabel("y")


def main() -> None:
    demo_moons_pca()
    demo_moons_kmeans()
    demo_quadratic_gp()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
