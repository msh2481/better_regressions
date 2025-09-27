from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline

from better_regressions.kernel import SupervisedNystroem


def demo_moons_pca(random_state: int = 0) -> None:
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=random_state)
    pipeline = make_pipeline(
        SupervisedNystroem(
            regression=False,
            min_samples_leaf=0.05,
            n_components=400,
            random_state=random_state,
        ),
        PCA(n_components=2),
    )
    embedding = pipeline.fit_transform(X, y)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=40)
    plt.title("SupNys + PCA on Moons")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=40)
    plt.title("Original Moons")
    plt.xlabel("x1")
    plt.ylabel("x2")


def demo_circles_pca(random_state: int = 4) -> None:
    X, y = make_circles(n_samples=800, factor=0.4, noise=0.06, random_state=random_state)
    pipeline = make_pipeline(
        # SupervisedNystroem(
        #     regression=False,
        #     min_samples_leaf=0.02,
        #     random_state=random_state,
        #     n_components=400,
        # ),
        Nystroem(n_components=200),
        TSNE(n_components=2),
    )
    embedding = pipeline.fit_transform(X, y)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30)
    plt.title("Original Circles")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.subplot(1, 2, 2)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=30)
    plt.title("Nystroem-TSNE on Circles")
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def demo_moons_kmeans(random_state: int = 1) -> None:
    X, y = make_moons(n_samples=400, noise=0.25, random_state=random_state)
    pipeline = make_pipeline(
        SupervisedNystroem(
            regression=False,
            min_samples_leaf=0.2,
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
    X = np.linspace(-3, 3, 1000)[:, None]
    y = X[:, 0] ** 2 + rng.normal(scale=0.5, size=X.shape[0])
    pipeline = make_pipeline(
        SupervisedNystroem(
            regression=True,
            random_state=random_state,
        ),
        GaussianProcessRegressor(
            kernel=DotProduct(sigma_0=1, sigma_0_bounds="fixed") + WhiteKernel(noise_level=0.1),
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


def demo_swiss_roll_pca(random_state: int = 5) -> None:
    X, t = make_swiss_roll(n_samples=1200, noise=0.05, random_state=random_state)
    pipeline = make_pipeline(
        SupervisedNystroem(
            regression=True,
            min_samples_leaf=0.1,
            random_state=random_state,
            n_components=500,
        ),
        PCA(n_components=2),
    )
    embedding = pipeline.fit_transform(X, t)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=12)
    ax.set_title("Original Swiss Roll")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.subplot(1, 2, 2)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=t, cmap="viridis", s=12)
    plt.title("SupNys + PCA on Swiss Roll")
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def main() -> None:
    # demo_moons_pca()
    demo_circles_pca()
    # demo_swiss_roll_pca()
    # demo_moons_kmeans()
    # demo_quadratic_gp()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
