import numpy as np
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from better_regressions.utils import Silencer

INF = 1e100


class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int = 10, one_hot: bool = False):
        self.n_bins = n_bins
        self.one_hot = one_hot

    def _fit_1d(self, x_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with Silencer():
            kmeans = KMeans(n_clusters=self.n_bins, init="k-means++", n_init="auto", random_state=42)
            kmeans.fit(x_1d.reshape(-1, 1))
        centroids = np.sort(kmeans.cluster_centers_.flatten())
        prev_centroids = np.roll(centroids, 1)
        rng = np.quantile(x_1d, 0.98) - np.quantile(x_1d, 0.02)
        mask = np.abs(centroids - prev_centroids) < 1e-9 * rng
        centroids = np.delete(centroids, mask)

        # Compute edges as midpoints between centroids
        edges = np.zeros(len(centroids) + 1)
        edges[1:-1] = (centroids[:-1] + centroids[1:]) / 2
        edges[0] = x_1d.min()
        edges[-1] = x_1d.max()
        return centroids, edges

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        if self.n_bins > 1:
            assert self.n_bins > 1, "n_bins must be greater than 1 (or 0 for no binning)"
            if X_arr.ndim == 1:
                centers, edges = self._fit_1d(X_arr)
            else:
                raise ValueError(f"X must be 1D, got ndim={X_arr.ndim}")
            self.bin_centers_ = centers
            self.bin_edges_ = edges
            self.bin_sizes_ = self.bin_edges_[1:] - self.bin_edges_[:-1]
        else:
            self.bin_centers_ = np.array([])
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        if self.n_bins == 0:
            return X_arr
        if X_arr.ndim == 1:
            dist = np.abs(X_arr[:, None] - self.bin_centers_)
            bin_indices = np.argmin(dist, axis=1)
        else:
            raise ValueError(f"X must be 1D array, got ndim={X_arr.ndim}")
        if self.one_hot:
            if X_arr.ndim == 1:
                return np.eye(self.n_bins)[bin_indices]
            # one-hot per feature and flatten: (n_samples, n_features, n_bins) -> (n_samples, n_features*n_bins)
            oh = np.eye(self.n_bins)[bin_indices]
            return oh.reshape(X_arr.shape[0], -1)
        return bin_indices


class Adapter(BaseEstimator, TransformerMixin):
    def __init__(self, classifier: BaseEstimator, X_bins: int = 10, y_bins: int = 10, mode: str = "concat"):
        self.X_bins = X_bins
        self.y_bins = y_bins
        self.classifier = classifier
        if mode not in ["concat", "outer"]:
            raise ValueError("mode must be 'concat' or 'outer'")
        self.mode = mode

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"] = None):
        self.X_binners_ = []
        self.y_binner_ = QuantileBinner(self.y_bins, one_hot=False)
        for i in range(X.shape[1]):
            self.X_binners_.append(QuantileBinner(self.X_bins, one_hot=True))
        for i in range(X.shape[1]):
            self.X_binners_[i].fit(X[:, i])
        self.y_binner_.fit(y)

        X_binned = self._transform_X(X)
        y_binned = self.y_binner_.transform(y)
        missing_classes = np.setdiff1d(np.arange(self.y_bins), np.unique(y_binned))
        x_median = np.median(X_binned, axis=0)
        add_to_X = []
        add_to_y = []
        for i in missing_classes:
            add_to_X.append(x_median)
            add_to_y.append(i)
        if len(missing_classes):
            add_to_X = np.stack(add_to_X, axis=0)
            add_to_y = np.array(add_to_y)
            X_binned = np.concatenate([X_binned, add_to_X], axis=0)
            y_binned = np.concatenate([y_binned, add_to_y], axis=0)

        self.classifier_ = clone(self.classifier)
        with Silencer():
            self.classifier_.fit(X_binned, y_binned)

        return self

    @typed
    def _transform_X(self, X: Float[ND, "n_samples n_featueres"]) -> Float[ND, "n_samples n_bins"]:
        if X.shape[1] == 0:
            return np.ones((X.shape[0], 1)) if self.mode == "outer" else np.empty((X.shape[0], 0))
        binned_features = [self.X_binners_[i].transform(X[:, i]) for i in range(X.shape[1])]
        if self.mode == "concat":
            return np.concatenate(binned_features, axis=1)
        elif self.mode == "outer":
            outer_prod = binned_features[0]
            for i in range(1, len(binned_features)):
                outer_prod = outer_prod[:, :, None] * binned_features[i][:, None, :]
                new_shape = (X.shape[0], -1)
                outer_prod = outer_prod.reshape(new_shape)
            return outer_prod
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @typed
    def predict_proba(self, X: Float[ND, "n_samples n_featueres"]) -> Float[ND, "n_samples n_bins"]:
        X_binned = self._transform_X(X)
        full_results = self.classifier_.predict_proba(X_binned)
        actual_y_bins = len(self.y_binner_.bin_centers_)
        return full_results[:, :actual_y_bins]

    @typed
    def predict(self, X: Float[ND, "n_samples n_featueres"]) -> Float[ND, "n_samples"]:
        distribution = self.predict_proba(X)
        mean = distribution @ self.y_binner_.bin_centers_
        return mean

    @typed
    def logpdf(self, X: Float[ND, "n_samples n_featueres"], y: Float[ND, "n_samples"]) -> float:
        distribution = self.predict_proba(X)
        log_distribution = np.log(distribution)
        y_bins = self.y_binner_.transform(y)
        bin_sizes = self.y_binner_.bin_sizes_[y_bins]
        log_pdf = log_distribution[np.arange(X.shape[0]), y_bins] - np.log(bin_sizes + 1e-18)
        return log_pdf.sum()

    @typed
    def sample(self, X: Float[ND, "n_samples n_featueres"]) -> Float[ND, "n_samples"]:
        distribution = self.predict_proba(X)
        # compute cumulative distribution for each sample
        cdf = np.cumsum(distribution, axis=1)
        # sample bin indices based on distribution
        random_vals = np.random.rand(X.shape[0], 1)
        bin_indices = np.argmax(random_vals < cdf, axis=1)
        edges = self.y_binner_.bin_edges_
        left = edges[bin_indices]
        right = edges[bin_indices + 1]
        # sample uniformly within each bin interval
        y = left + np.random.rand(X.shape[0]) * (right - left)
        return y


class AutoAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, classifier: BaseEstimator, mode: str = "concat"):
        self.classifier = classifier
        if mode not in ["concat", "outer"]:
            raise ValueError("mode must be 'concat' or 'outer'")
        self.mode = mode

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"] = None, show_plot: bool = False):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)

        def test_bin_count(k: int) -> float:
            adapter = Adapter(self.classifier, X_bins=k, y_bins=k, mode=self.mode)
            adapter.fit(X_train, y_train)
            return adapter.logpdf(X_val, y_val)

        candidates = [2, 3, 5, 8, 10, 15, 20, 30, 50]
        scores = [test_bin_count(k) for k in candidates]
        if show_plot:
            plt.plot(candidates, scores)
            plt.show()
        idx = np.argmax(scores)
        k = int(1.5 * candidates[idx])
        self.adapter_ = Adapter(self.classifier, X_bins=k, y_bins=k, mode=self.mode)
        self.adapter_.fit(X, y)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return self.adapter_.predict(X)

    @typed
    def logpdf(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> Float[ND, "n_samples"]:
        return self.adapter_.logpdf(X, y)

    @typed
    def predict_proba(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples n_bins"]:
        return self.adapter_.predict_proba(X)

    @typed
    def sample(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return self.adapter_.sample(X)


def binning_regressor(X_bins: int | None = None, y_bins: int | None = None, mode: str = "concat"):
    if (X_bins is None) != (y_bins is None):
        raise ValueError("X_bins and y_bins must both be None (for automatic selection) or both be an integer")
    inner = LogisticRegression(C=1e6)
    if X_bins is None or y_bins is None:
        model = AutoAdapter(inner, mode=mode)
    else:
        model = Adapter(inner, X_bins=X_bins, y_bins=y_bins, mode=mode)
    return model


def test_missing_classes():
    X = np.random.randn(10, 1)
    y = np.random.randint(0, 3, size=10).astype(float)
    model = binning_regressor(X_bins=20, y_bins=20)
    model.fit(X, y)
    print(model.predict_proba(X))
    print(model.predict(X))
    print(model.logpdf(X, y))
    print(model.sample(X))


def test_adapter():
    X = np.random.randn(1000, 1)
    mean = X
    std = 1 / (0.3 + X**2)
    y = np.random.normal(mean, std).ravel()
    adapter = AutoAdapter(LogisticRegression(C=1e6))
    adapter.fit(X, y)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(X, y, "x", alpha=0.2)
    plt.xlim(-3, 3)
    plt.ylim(-15, 15)
    plt.subplot(1, 2, 2)
    k = 10
    xs = [np.random.randn(1000).reshape(-1, 1) for _ in range(k)]
    xs = np.concatenate(xs, axis=0)
    samples = adapter.sample(xs)
    plt.plot(xs, samples, "x", alpha=0.2)
    plt.xlim(-3, 3)
    plt.ylim(-15, 15)
    plt.show()


def test_max():
    N = 10000
    D = 5
    X = np.random.randn(N, D)
    y = np.max(X, axis=1)

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    models = {
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000),
        "XGBRegressor": XGBRegressor(n_estimators=300, max_depth=3),
        "BinningRegressor": binning_regressor(),
    }

    for model_name, model in models.items():
        model.fit(X, y)
    # samples = model.sample(X)

    X = np.random.randn(N, D)
    y = np.max(X, axis=1)
    plt.figure(figsize=(15, 5))

    for i, (model_name, model) in enumerate(models.items()):
        samples = model.predict(X)
        plt.subplot(1, len(models), i + 1)
        plt.plot(y, samples, "x", alpha=0.2)
        plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), "k--")
        plt.xlim(-1, 4)
        plt.ylim(0, 3)
        plt.xlabel("max(x_1, ..., x_D)")
        plt.ylabel(f"{model_name} prediction")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_adapter()
