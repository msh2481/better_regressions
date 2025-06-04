import numpy as np
import pandas as pd
from beartype import beartype as typed
from beartype.typing import Literal, Self
from jaxtyping import Float
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_predict

from better_regressions import Linear, Scaler, binning_regressor


@typed
def mi_knn(x: Float[ND, "n"], y: Float[ND, "n"]) -> float:
    result = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3)
    assert result.shape == (1,)
    return float(result[0])


@typed
def joint_entropy_quantile(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8) -> float:
    x = np.argsort(np.argsort(x)) / len(x)
    y = np.argsort(np.argsort(y)) / len(y)
    counts, _, _ = np.histogram2d(x, y, bins=q)
    probabilities = counts.flatten() / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))


@typed
def mi_quantile(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8) -> float:
    """
    I(x, y) = H(x) + H(y) - H(x, y)
    H(x) = H(y) = log(q)
    log(q) <= I(x, y) <= 2 * log(q)
    """
    logq = np.log2(q)
    mi = np.clip((2 * logq - joint_entropy_quantile(x, y, q)) / logq, 0, 1)
    return mi


class MITree:
    mi_target: float
    mi_join: float | None = None
    left: Self | None = None
    right: Self | None = None
    name: str | None = None

    def __init__(self, mi_target: float, name: str | None = None, mi_join: float | None = None, left: Self | None = None, right: Self | None = None):
        self.mi_target = mi_target
        self.name = name
        self.mi_join = mi_join
        self.left = left
        self.right = right

    def __str__(self) -> str:
        if not self.left and not self.right:
            lines = [
                f"---(name: {self.name})",
                f"   (target: {self.mi_target:.3f})",
            ]
            return "\n".join(lines)

        left_lines = str(self.left).splitlines()
        left_lines.append("")
        right_lines = str(self.right).splitlines()
        stats = [
            f"---(target: {self.mi_target:.3f})",
            f"     (join: {self.mi_join:.3f})" if self.mi_join is not None else "join: None",
        ]
        stats.extend(["|" for _ in range(len(left_lines) - 2)])
        stats.extend(["" for _ in range(len(right_lines))])
        stats_width = max(len(line) for line in stats)
        stats = [line.rjust(stats_width) for line in stats]
        assert len(stats) == len(left_lines) + len(right_lines)
        result_lines = [x + y for x, y in zip(stats, left_lines + right_lines)]
        return "\n".join(result_lines)


@typed
def build_mi_tree(x: Float[ND, "n k"], y: Float[ND, "n"], q: int = 8, names: list[str] | None = None) -> MITree:
    def merge(xi: Float[ND, "n"], xj: Float[ND, "n"]) -> Float[ND, "n"]:
        linear_model = Scaler(Linear())
        binning_model = binning_regressor(X_bins=q, y_bins=q)
        X = np.hstack([xi, xj])
        linear_predictions = cross_val_predict(linear_model, X, y, cv=3, n_jobs=3)
        binning_predictions = cross_val_predict(binning_model, X, y, cv=3, n_jobs=3)
        linear_spearman = stats.spearmanr(linear_predictions, y)[0]
        binning_spearman = stats.spearmanr(binning_predictions, y)[0]
        if linear_spearman > binning_spearman:
            return linear_predictions
        else:
            return binning_predictions

    n, k = x.shape
    MIs: dict[tuple[int, int], float] = {}
    for merge_i in range(k):
        for merge_j in range(merge_i + 1, k):
            mi = mi_quantile(x[:, merge_i], x[:, merge_j], q)
            MIs[(merge_i, merge_j)] = mi
    active = set(range(k))
    if names is None:
        names = [f"x{i}" for i in range(k)]
    trees = [
        MITree(
            mi_target=mi_quantile(x[:, i], y, q),
            name=names[i],
        )
        for i in range(k)
    ]
    while len(active) > 1:
        to_sort = [(mi, i, j) for (i, j), mi in MIs.items() if i in active and j in active]
        to_sort.sort(key=lambda x: x[0], reverse=True)
        merge_i, merge_j = to_sort[0][1:]
        merged = merge(x[:, merge_i], x[:, merge_j])
        k = x.shape[1]

        assert len(trees) == k
        trees.append(
            MITree(
                mi_target=mi_quantile(merged, y, q),
                mi_join=MIs[(merge_i, merge_j)],
                left=trees[merge_i],
                right=trees[merge_j],
            )
        )

        active.remove(merge_i)
        active.remove(merge_j)
        active.add(k)

        x = np.hstack([x, merged[:, None]])
        MIs.pop((merge_i, merge_j))
        for i in active:
            MIs.pop((i, merge_i))
            MIs.pop((i, merge_j))
            if i != k:
                MIs[(i, k)] = mi_quantile(x[:, i], x[:, k], q)

    return trees[-1]


def test_entropy():
    corrs = []
    mis = []
    for corr in np.linspace(-1, 1, 100):
        x = np.random.randn(10000)
        y = x * corr + np.sqrt(1 - np.abs(corr)) * np.random.randn(10000)
        c = np.corrcoef(x, y)[0, 1]
        corrs.append(c)
        mis.append(mi_quantile(x, y))
    plt.plot(corrs, mis)
    plt.show()


def test_mi_tree():
    a = MITree(mi_target=0.5, name="a")
    b = MITree(mi_target=0.3, name="b")
    c = MITree(mi_target=0.2, name="c")
    x = MITree(mi_target=0.6, mi_join=0.8, name="x", left=a, right=b)
    y = MITree(mi_target=0.9345, mi_join=0.1245, left=x, right=c)
    print(y)


if __name__ == "__main__":
    test_mi_tree()
