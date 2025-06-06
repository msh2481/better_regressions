import os

import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from better_regressions import binning_regressor
from better_regressions.tree_rendering import MITree, PIDResult, render_tree_interactive

EPS = 1e-12


@typed
def mi_knn(x: Float[ND, "n"], y: Float[ND, "n"]) -> float:
    result = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3)
    assert result.shape == (1,)
    return float(result[0])


@typed
def joint_entropy_quantile(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8) -> float:
    x_edges = quantile_bins(x, q)
    y_edges = quantile_bins(y, q)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    probabilities = counts.flatten() / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities + EPS))


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


@typed
def quantile_bins(x: Float[ND, "n"], q: int) -> Float[ND, "qp1"]:
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.quantile(x, quantiles)
    rng = edges[-1] - edges[0] + 1e-9
    min_diff = 1e-3 * (rng / q)
    edges[:-1] -= min_diff / 2
    edges[-1] += min_diff / 2
    while True:
        changed = False
        for i in range(len(edges) - 1):
            cur_diff = edges[i + 1] - edges[i]
            if cur_diff < min_diff:
                edges = np.delete(edges, i + 1)
                changed = True
                break
        if not changed:
            break
    while len(edges) < q + 1:
        edges = np.concatenate([edges, [edges[-1] + min_diff]])
    return edges


@typed
def pid_quantile(y: Float[ND, "n"], a: Float[ND, "n"], b: Float[ND, "n"], q: int = 6) -> PIDResult:
    y_edges = quantile_bins(y, q)
    a_edges = quantile_bins(a, q)
    b_edges = quantile_bins(b, q)

    p_ya, _, _ = np.histogram2d(y, a, bins=[y_edges, a_edges])
    p_yb, _, _ = np.histogram2d(y, b, bins=[y_edges, b_edges])
    p_yab, _ = np.histogramdd([y, a, b], bins=[y_edges, a_edges, b_edges])

    p_y, _ = np.histogram(y, bins=y_edges)
    p_a, _ = np.histogram(a, bins=a_edges)
    p_b, _ = np.histogram(b, bins=b_edges)

    LAPLACE = 1e-6
    p_ya = (p_ya + LAPLACE) / (len(y) + LAPLACE * q * q)
    p_yb = (p_yb + LAPLACE) / (len(y) + LAPLACE * q * q)
    p_yab = (p_yab + LAPLACE) / (len(y) + LAPLACE * q * q * q)
    p_y = (p_y + LAPLACE) / (len(y) + LAPLACE * q)
    p_a = (p_a + LAPLACE) / (len(y) + LAPLACE * q)
    p_b = (p_b + LAPLACE) / (len(y) + LAPLACE * q)

    p_a_given_y = p_ya / (p_y[:, None] + EPS)
    p_b_given_y = p_yb / (p_y[:, None] + EPS)

    pmi_a_per_y = np.sum(p_a_given_y * np.log2(p_a_given_y / (p_a + EPS) + EPS), axis=1)
    pmi_b_per_y = np.sum(p_b_given_y * np.log2(p_b_given_y / (p_b + EPS) + EPS), axis=1)

    redundancy = np.sum(p_y * np.minimum(pmi_a_per_y, pmi_b_per_y))
    unique_a = np.sum(p_y * (pmi_a_per_y - np.minimum(pmi_a_per_y, pmi_b_per_y)))
    unique_b = np.sum(p_y * (pmi_b_per_y - np.minimum(pmi_a_per_y, pmi_b_per_y)))

    p_ab = np.sum(p_yab, axis=0)
    p_y_given_ab = p_yab / (p_ab + EPS)
    total = np.sum(p_yab * np.log2(p_y_given_ab / (p_y + EPS) + EPS))

    synergy = total - (unique_a + unique_b + redundancy)

    y_entropy = -np.sum(p_y * np.log2(p_y + EPS))
    redundancy = np.clip(redundancy / y_entropy, 0, 1)
    unique_a = np.clip(unique_a / y_entropy, 0, 1)
    unique_b = np.clip(unique_b / y_entropy, 0, 1)
    total = np.clip(total / y_entropy, 0, 1)
    synergy = np.clip(synergy / y_entropy, 0, 1)

    return PIDResult(redundancy=redundancy, unique_a=unique_a, unique_b=unique_b, synergy=synergy, total=total)


@typed
def build_mi_tree(X: Float[ND, "n k"], y: Float[ND, "n"], q: int = 6, names: list[str] | None = None) -> tuple[MITree, dict[tuple[int, int], float]]:
    def merge(xi: Float[ND, "n"], xj: Float[ND, "n"]) -> Float[ND, "n"]:
        binning_model = binning_regressor(X_bins=q, y_bins=q)
        X_stack = np.column_stack([xi, xj])
        binning_predictions = cross_val_predict(binning_model, X_stack, y, cv=3, n_jobs=3)
        assert binning_predictions.shape == y.shape
        return binning_predictions

    n, k_initial = X.shape
    MIs_to_return: dict[tuple[int, int], float] = {}
    for i in range(k_initial):
        for j in range(i + 1, k_initial):
            mi = mi_quantile(X[:, i], X[:, j], q)
            MIs_to_return[(i, j)] = mi

    MIs = MIs_to_return.copy()

    active = set(range(k_initial))
    if names is None:
        names = [f"x{i}" for i in range(k_initial)]
    trees = [
        MITree(
            mi_target=mi_quantile(X[:, i], y, q),
            name=names[i],
        )
        for i in range(k_initial)
    ]
    for _ in tqdm(range(k_initial - 1), desc="Building MI tree"):
        to_sort = [(mi, i, j) for (i, j), mi in MIs.items() if i in active and j in active]
        to_sort.sort(key=lambda x: x[0], reverse=True)

        _, u, v = to_sort[0]
        merge_i, merge_j = min(u, v), max(u, v)

        merged = merge(X[:, merge_i], X[:, merge_j])
        pid = pid_quantile(y, X[:, merge_i], X[:, merge_j], q)
        k_new = X.shape[1]

        assert len(trees) == k_new
        trees.append(
            MITree(
                mi_target=mi_quantile(merged, y, q),
                mi_join=MIs[(merge_i, merge_j)],
                pid=pid,
                left=trees[merge_i],
                right=trees[merge_j],
            )
        )

        active.remove(merge_i)
        active.remove(merge_j)
        X = np.hstack([X, merged[:, None]])
        MIs.pop((merge_i, merge_j))
        for i in active:
            MIs.pop((min(i, merge_i), max(i, merge_i)), None)
            MIs.pop((min(i, merge_j), max(i, merge_j)), None)
            MIs[(i, k_new)] = mi_quantile(X[:, i], X[:, k_new], q)
        active.add(k_new)

    return trees[-1], MIs_to_return


def get_leaf_names(tree: MITree) -> list[str]:
    if tree.left is None and tree.right is None:
        # Assuming leaves have names
        return [tree.name]

    leaf_names = []
    if tree.left:
        leaf_names.extend(get_leaf_names(tree.left))
    if tree.right:
        leaf_names.extend(get_leaf_names(tree.right))
    return leaf_names


def show_structure(X: pd.DataFrame, y: pd.Series, output_dir: str, q: int = 6):
    n, k = X.shape

    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()

    tree, initial_mis = build_mi_tree(X_numpy, y_numpy, q, names=list(X.columns))

    MIs = np.zeros((k, k))
    for (i, j), mi in initial_mis.items():
        MIs[i, j] = MIs[j, i] = mi

    leaf_names = get_leaf_names(tree)
    mi_df = pd.DataFrame(MIs, index=X.columns, columns=X.columns)
    mi_df_ordered = mi_df.loc[leaf_names, leaf_names]

    logger.info(f"MI matrix computed")
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(mi_df_ordered, dtype=bool))
    mask = mask | (mi_df_ordered < 2e-3)
    sns.heatmap(mi_df_ordered * 100, annot=True, fmt=".1f", cmap="viridis", mask=mask, square=True)
    plt.title("MI Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mi_matrix.png"))
    plt.close()

    with open(os.path.join(output_dir, "mi_tree.txt"), "w") as f:
        f.write(str(tree))

    render_tree_interactive(tree, output_file=os.path.join(output_dir, "interactive_tree.html"))


def test_entropy():
    N = 1000
    X = pd.DataFrame(np.random.randn(N, 10), columns=[f"x{i}" for i in range(10)])
    print(X.head())
    y = X.sum(axis=1) * 0 + np.random.randn(N)
    show_structure(X, y, "output")


def test_pid():
    N = 1000
    # np.random.seed(42)

    print("=== PID Tests ===\n")
    print("1. Independent coins:")
    a = np.random.binomial(
        1,
        0.5,
        N,
    ).astype(float)
    b = np.random.binomial(1, 0.5, N).astype(float)
    y = np.random.binomial(1, 0.5, N).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Redundancy: {result.redundancy:.4f}")
    print(f"   Unique A:   {result.unique_a:.4f}")
    print(f"   Unique B:   {result.unique_b:.4f}")
    print(f"   Synergy:    {result.synergy:.4f}")
    print(f"   Total:      {result.total:.4f}")
    print()
    print("2. Y = A XOR B (synergy):")
    a = np.random.binomial(1, 0.5, N).astype(float)
    b = np.random.binomial(1, 0.5, N).astype(float)
    y = (a.astype(int) ^ b.astype(int)).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Redundancy: {result.redundancy:.4f}")
    print(f"   Unique A:   {result.unique_a:.4f}")
    print(f"   Unique B:   {result.unique_b:.4f}")
    print(f"   Synergy:    {result.synergy:.4f}")
    print(f"   Total:      {result.total:.4f}")
    print()
    print("3. A = Y + noise, B = Y + noise (redundancy):")
    y = np.random.binomial(1, 0.5, N).astype(float)
    noise_a = np.random.binomial(1, 0.1, N).astype(float)
    noise_b = np.random.binomial(1, 0.1, N).astype(float)
    a = (y.astype(int) ^ noise_a.astype(int)).astype(float)
    b = (y.astype(int) ^ noise_b.astype(int)).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Redundancy: {result.redundancy:.4f}")
    print(f"   Unique A:   {result.unique_a:.4f}")
    print(f"   Unique B:   {result.unique_b:.4f}")
    print(f"   Synergy:    {result.synergy:.4f}")
    print(f"   Total:      {result.total:.4f}")
    print()


if __name__ == "__main__":
    test_pid()
    # test_entropy()
