import os

import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype as typed
from factor_analyzer import FactorAnalyzer
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import ndarray as ND
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from better_regressions import Linear, Scaler, Silencer
from better_regressions.binned import BinnedRegression
from better_regressions.tree_rendering import (
    InfoDecomposition,
    MITree,
    render_tree_interactive,
)

EPS = 1e-12


@typed
def mi_knn(x: Float[ND, "n"], y: Float[ND, "n"]) -> float:
    result = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3)
    assert result.shape == (1,)
    return float(result[0])


@typed
def quantile_bins(x: Float[ND, "n"], q: int) -> Float[ND, "m"]:
    # Using KMeans++ to find centroids for binning
    with Silencer():  # KMeans can be verbose with convergence warnings
        kmeans = KMeans(n_clusters=q, init="k-means++", n_init="auto", random_state=42)
        kmeans.fit(x.reshape(-1, 1))
    centroids = np.sort(kmeans.cluster_centers_.flatten())
    prev_centroids = np.roll(centroids, 1)
    rng = np.quantile(x, 0.98) - np.quantile(x, 0.02)
    mask = np.abs(centroids - prev_centroids) < 1e-9 * rng
    centroids = np.delete(centroids, mask)

    # Compute edges as midpoints between centroids
    edges = np.zeros(len(centroids) + 1)
    edges[1:-1] = (centroids[:-1] + centroids[1:]) / 2
    edges[0] = x.min() - 1e-3 * rng
    edges[-1] = x.max() + 1e-3 * rng
    return edges


@typed
def entropy_quantile(x: Float[ND, "n"], q: int = 8) -> float:
    x_edges = quantile_bins(x, q)
    p_x, _ = np.histogram(x, bins=x_edges)
    p_x = np.clip(p_x / np.sum(p_x), EPS, 1)
    return -np.sum(p_x * np.log2(p_x))


@typed
def mi_quantile(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8, norm: bool = False) -> float:
    """
    Computes normalized mutual information between x and y using bins found by KMeans.
    I(x, y) = H(x) + H(y) - H(x, y)
    Result is normalized by min(H(x), H(y)).
    """
    x_edges = quantile_bins(x, q)
    y_edges = quantile_bins(y, q)

    h_x = entropy_quantile(x, q)
    h_y = entropy_quantile(y, q)

    counts_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = np.clip(counts_xy / np.sum(counts_xy), EPS, 1)
    h_xy = -np.sum(p_xy * np.log2(p_xy))

    mi = max(float(h_x + h_y - h_xy), 0)
    if norm:
        mi /= min(h_x, h_y) + EPS
    return mi


@typed
def mi_quantile_regional(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8) -> tuple[Float[ND, "q"], Float[ND, "q"]]:
    x_edges = quantile_bins(x, q)
    y_edges = quantile_bins(y, q)
    p_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_x, _ = np.histogram(x, bins=x_edges)
    p_y, _ = np.histogram(y, bins=y_edges)

    p_xy = np.clip(p_xy / len(y), EPS, 1)
    p_x = np.clip(p_x / len(y), EPS, 1)
    p_y = np.clip(p_y / len(y), EPS, 1)

    # mi per y quantile
    p_x_given_y = p_xy.T / p_y[:, None]
    mi_per_y = np.sum(p_x_given_y * np.log2(p_x_given_y / p_x[None, :]), axis=1)
    # mi per x quantile
    p_y_given_x = p_xy / p_x[:, None]
    mi_per_x = np.sum(p_y_given_x * np.log2(p_y_given_x / p_y[None, :]), axis=1)
    return mi_per_x, mi_per_y


@typed
def plot_copula(
    x: Float[ND, "n"],
    y: Float[ND, "n"],
    q: int = 8,
    output_file: str | None = None,
    x_name: str = "x",
    y_name: str = "y",
) -> tuple[Float[ND, "q"], Float[ND, "q"]]:
    mi_per_x, mi_per_y = mi_quantile_regional(x, y, q)
    q_x = len(mi_per_x)
    q_y = len(mi_per_y)

    x_ranked = stats.rankdata(x) / len(x)
    y_ranked = stats.rankdata(y) / len(y)

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(5, 5)

    ax_scatter = fig.add_subplot(gs[1:5, 0:4])
    ax_hist_x = fig.add_subplot(gs[0, 0:4], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:5, 4], sharey=ax_scatter)

    ax_scatter.scatter(x_ranked, y_ranked, alpha=0.5, s=1, rasterized=True)
    ax_scatter.set_xlabel(f"Rank of {x_name}")
    ax_scatter.set_ylabel(f"Rank of {y_name}")
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)

    ax_scatter.tick_params(axis="x", labelbottom=True, labeltop=False)
    ax_scatter.tick_params(axis="y", labelleft=True, labelright=False)

    x_step_pos = np.linspace(0, 1, q_x)
    ax_hist_x.step(x_step_pos, mi_per_x, where="mid", color="black", alpha=0.8, linewidth=2)
    ax_hist_x.set_ylabel("MI (bits)")
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_x.set_title(f"Copula of {x_name} and {y_name}")

    y_step_pos = np.linspace(0, 1, q_y)
    ax_hist_y.step(mi_per_y, y_step_pos, where="mid", color="black", alpha=0.8, linewidth=2)
    ax_hist_y.set_xlabel("MI (bits)")
    ax_hist_y.tick_params(axis="y", labelleft=False)

    plt.tight_layout(pad=1.0)
    if output_file:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()
    return mi_per_x, mi_per_y


@typed
def pid_quantile(y: Float[ND, "n"], a: Float[ND, "n"], b: Float[ND, "n"], q: int = 6) -> InfoDecomposition:
    X = np.column_stack([a, b])
    linear = Scaler(Linear(alpha=1e-9))
    linear.fit(X, y)
    linear_preds = cross_val_predict(linear, X, y, cv=3, method="predict")

    a_edges = quantile_bins(a, q)
    b_edges = quantile_bins(b, q)
    a_indices = np.clip(np.digitize(a, a_edges), 0, q - 1)
    b_indices = np.clip(np.digitize(b, b_edges), 0, q - 1)
    joint_indices = a_indices * q + b_indices
    joint_preds = np.zeros(len(y))
    _, inverse_indices = np.unique(joint_indices, return_inverse=True)
    joint_means = np.bincount(inverse_indices, weights=y) / (np.bincount(inverse_indices) + EPS)
    joint_preds = joint_means[inverse_indices]

    h_y = entropy_quantile(y, q)
    mi_a = mi_quantile(a, y, q) / h_y
    mi_b = mi_quantile(b, y, q) / h_y
    mi_linear = mi_quantile(linear_preds, y, q) / h_y
    mi_joint = mi_quantile(joint_preds, y, q) / h_y
    return InfoDecomposition(mi_joint=mi_joint, mi_linear=mi_linear, mi_a=mi_a, mi_b=mi_b)


@typed
def build_mi_tree(X: Float[ND, "n k"], y: Float[ND, "n"], q: int = 6, names: list[str] | None = None) -> tuple[MITree, dict[tuple[int, int], float]]:
    def merge(xi: Float[ND, "n"], xj: Float[ND, "n"]) -> Float[ND, "n"]:
        model = Scaler(Linear(alpha=1e-9))
        X = np.column_stack([xi, xj])
        model.fit(X, y)
        return model.predict(X)

    n, k_initial = X.shape
    h_y = entropy_quantile(y, q)
    MIs_to_return: dict[tuple[int, int], float] = {}
    for i in range(k_initial):
        for j in range(i + 1, k_initial):
            mi = mi_quantile(X[:, i], X[:, j], q, norm=True)
            MIs_to_return[(i, j)] = mi

    MIs = MIs_to_return.copy()

    active = set(range(k_initial))
    if names is None:
        names = [f"x{i}" for i in range(k_initial)]
    trees = [
        MITree(
            mi_target=mi_quantile(X[:, i], y, q) / h_y,
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
                mi_target=mi_quantile(merged, y, q) / h_y,
                mi_join=MIs[(merge_i, merge_j)],
                decomp=pid,
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
            MIs[(i, k_new)] = mi_quantile(X[:, i], X[:, k_new], q, norm=True)
        active.add(k_new)

    return trees[-1], MIs_to_return


def get_leaf_names(tree: MITree) -> list[str]:
    if tree.left is None and tree.right is None:
        return [tree.name]
    leaf_names = []
    if tree.left:
        leaf_names.extend(get_leaf_names(tree.left))
    if tree.right:
        leaf_names.extend(get_leaf_names(tree.right))
    return leaf_names


def _compute_and_plot_copulas_and_regional_mi(X: pd.DataFrame, y_numpy: Float[ND, "n"], output_dir: str):
    copulas_dir = os.path.join(output_dir, "copulas")
    os.makedirs(copulas_dir, exist_ok=True)
    regional_q = 8
    regional_mis_x, regional_mis_y = [], []
    for col in tqdm(X.columns, desc="Plotting copulas"):
        x_col = X[col].to_numpy()
        mi_per_x, mi_per_y = plot_copula(
            x_col,
            y_numpy,
            q=regional_q,
            output_file=os.path.join(copulas_dir, f"{col}.png"),
            x_name=col,
            y_name="target",
        )
        padded_mi_x = np.zeros(regional_q)
        padded_mi_x[: len(mi_per_x)] = mi_per_x
        regional_mis_x.append(padded_mi_x)
        padded_mi_y = np.zeros(regional_q)
        padded_mi_y[: len(mi_per_y)] = mi_per_y
        regional_mis_y.append(padded_mi_y)
    x_regions_df = pd.DataFrame(regional_mis_x, index=X.columns, columns=[f"q{i}" for i in range(regional_q)])
    x_regions_df["total"] = x_regions_df.mean(axis=1)
    x_regions_df.iloc[:, :-1] = x_regions_df.iloc[:, :-1] / x_regions_df["total"].values[:, None]
    x_regions_df = (100 * x_regions_df.sort_values("total", ascending=False)).round().astype(int)
    with open(os.path.join(output_dir, "x_regions.txt"), "w") as f:
        f.write(x_regions_df.to_string())
    y_regions_df = pd.DataFrame(regional_mis_y, index=X.columns, columns=[f"q{i}" for i in range(regional_q)])
    y_regions_df["total"] = y_regions_df.mean(axis=1)
    y_regions_df.iloc[:, :-1] = y_regions_df.iloc[:, :-1] / y_regions_df["total"].values[:, None]
    y_regions_df = (100 * y_regions_df.sort_values("total", ascending=False)).round().astype(int)
    with open(os.path.join(output_dir, "y_regions.txt"), "w") as f:
        f.write(y_regions_df.to_string())


def _compute_and_plot_structure_matrices(X: pd.DataFrame, X_numpy: Float[ND, "n k"], y_numpy: Float[ND, "n"], output_dir: str, q: int) -> list[str]:
    k = X.shape[1]
    tree, initial_mis = build_mi_tree(X_numpy, y_numpy, q, names=list(X.columns))
    MIs = np.zeros((k, k))
    for (i, j), mi in initial_mis.items():
        MIs[i, j] = MIs[j, i] = mi
    leaf_names = get_leaf_names(tree)
    mi_df = pd.DataFrame(MIs, index=X.columns, columns=X.columns)
    mi_df_ordered = mi_df.loc[leaf_names, leaf_names]
    logger.info("MI matrix computed")
    plt.figure(figsize=(16, 14))
    with Silencer():
        mask = mi_df_ordered < 2e-3
        sns.heatmap(mi_df_ordered * 100, annot=True, fmt=".1f", cmap="Blues", mask=mask, square=True, cbar=False)
    plt.title("MI Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mi_matrix.png"))
    plt.close()
    synergies = np.zeros((k, k))
    redundancies = np.zeros((k, k))
    cut_small = lambda x: x if x > 5e-3 else 0
    for i in tqdm(range(k), desc="Computing PID for matrices"):
        for j in range(i + 1, k):
            pid_result = pid_quantile(y_numpy, X_numpy[:, i], X_numpy[:, j], q)
            synergy = cut_small(pid_result.mi_joint - pid_result.mi_linear)
            redundancy = cut_small(pid_result.mi_a + pid_result.mi_b - pid_result.mi_linear)
            total = max(pid_result.mi_joint, 0.05)
            synergies[i, j] = synergies[j, i] = synergy / total
            redundancies[i, j] = redundancies[j, i] = redundancy / total
    synergy_df = pd.DataFrame(synergies, index=X.columns, columns=X.columns)
    synergy_df_ordered = synergy_df.loc[leaf_names, leaf_names]
    logger.info("Synergy matrix computed")
    plt.figure(figsize=(16, 14))
    with Silencer():
        mask = synergy_df_ordered < 2e-3
        sns.heatmap(synergy_df_ordered * 100, annot=True, fmt=".1f", cmap="Blues", mask=mask, square=True, cbar=False)
    plt.title("Synergy Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synergy_matrix.png"))
    plt.close()
    redundancy_df = pd.DataFrame(redundancies, index=X.columns, columns=X.columns)
    redundancy_df_ordered = redundancy_df.loc[leaf_names, leaf_names]
    logger.info("Redundancy matrix computed")
    plt.figure(figsize=(16, 14))
    with Silencer():
        mask = redundancy_df_ordered < 2e-3
        sns.heatmap(redundancy_df_ordered * 100, annot=True, fmt=".1f", cmap="Blues", mask=mask, square=True, cbar=False)
    plt.title("Redundancy Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "redundancy_matrix.png"))
    plt.close()
    with open(os.path.join(output_dir, "mi_tree.txt"), "w") as f:
        f.write(str(tree))
    render_tree_interactive(tree, output_file=os.path.join(output_dir, "interactive_tree.html"))
    return leaf_names


def _perform_and_plot_factor_analysis(
    X: pd.DataFrame,
    X_numpy: Float[ND, "n k"],
    y_numpy: Float[ND, "n"],
    output_dir: str,
    leaf_names: list[str] | None,
):
    k = X.shape[1]
    logger.info("Performing factor analysis")
    all_features_numpy = np.hstack([X_numpy, y_numpy.reshape(-1, 1)])
    all_feature_names = list(X.columns) + ["target"]
    # Convert to quantiles to avoid dependency on marginal distributions
    X_quantiles = pd.DataFrame(all_features_numpy).rank(pct=True).to_numpy()
    fa_check = FactorAnalyzer(n_factors=k + 1, rotation=None)
    with Silencer():
        fa_check.fit(X_quantiles)
    ev, _ = fa_check.get_eigenvalues()
    n_factors = sum(ev > 1)
    if n_factors == 0:
        n_factors = 1
    fa = FactorAnalyzer(n_factors=n_factors, rotation="quartimin")
    with Silencer():
        fa.fit(X_quantiles)
    loadings = fa.loadings_
    loadings_df = pd.DataFrame(loadings, index=all_feature_names, columns=[f"Factor {i + 1}" for i in range(n_factors)])
    if leaf_names:
        ordered_names = leaf_names + ["target"]
        loadings_df = loadings_df.loc[ordered_names]
    plt.figure(figsize=(8, 12))
    sns.heatmap(loadings_df, annot=True, fmt=".2f", cmap="vlag", center=0.0)
    plt.title("Factors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "factors.png"))
    plt.close()


def show_structure(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    q: int = 6,
    do_regional_mi: bool = True,
    do_structure_matrices: bool = True,
    do_factor_analysis: bool = True,
):
    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()
    leaf_names: list[str] | None = None
    if do_regional_mi:
        _compute_and_plot_copulas_and_regional_mi(X, y_numpy, output_dir)
    if do_structure_matrices:
        leaf_names = _compute_and_plot_structure_matrices(X, X_numpy, y_numpy, output_dir, q)
    if do_factor_analysis:
        _perform_and_plot_factor_analysis(X, X_numpy, y_numpy, output_dir, leaf_names)


def test_mi_quantile():
    N = 10**4
    a = np.random.randint(0, 2, N)
    b = np.random.randint(0, 2, N)
    y = 2 * a + b
    a = a.astype(float)
    b = b.astype(float)
    y = y.astype(float)


def test_pid():
    N = 10**5

    print("1. Independent coins:")
    a = np.random.binomial(
        1,
        0.5,
        N,
    ).astype(float)
    b = np.random.binomial(1, 0.5, N).astype(float)
    y = np.random.binomial(1, 0.5, N).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Joint:      {result.mi_joint:.4f}")
    print(f"   Additive:   {result.mi_linear:.4f}")
    print(f"   A:          {result.mi_a:.4f}")
    print(f"   B:          {result.mi_b:.4f}")
    print()

    print("2. Y = A XOR B (synergy):")
    a = np.random.binomial(1, 0.5, N).astype(float)
    b = np.random.binomial(1, 0.5, N).astype(float)
    y = (a.astype(int) ^ b.astype(int)).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Joint:      {result.mi_joint:.4f}")
    print(f"   Additive:   {result.mi_linear:.4f}")
    print(f"   A:          {result.mi_a:.4f}")
    print(f"   B:          {result.mi_b:.4f}")
    print()

    print("3. A = Y + noise, B = Y + noise (redundancy):")
    y = np.random.binomial(1, 0.5, N).astype(float)
    noise_a = np.random.binomial(1, 0.1, N).astype(float)
    noise_b = np.random.binomial(1, 0.1, N).astype(float)
    a = (y.astype(int) ^ noise_a.astype(int)).astype(float)
    b = (y.astype(int) ^ noise_b.astype(int)).astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Joint:      {result.mi_joint:.4f}")
    print(f"   Additive:   {result.mi_linear:.4f}")
    print(f"   A:          {result.mi_a:.4f}")
    print(f"   B:          {result.mi_b:.4f}")
    print()

    print("4. Linear relationship:")
    a = np.random.randn(N)
    b = np.random.randn(N)
    y = a + b
    result = pid_quantile(y, a, b)
    print(f"   Joint:      {result.mi_joint:.4f}")
    print(f"   Additive:   {result.mi_linear:.4f}")
    print(f"   A:          {result.mi_a:.4f}")
    print(f"   B:          {result.mi_b:.4f}")
    print()

    print("5. Complete separation:")
    N = 10**4
    a = np.random.randint(0, 2, N)
    b = np.random.randint(0, 2, N)
    y = 2 * a + b
    a = a.astype(float)
    b = b.astype(float)
    y = y.astype(float)
    result = pid_quantile(y, a, b)
    print(f"   Joint:      {result.mi_joint:.4f}")
    print(f"   Additive:   {result.mi_linear:.4f}")
    print(f"   A:          {result.mi_a:.4f}")
    print(f"   B:          {result.mi_b:.4f}")


if __name__ == "__main__":
    test_pid()
    # test_mi_quantile()
