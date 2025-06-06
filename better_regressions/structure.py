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
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from better_regressions import Silencer, binning_regressor
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
def mi_quantile_regional(x: Float[ND, "n"], y: Float[ND, "n"], q: int = 8) -> tuple[Float[ND, "q"], Float[ND, "q"]]:
    x_edges = quantile_bins(x, q)
    y_edges = quantile_bins(y, q)
    q_x = len(x_edges) - 1
    q_y = len(y_edges) - 1
    p_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_x, _ = np.histogram(x, bins=x_edges)
    p_y, _ = np.histogram(y, bins=y_edges)
    LAPLACE = 1e-6
    p_xy = (p_xy + LAPLACE) / (len(y) + LAPLACE * q_x * q_y)
    p_x = (p_x + LAPLACE) / (len(y) + LAPLACE * q_x)
    p_y = (p_y + LAPLACE) / (len(y) + LAPLACE * q_y)
    # mi per y quantile
    p_x_given_y = p_xy.T / (p_y[:, None] + EPS)
    mi_per_y = np.sum(p_x_given_y * np.log2(p_x_given_y / (p_x[None, :] + EPS) + EPS), axis=1)
    # mi per x quantile
    p_y_given_x = p_xy / (p_x[:, None] + EPS)
    mi_per_x = np.sum(p_y_given_x * np.log2(p_y_given_x / (p_y[None, :] + EPS) + EPS), axis=1)
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

    kde = stats.gaussian_kde(np.vstack([x_ranked, y_ranked]))
    xx, yy = np.mgrid[0:1:100j, 0:1:100j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
    ax_scatter.contourf(xx, yy, zz.reshape(xx.shape), levels=15, cmap="Greys")
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
            total = max(pid_result.total, 0.05)
            synergies[i, j] = synergies[j, i] = cut_small(pid_result.synergy) / total
            redundancies[i, j] = redundancies[j, i] = cut_small(pid_result.redundancy) / total
    synergy_df = pd.DataFrame(synergies, index=X.columns, columns=X.columns)
    synergy_df_ordered = synergy_df.loc[leaf_names, leaf_names]
    logger.info("Synergy matrix computed")
    plt.figure(figsize=(16, 14))
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
    X_scaled = StandardScaler().fit_transform(all_features_numpy)
    fa_check = FactorAnalyzer(n_factors=k + 1, rotation=None)
    with Silencer():
        fa_check.fit(X_scaled)
    ev, _ = fa_check.get_eigenvalues()
    n_factors = sum(ev > 1)
    if n_factors == 0:
        n_factors = 1
    fa = FactorAnalyzer(n_factors=n_factors, rotation="quartimin")
    with Silencer():
        fa.fit(X_scaled)
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


def test_pid():
    N = 1000
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
