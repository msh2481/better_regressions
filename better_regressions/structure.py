import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from beartype import beartype as typed
from beartype.typing import Literal, Self
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from better_regressions import binning_regressor

EPS = 1e-12


@dataclass
class PIDResult:
    redundancy: float
    unique_a: float
    unique_b: float
    synergy: float
    total: float


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


class MITree:
    mi_target: float
    mi_join: float | None = None
    pid: PIDResult | None = None
    left: Self | None = None
    right: Self | None = None
    name: str | None = None

    def __init__(self, mi_target: float, name: str | None = None, mi_join: float | None = None, pid: PIDResult | None = None, left: Self | None = None, right: Self | None = None):
        self.mi_target = mi_target
        self.name = name
        self.pid = pid
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
        left_lines.extend(["", "", ""])
        right_lines = str(self.right).splitlines()
        fmt = lambda x: f"{x:.3f}" if x is not None else "None"
        a, b = fmt(self.mi_target), fmt(self.mi_join)
        c, d = fmt(self.pid.total), fmt(self.pid.redundancy)
        e, f = fmt(self.pid.unique_a), fmt(self.pid.unique_b)
        g = fmt(self.pid.synergy)
        stats = [
            f"---(target: {a} |       join: {b})",
            f"    (total: {c} | redundancy: {d})",
            f" (unique_a: {e} |   unique_b: {f})",
            f"                    (synergy: {g})",
        ]
        stats.extend(["|" for _ in range(len(left_lines) - len(stats))])
        stats.extend(["" for _ in range(len(right_lines))])
        stats_width = max(len(line) for line in stats)
        stats = [line.rjust(stats_width) for line in stats]
        assert len(stats) == len(left_lines) + len(right_lines)
        result_lines = [x + y for x, y in zip(stats, left_lines + right_lines)]
        return "\n".join(result_lines)


@typed
def build_mi_tree(X: Float[ND, "n k"], y: Float[ND, "n"], q: int = 6, names: list[str] | None = None) -> MITree:
    def merge(xi: Float[ND, "n"], xj: Float[ND, "n"]) -> Float[ND, "n"]:
        binning_model = binning_regressor(X_bins=q, y_bins=q)
        X = np.column_stack([xi, xj])
        binning_predictions = cross_val_predict(binning_model, X, y, cv=3, n_jobs=3)
        assert binning_predictions.shape == y.shape
        return binning_predictions

    n, k = X.shape
    MIs: dict[tuple[int, int], float] = {}
    for merge_i in range(k):
        for merge_j in range(merge_i + 1, k):
            mi = mi_quantile(X[:, merge_i], X[:, merge_j], q)
            MIs[(merge_i, merge_j)] = mi
    active = set(range(k))
    if names is None:
        names = [f"x{i}" for i in range(k)]
    trees = [
        MITree(
            mi_target=mi_quantile(X[:, i], y, q),
            name=names[i],
        )
        for i in range(k)
    ]
    for _ in tqdm(range(k - 1), desc="Building MI tree"):
        to_sort = [(mi, i, j) for (i, j), mi in MIs.items() if i in active and j in active]
        to_sort.sort(key=lambda x: x[0], reverse=True)
        merge_i, merge_j = to_sort[0][1:]
        merged = merge(X[:, merge_i], X[:, merge_j])
        pid = pid_quantile(y, X[:, merge_i], X[:, merge_j], q)
        k = X.shape[1]

        assert len(trees) == k
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
        active.add(k)

        X = np.hstack([X, merged[:, None]])
        MIs.pop((merge_i, merge_j))
        for i in active:
            MIs.pop((i, merge_i), None)
            MIs.pop((i, merge_j), None)
            if i != k:
                MIs[(i, k)] = mi_quantile(X[:, i], X[:, k], q)

    return trees[-1]


@typed
def tree_to_networkx(tree: MITree, node_id: int = 0) -> tuple[nx.DiGraph, dict[int, MITree], int]:
    G = nx.DiGraph()
    node_map = {node_id: tree}
    G.add_node(node_id)

    next_id = node_id + 1
    if tree.left:
        G_left, map_left, next_id = tree_to_networkx(tree.left, next_id)
        G = nx.compose(G, G_left)
        node_map.update(map_left)
        G.add_edge(node_id, next_id - len(map_left))

    if tree.right:
        G_right, map_right, next_id = tree_to_networkx(tree.right, next_id)
        G = nx.compose(G, G_right)
        node_map.update(map_right)
        G.add_edge(node_id, next_id - len(map_right))

    return G, node_map, next_id


@typed
def pid_to_color(pid: PIDResult | None, total: float | int) -> str:
    if pid is None or total < 1e-6:
        return "rgb(128,128,128)"

    red = int(255 * pid.redundancy / total)
    blue = int(255 * pid.synergy / total)
    green = int(255 * (pid.unique_a + pid.unique_b) / total)

    total_color = red + blue + green
    if total_color > 0:
        scale = 255 / total_color
        red = int(red * scale)
        blue = int(blue * scale)
        green = int(green * scale)

    return f"rgb({red},{green},{blue})"


@typed
def hierarchical_layout(G: nx.DiGraph, root: int) -> dict[int, tuple[float, float]]:
    levels = {}

    def assign_levels(node: int, level: int):
        levels[node] = level
        for child in G.neighbors(node):
            assign_levels(child, level + 1)

    assign_levels(root, 0)

    leaves = [node for node in G.nodes() if len(list(G.neighbors(node))) == 0]
    leaf_positions = {leaf: float(i) for i, leaf in enumerate(leaves)}

    x_coords = {}

    def assign_x_coords(node: int):
        children = list(G.neighbors(node))
        if not children:
            x_coords[node] = leaf_positions[node]
        else:
            for child in children:
                assign_x_coords(child)
            child_x_coords = [x_coords[child] for child in children]
            x_coords[node] = sum(child_x_coords) / len(child_x_coords)

    assign_x_coords(root)

    pos = {}
    max_level = max(levels.values())
    for node in G.nodes():
        pos[node] = (x_coords[node], float(max_level - levels[node]) + np.random.rand())

    return pos


@typed
def render_tree_interactive(tree: MITree, output_file: str = "tree.html") -> None:
    G, node_map, _ = tree_to_networkx(tree)
    pos = hierarchical_layout(G, 0)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color="#888"), hoverinfo="none", mode="lines")

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    hover_text = []

    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)

        tree_node = node_map[node_id]
        color = pid_to_color(tree_node.pid, tree_node.pid.total if tree_node.pid else 0)
        node_colors.append(color)

        if tree_node.name:
            node_text.append(tree_node.name)
        else:
            node_text.append("")

        fmt = lambda x: f"{x:.3f}" if x is not None else "None"
        if tree_node.pid:
            mi_target, mi_join = fmt(tree_node.mi_target), fmt(tree_node.mi_join)
            pid_total, pid_redundancy = fmt(tree_node.pid.total), fmt(tree_node.pid.redundancy)
            pid_unique_a, pid_unique_b = fmt(tree_node.pid.unique_a), fmt(tree_node.pid.unique_b)
            pid_synergy = fmt(tree_node.pid.synergy)
            hover_info = f"Name: {tree_node.name or 'Internal'}<br>" f"Target MI: {mi_target}<br>" f"Join MI: {mi_join}<br>" f"PID Total: {pid_total}<br>" f"Redundancy: {pid_redundancy}<br>" f"Unique A: {pid_unique_a}<br>" f"Unique B: {pid_unique_b}<br>" f"Synergy: {pid_synergy}"
        else:
            mi_target = fmt(tree_node.mi_target)
            hover_info = f"Name: {tree_node.name or 'Internal'}<br>" f"Target MI: {mi_target}"
        hover_text.append(hover_info)

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", hoverinfo="text", hovertext=hover_text, text=node_text, textposition="bottom center", textfont=dict(color="black", size=14), marker=dict(size=20, color=node_colors, line=dict(width=2, color="black")))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f"Interactive MI Tree", showlegend=False, hovermode="closest", margin=dict(b=20, l=5, r=5, t=40), annotations=[dict(text="Red=Redundancy, Blue=Synergy, Green=Unique", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002, xanchor="left", yanchor="bottom", font=dict(size=12))], xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"Interactive tree saved to {output_file}")


def show_structure(X: pd.DataFrame, y: pd.Series, output_dir: str, q: int = 6):
    n, k = X.shape
    MIs = np.zeros((k, k))
    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()
    for i in range(k):
        for j in range(i + 1, k):
            MIs[i, j] = mi_quantile(X_numpy[:, i], X_numpy[:, j], q)
    MIs = MIs + MIs.T
    logger.info(f"MI matrix computed")
    plt.figure(figsize=(10, 8))
    sns.heatmap(MIs, annot=True, fmt=".3f", xticklabels=X.columns, yticklabels=X.columns, cmap="viridis")
    plt.title("MI Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mi_matrix.png"))
    plt.close()

    tree = build_mi_tree(X_numpy, y_numpy, q, names=list(X.columns))
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
