from dataclasses import dataclass

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from beartype import beartype as typed
from beartype.typing import Self


@dataclass
class InfoDecomposition:
    mi_joint: float
    mi_additive: float
    mi_a: float
    mi_b: float


class MITree:
    mi_target: float
    mi_join: float | None = None
    decomp: InfoDecomposition | None = None
    left: Self | None = None
    right: Self | None = None
    name: str | None = None

    def __init__(self, mi_target: float, name: str | None = None, mi_join: float | None = None, decomp: InfoDecomposition | None = None, left: Self | None = None, right: Self | None = None):
        self.mi_target = mi_target
        self.name = name
        self.decomp = decomp
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
        c, d = fmt(self.decomp.mi_joint), fmt(self.decomp.mi_additive)
        e, f = fmt(self.decomp.mi_a), fmt(self.decomp.mi_b)
        stats = [
            f"---(target: {a} | join: {b})",
            f"    (total: {c} |  add: {d})",
            f"        (A: {e} |    B: {f})",
        ]
        stats.extend(["|" for _ in range(len(left_lines) - len(stats))])
        stats.extend(["" for _ in range(len(right_lines))])
        stats_width = max(len(line) for line in stats)
        stats = [line.rjust(stats_width) for line in stats]
        assert len(stats) == len(left_lines) + len(right_lines)
        result_lines = [x + y for x, y in zip(stats, left_lines + right_lines)]
        return "\n".join(result_lines)


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
def pid_to_color(pid: InfoDecomposition | None) -> str:
    if pid is None:
        return "rgb(128,128,128)"

    baseline = max(max(pid.mi_a, pid.mi_b), 0)
    additive_synergy = max(pid.mi_additive - baseline, 0)
    joint_synergy = max(pid.mi_joint - pid.mi_additive, 0)

    red = int(255 * baseline)
    green = int(255 * additive_synergy)
    blue = int(255 * joint_synergy)

    total_color = red + blue + green + 1
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
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}

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
        color = pid_to_color(tree_node.decomp)
        node_colors.append(color)

        if tree_node.name:
            node_text.append(tree_node.name)
        else:
            node_text.append("")

        fmt = lambda x: f"{x:.3f}" if x is not None else "None"
        if tree_node.decomp:
            mi_target, mi_join = fmt(tree_node.mi_target), fmt(tree_node.mi_join)
            mi_joint, mi_additive = fmt(tree_node.decomp.mi_joint), fmt(tree_node.decomp.mi_additive)
            mi_a, mi_b = fmt(tree_node.decomp.mi_a), fmt(tree_node.decomp.mi_b)
            hover_info = f"Name: {tree_node.name or 'Internal'}<br>" f"Target MI: {mi_target}<br>" f"Join MI: {mi_join}<br>" f"MI Joint: {mi_joint}<br>" f"MI Additive: {mi_additive}<br>" f"MI A: {mi_a}<br>" f"MI B: {mi_b}"
        else:
            mi_target = fmt(tree_node.mi_target)
            hover_info = f"Name: {tree_node.name or 'Internal'}<br>" f"Target MI: {mi_target}"
        hover_text.append(hover_info)

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", hoverinfo="text", hovertext=hover_text, text=node_text, textposition="bottom center", textfont=dict(color="black", size=14), marker=dict(size=20, color=node_colors, line=dict(width=2, color="black")))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f"Interactive MI Tree", showlegend=False, hovermode="closest", margin=dict(b=20, l=5, r=5, t=40), annotations=[dict(text="Red=Redundancy, Blue=Synergy, Green=Unique", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002, xanchor="left", yanchor="bottom", font=dict(size=12))], xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"Interactive tree saved to {output_file}")
