"""Canonical local origami constraint utilities.

These helpers are intentionally separate from ``line_graph_features.py``.
The line-GNN checkpoint was trained with the old feature definitions, so this
module is used for repair, diagnostics, and final validity reporting without
silently changing the trained classifier input distribution.
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import LineString


SCALE = 200.0
BOUNDARY_TOL = 3.0
EdgeKey = Tuple[int, int]


@dataclass
class ConstraintSummary:
    kawasaki: float
    maekawa: float
    max_kawasaki: float
    max_maekawa: float
    interior_vertices: int
    crossing_free: bool


def edge_key(u: int, v: int) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def coords(G: nx.Graph, node: int) -> Tuple[float, float]:
    return float(G.nodes[node]["x"]), float(G.nodes[node]["y"])


def bbox(G: nx.Graph) -> Tuple[float, float, float, float]:
    xs = [coords(G, n)[0] for n in G.nodes()]
    ys = [coords(G, n)[1] for n in G.nodes()]
    return min(xs), max(xs), min(ys), max(ys)


def is_boundary_node(
    G: nx.Graph,
    node: int,
    graph_bbox: Tuple[float, float, float, float] | None = None,
    tol: float = BOUNDARY_TOL,
) -> bool:
    if graph_bbox is None:
        graph_bbox = bbox(G)
    x, y = coords(G, node)
    min_x, max_x, min_y, max_y = graph_bbox
    return (
        abs(x - min_x) <= tol
        or abs(x - max_x) <= tol
        or abs(y - min_y) <= tol
        or abs(y - max_y) <= tol
    )


def non_border_neighbors(G: nx.Graph, node: int) -> List[int]:
    return [
        nb for nb in G.neighbors(node)
        if G[node][nb].get("fold_type") != 1
    ]


def non_border_degree(G: nx.Graph, node: int) -> int:
    return len(non_border_neighbors(G, node))


def is_interior_vertex(
    G: nx.Graph,
    node: int,
    graph_bbox: Tuple[float, float, float, float] | None = None,
) -> bool:
    if graph_bbox is None:
        graph_bbox = bbox(G)
    if is_boundary_node(G, node, graph_bbox):
        return False
    return non_border_degree(G, node) >= 2


def ordered_neighbors(
    G: nx.Graph,
    node: int,
    *,
    non_border_only: bool = True,
) -> List[int]:
    neighbors = non_border_neighbors(G, node) if non_border_only else list(G.neighbors(node))
    cx, cy = coords(G, node)
    return sorted(
        neighbors,
        key=lambda nb: math.atan2(coords(G, nb)[1] - cy, coords(G, nb)[0] - cx),
    )


def angle_gaps(
    G: nx.Graph,
    node: int,
    *,
    non_border_only: bool = True,
) -> List[float]:
    neighbors = ordered_neighbors(G, node, non_border_only=non_border_only)
    if len(neighbors) < 2:
        return []
    cx, cy = coords(G, node)
    angles = [
        math.atan2(coords(G, nb)[1] - cy, coords(G, nb)[0] - cx)
        for nb in neighbors
    ]
    gaps = [
        angles[(i + 1) % len(angles)] - angles[i]
        for i in range(len(angles))
    ]
    return [gap + 2 * math.pi if gap < 0 else gap for gap in gaps]


def kawasaki_at(G: nx.Graph, node: int) -> float:
    graph_bbox = bbox(G)
    if not is_interior_vertex(G, node, graph_bbox):
        return 0.0
    degree = non_border_degree(G, node)
    if degree < 4:
        return 0.0
    if degree % 2 == 1:
        return 10.0

    gaps = angle_gaps(G, node, non_border_only=True)
    if not gaps:
        return 0.0
    even_sum = sum(gaps[i] for i in range(0, len(gaps), 2))
    odd_sum = sum(gaps[i] for i in range(1, len(gaps), 2))
    return float(abs(even_sum - math.pi) + abs(odd_sum - math.pi))


def maekawa_at(G: nx.Graph, node: int) -> float:
    graph_bbox = bbox(G)
    if not is_interior_vertex(G, node, graph_bbox):
        return 0.0

    neighbors = non_border_neighbors(G, node)
    degree = len(neighbors)
    if degree < 2:
        return 0.0
    if degree % 2 == 1:
        return 10.0

    mountain = sum(1 for nb in neighbors if G[node][nb].get("fold_type") == 2)
    valley = sum(1 for nb in neighbors if G[node][nb].get("fold_type") == 3)
    return float(abs(abs(mountain - valley) - 2))


def _nonzero_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) > 1e-9]
    return float(np.mean(vals)) if vals else 0.0


def kawasaki_penalty(G: nx.Graph) -> float:
    return _nonzero_mean(kawasaki_at(G, node) for node in G.nodes())


def maekawa_penalty(G: nx.Graph) -> float:
    return _nonzero_mean(maekawa_at(G, node) for node in G.nodes())


def recompute_features(G: nx.Graph) -> nx.Graph:
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        cx, cy = coords(G, node)
        angles = sorted(
            math.atan2(coords(G, nb)[1] - cy, coords(G, nb)[0] - cx)
            for nb in neighbors
        )
        G.nodes[node]["degree"] = len(neighbors)
        G.nodes[node]["angles"] = angles
    return G


def has_crossings(G: nx.Graph) -> bool:
    edges = list(G.edges())
    segments = [LineString([coords(G, u), coords(G, v)]) for u, v in edges]
    for i, (u1, v1) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            if len({u1, v1, u2, v2}) < 4:
                continue
            if segments[i].crosses(segments[j]):
                return True
    return False


def constraint_summary(G: nx.Graph) -> ConstraintSummary:
    kaw_values = [kawasaki_at(G, node) for node in G.nodes()]
    mae_values = [maekawa_at(G, node) for node in G.nodes()]
    graph_bbox = bbox(G)
    interior = [
        node for node in G.nodes()
        if is_interior_vertex(G, node, graph_bbox)
    ]
    return ConstraintSummary(
        kawasaki=_nonzero_mean(kaw_values),
        maekawa=_nonzero_mean(mae_values),
        max_kawasaki=max(kaw_values) if kaw_values else 0.0,
        max_maekawa=max(mae_values) if mae_values else 0.0,
        interior_vertices=len(interior),
        crossing_free=not has_crossings(G),
    )


def worst_vertices(G: nx.Graph, k: int = 8, metric: str = "kawasaki") -> List[Tuple[int, float]]:
    if metric == "maekawa":
        scorer = maekawa_at
    elif metric == "combined":
        scorer = lambda graph, node: kawasaki_at(graph, node) + maekawa_at(graph, node)
    else:
        scorer = kawasaki_at

    graph_bbox = bbox(G)
    scored = [
        (node, float(scorer(G, node)))
        for node in G.nodes()
        if is_interior_vertex(G, node, graph_bbox)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def constraint_rows(G: nx.Graph) -> List[dict[str, object]]:
    graph_bbox = bbox(G)
    rows = []
    for node in G.nodes():
        if not is_interior_vertex(G, node, graph_bbox):
            continue
        folds = [G[node][nb].get("fold_type") for nb in non_border_neighbors(G, node)]
        rows.append({
            "node": node,
            "non_border_degree": non_border_degree(G, node),
            "kawasaki": kawasaki_at(G, node),
            "maekawa": maekawa_at(G, node),
            "folds": folds,
            "gaps_degrees": [
                round(gap * 180.0 / math.pi, 2)
                for gap in angle_gaps(G, node, non_border_only=True)
            ],
        })
    return rows


def print_constraint_report(G: nx.Graph, label: str = "", limit: int | None = None) -> None:
    if label:
        print(f"\n--- {label} ---")
    rows = constraint_rows(G)
    rows.sort(key=lambda row: float(row["kawasaki"]) + float(row["maekawa"]), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        print(
            "node", row["node"],
            "deg", row["non_border_degree"],
            "kaw", round(float(row["kawasaki"]), 4),
            "mae", round(float(row["maekawa"]), 4),
            "folds", row["folds"],
            "gaps_deg", row["gaps_degrees"],
        )


def graph_signature(G: nx.Graph) -> np.ndarray:
    degrees = [degree for _, degree in G.degree()]
    degree_hist = np.histogram(degrees, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 20])[0].astype(float)

    lengths = []
    angles = []
    for u, v in G.edges():
        x1, y1 = coords(G, u)
        x2, y2 = coords(G, v)
        dx, dy = x2 - x1, y2 - y1
        lengths.append(math.hypot(dx, dy))
        angles.append((math.atan2(dy, dx) + math.pi) % math.pi)

    length_hist = np.histogram(lengths, bins=10, range=(0.0, 400.0))[0].astype(float)
    angle_hist = np.histogram(angles, bins=12, range=(0.0, math.pi))[0].astype(float)
    kaw_values = [min(kawasaki_at(G, node), 3.0) for node in G.nodes()]
    kaw_hist = np.histogram(kaw_values, bins=8, range=(0.0, 3.0))[0].astype(float)

    signature = np.concatenate([degree_hist, length_hist, angle_hist, kaw_hist])
    total = np.linalg.norm(signature)
    return signature / total if total > 0 else signature


def graph_similarity(G1: nx.Graph, G2: nx.Graph) -> float:
    v1 = graph_signature(G1)
    v2 = graph_signature(G2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


def visualise_with_violations(G: nx.Graph, title: str = "", ax=None):
    import matplotlib.pyplot as plt

    pos = {node: coords(G, node) for node in G.nodes()}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for fold_type, color, width in [(2, "red", 1.2), (3, "blue", 0.8), (1, "black", 2.0)]:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (u, v) for u, v, data in G.edges(data=True)
                if data.get("fold_type") == fold_type
            ],
            edge_color=color,
            width=width,
            ax=ax,
        )

    values = [
        kawasaki_at(G, node) + maekawa_at(G, node)
        for node in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[20 + 120 * min(value, 2.0) for value in values],
        node_color=values,
        cmap=plt.cm.Reds,
        ax=ax,
    )
    ax.set_title(title, fontsize=8)
    ax.axis("equal")
    ax.axis("off")
    return ax


def _load_graphs(path: str):
    with open(path, "rb") as handle:
        item = pickle.load(handle)
    return item if isinstance(item, list) else [item]


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python origami_constraints.py <graph-or-list.pkl> [...]")
        return 1
    for path in argv[1:]:
        graphs = _load_graphs(path)
        print(f"\n{path}: {len(graphs)} graph(s)")
        for i, graph in enumerate(graphs):
            summary = constraint_summary(graph)
            print(
                f"  [{i}] kaw={summary.kawasaki:.4f} mae={summary.maekawa:.4f} "
                f"max_kaw={summary.max_kawasaki:.4f} max_mae={summary.max_maekawa:.4f} "
                f"interior={summary.interior_vertices} crossing_free={summary.crossing_free}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
