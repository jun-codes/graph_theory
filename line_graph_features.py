"""
Line-graph features for crease-pattern classification.

This module keeps the crease-first representation separate from the existing
vertex-GNN pipeline.  It converts an original crease-pattern graph G, where
nodes are CP vertices and edges are creases, into a line graph where each node
is one crease.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np


SCALE = 200.0
BOUNDARY_TOL = 3.0


LINE_NODE_FEATURE_NAMES = [
    "fold_border",
    "fold_mountain",
    "fold_valley",
    "fold_unknown",
    "length_norm",
    "orient_cos2",
    "orient_sin2",
    "mid_x_norm",
    "mid_y_norm",
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "is_border_fold",
    "touches_boundary",
    "endpoint_a_degree_norm",
    "endpoint_b_degree_norm",
    "endpoint_a_even_degree",
    "endpoint_b_even_degree",
    "endpoint_a_kaw_norm",
    "endpoint_b_kaw_norm",
    "endpoint_a_mae_norm",
    "endpoint_b_mae_norm",
    "endpoint_a_boundary",
    "endpoint_b_boundary",
]


LINE_EDGE_FEATURE_NAMES = [
    "angle_norm",
    "angle_cos",
    "angle_sin",
    "is_consecutive",
    "same_fold_type",
    "opposite_mv",
    "shared_boundary",
    "shared_degree_norm",
    "shared_kaw_norm",
    "shared_mae_norm",
]


EdgeKey = Tuple[int, int]


@dataclass(frozen=True)
class LineGraphArrays:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_keys: List[EdgeKey]
    y: Optional[int] = None


def _edge_key(u: int, v: int) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def _coords(G: nx.Graph, node: int) -> Tuple[float, float]:
    attrs = G.nodes[node]
    return float(attrs["x"]), float(attrs["y"])


def _bbox(G: nx.Graph) -> Tuple[float, float, float, float]:
    xs = [float(G.nodes[n]["x"]) for n in G.nodes()]
    ys = [float(G.nodes[n]["y"]) for n in G.nodes()]
    return min(xs), max(xs), min(ys), max(ys)


def _is_boundary_node(
    G: nx.Graph,
    node: int,
    bbox: Tuple[float, float, float, float],
    tol: float = BOUNDARY_TOL,
) -> bool:
    x, y = _coords(G, node)
    min_x, max_x, min_y, max_y = bbox
    return (
        abs(x - min_x) <= tol
        or abs(x - max_x) <= tol
        or abs(y - min_y) <= tol
        or abs(y - max_y) <= tol
    )


def _angle_gaps(G: nx.Graph, node: int) -> List[float]:
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return []

    cx, cy = _coords(G, node)
    angles = sorted(
        math.atan2(_coords(G, nb)[1] - cy, _coords(G, nb)[0] - cx)
        for nb in neighbors
    )
    gaps = [angles[(i + 1) % len(angles)] - angles[i] for i in range(len(angles))]
    return [gap + 2 * math.pi if gap < 0 else gap for gap in gaps]


def kawasaki_violation(G: nx.Graph, node: int) -> float:
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get("fold_type") == 1 for nb in nbs):
        return 0.0

    gaps = _angle_gaps(G, node)
    if not gaps:
        return 0.0

    even_sum = sum(gaps[i] for i in range(0, len(gaps), 2))
    odd_sum = sum(gaps[i] for i in range(1, len(gaps), 2))
    return float(abs(even_sum - math.pi) + abs(odd_sum - math.pi))


def maekawa_violation(G: nx.Graph, node: int) -> float:
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get("fold_type") == 1 for nb in nbs):
        return 0.0

    mountain = sum(1 for nb in nbs if G[node][nb].get("fold_type") == 2)
    valley = sum(1 for nb in nbs if G[node][nb].get("fold_type") == 3)
    return float(abs(abs(mountain - valley) - 2))


def _fold_one_hot(fold_type: int) -> List[float]:
    return [
        1.0 if fold_type == 1 else 0.0,
        1.0 if fold_type == 2 else 0.0,
        1.0 if fold_type == 3 else 0.0,
        1.0 if fold_type not in (1, 2, 3) else 0.0,
    ]


def _canonical_endpoint_order(G: nx.Graph, u: int, v: int) -> Tuple[int, int]:
    ux, uy = _coords(G, u)
    vx, vy = _coords(G, v)
    return (u, v) if (ux, uy, u) <= (vx, vy, v) else (v, u)


def _crease_node_features(
    G: nx.Graph,
    edge: EdgeKey,
    bbox: Tuple[float, float, float, float],
    scale: float,
) -> List[float]:
    u, v = edge
    a, b = _canonical_endpoint_order(G, u, v)
    ax, ay = _coords(G, a)
    bx, by = _coords(G, b)
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    fold_type = int(G[u][v].get("fold_type", 0))

    deg_a = G.degree(a)
    deg_b = G.degree(b)
    a_boundary = _is_boundary_node(G, a, bbox)
    b_boundary = _is_boundary_node(G, b, bbox)

    return (
        _fold_one_hot(fold_type)
        + [
            length / scale,
            math.cos(2 * theta),
            math.sin(2 * theta),
            ((ax + bx) * 0.5) / scale,
            ((ay + by) * 0.5) / scale,
            ax / scale,
            ay / scale,
            bx / scale,
            by / scale,
            1.0 if fold_type == 1 else 0.0,
            1.0 if (a_boundary or b_boundary) else 0.0,
            deg_a / 10.0,
            deg_b / 10.0,
            1.0 if deg_a % 2 == 0 else 0.0,
            1.0 if deg_b % 2 == 0 else 0.0,
            kawasaki_violation(G, a) / math.pi,
            kawasaki_violation(G, b) / math.pi,
            maekawa_violation(G, a) / 4.0,
            maekawa_violation(G, b) / 4.0,
            1.0 if a_boundary else 0.0,
            1.0 if b_boundary else 0.0,
        ]
    )


def _angle_from_shared_vertex(G: nx.Graph, shared: int, edge: EdgeKey) -> float:
    u, v = edge
    other = v if shared == u else u
    sx, sy = _coords(G, shared)
    ox, oy = _coords(G, other)
    return math.atan2(oy - sy, ox - sx)


def _small_angle(a: float, b: float) -> float:
    diff = abs(a - b)
    return 2 * math.pi - diff if diff > math.pi else diff


def _incident_edge_keys(G: nx.Graph, node: int) -> List[EdgeKey]:
    return [_edge_key(node, nb) for nb in G.neighbors(node)]


def _consecutive_pairs(G: nx.Graph) -> set[Tuple[int, EdgeKey, EdgeKey]]:
    consecutive = set()
    for node in G.nodes():
        incident = _incident_edge_keys(G, node)
        if len(incident) < 2:
            continue
        incident.sort(key=lambda edge: _angle_from_shared_vertex(G, node, edge))
        for i, edge_a in enumerate(incident):
            edge_b = incident[(i + 1) % len(incident)]
            pair = tuple(sorted((edge_a, edge_b)))
            consecutive.add((node, pair[0], pair[1]))
    return consecutive


def _line_edge_features(
    G: nx.Graph,
    shared: int,
    edge_a: EdgeKey,
    edge_b: EdgeKey,
    bbox: Tuple[float, float, float, float],
    consecutive: set[Tuple[int, EdgeKey, EdgeKey]],
) -> List[float]:
    angle_a = _angle_from_shared_vertex(G, shared, edge_a)
    angle_b = _angle_from_shared_vertex(G, shared, edge_b)
    angle = _small_angle(angle_a, angle_b)
    fold_a = int(G[edge_a[0]][edge_a[1]].get("fold_type", 0))
    fold_b = int(G[edge_b[0]][edge_b[1]].get("fold_type", 0))
    pair = tuple(sorted((edge_a, edge_b)))

    return [
        angle / math.pi,
        math.cos(angle),
        math.sin(angle),
        1.0 if (shared, pair[0], pair[1]) in consecutive else 0.0,
        1.0 if fold_a == fold_b else 0.0,
        1.0 if {fold_a, fold_b} == {2, 3} else 0.0,
        1.0 if _is_boundary_node(G, shared, bbox) else 0.0,
        G.degree(shared) / 10.0,
        kawasaki_violation(G, shared) / math.pi,
        maekawa_violation(G, shared) / 4.0,
    ]


def _iter_line_edges(G: nx.Graph) -> Iterable[Tuple[int, EdgeKey, EdgeKey]]:
    for shared in G.nodes():
        incident = _incident_edge_keys(G, shared)
        for i, edge_a in enumerate(incident):
            for edge_b in incident[i + 1 :]:
                yield shared, edge_a, edge_b


def cp_to_line_graph_arrays(
    G: nx.Graph,
    label: Optional[int] = None,
    scale: float = SCALE,
) -> Optional[LineGraphArrays]:
    if G.number_of_edges() == 0:
        return None

    bbox = _bbox(G)
    edge_keys = sorted(_edge_key(u, v) for u, v in G.edges())
    edge_to_idx: Dict[EdgeKey, int] = {edge: i for i, edge in enumerate(edge_keys)}

    x = np.asarray(
        [_crease_node_features(G, edge, bbox, scale) for edge in edge_keys],
        dtype=np.float32,
    )

    consecutive = _consecutive_pairs(G)
    directed_edges: List[Tuple[int, int]] = []
    directed_attrs: List[List[float]] = []

    for shared, edge_a, edge_b in _iter_line_edges(G):
        idx_a = edge_to_idx[edge_a]
        idx_b = edge_to_idx[edge_b]
        attr = _line_edge_features(G, shared, edge_a, edge_b, bbox, consecutive)
        directed_edges.append((idx_a, idx_b))
        directed_attrs.append(attr)
        directed_edges.append((idx_b, idx_a))
        directed_attrs.append(attr)

    if directed_edges:
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        edge_attr = np.asarray(directed_attrs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, len(LINE_EDGE_FEATURE_NAMES)), dtype=np.float32)

    return LineGraphArrays(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_keys=edge_keys,
        y=label,
    )


def line_graph_to_pyg(G: nx.Graph, label: int, scale: float = SCALE):
    arrays = cp_to_line_graph_arrays(G, label=label, scale=scale)
    if arrays is None:
        return None

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "line_graph_to_pyg requires torch and torch_geometric. "
            "Use cp_to_line_graph_arrays for dependency-light conversion."
        ) from exc

    data = Data(
        x=torch.tensor(arrays.x, dtype=torch.float),
        edge_index=torch.tensor(arrays.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(arrays.edge_attr, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
    )
    data.num_original_nodes = G.number_of_nodes()
    data.num_original_edges = G.number_of_edges()
    return data
