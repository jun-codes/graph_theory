"""
Topology repair for Maekawa-satisfiable crease-pattern candidates.

Z3 can repair mountain/valley labels only when every true interior vertex has
an even non-border crease degree of at least 2.  This module adjusts topology
before Z3 by toggling parity at odd interior vertices:

- remove non-border odd-odd creases when both endpoints stay degree >= 2
- add short non-crossing odd-odd creases for remaining odd vertices

Boundary vertices are excluded because Maekawa is an interior-vertex theorem.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import List, Tuple

import networkx as nx
from shapely.geometry import LineString


BOUNDARY_TOL = 3.0
EdgeKey = Tuple[int, int]


@dataclass
class TopologyRepairStats:
    status: str
    before_bad_vertices: int
    after_bad_vertices: int
    added_edges: int
    removed_edges: int
    attempts: int
    reason: str


def _edge_key(u: int, v: int) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def _coords(G: nx.Graph, node: int) -> Tuple[float, float]:
    return float(G.nodes[node]["x"]), float(G.nodes[node]["y"])


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


def _is_true_interior_vertex(
    G: nx.Graph,
    node: int,
    bbox: Tuple[float, float, float, float],
) -> bool:
    if _is_boundary_node(G, node, bbox):
        return False
    neighbors = list(G.neighbors(node))
    return len(neighbors) >= 2 and not all(
        G[node][nb].get("fold_type") == 1 for nb in neighbors
    )


def non_border_degree(G: nx.Graph, node: int) -> int:
    return sum(
        1 for nb in G.neighbors(node)
        if G[node][nb].get("fold_type") != 1
    )


def bad_maekawa_topology_vertices(G: nx.Graph) -> List[int]:
    bbox = _bbox(G)
    bad = []
    for node in G.nodes():
        if not _is_true_interior_vertex(G, node, bbox):
            continue
        degree = non_border_degree(G, node)
        if degree < 2 or degree % 2 == 1:
            bad.append(node)
    return bad


def _segment(G: nx.Graph, u: int, v: int) -> LineString:
    return LineString([_coords(G, u), _coords(G, v)])


def _edge_crosses_any(G: nx.Graph, u: int, v: int) -> bool:
    new_seg = _segment(G, u, v)
    for a, b in G.edges():
        if len({u, v, a, b}) < 4:
            continue
        if new_seg.crosses(_segment(G, a, b)):
            return True
    return False


def _would_remain_connected_after_remove(G: nx.Graph, u: int, v: int) -> bool:
    if not nx.is_connected(G):
        return True
    H = G.copy()
    H.remove_edge(u, v)
    return nx.is_connected(H)


def _try_remove_odd_odd_edges(G: nx.Graph, bad: List[int]) -> int:
    bad_set = set(bad)
    candidates = []
    for u, v, data in G.edges(data=True):
        if data.get("fold_type") == 1:
            continue
        if u not in bad_set or v not in bad_set:
            continue
        if non_border_degree(G, u) <= 2 or non_border_degree(G, v) <= 2:
            continue
        ux, uy = _coords(G, u)
        vx, vy = _coords(G, v)
        candidates.append((math.hypot(ux - vx, uy - vy), u, v))

    removed = 0
    for _, u, v in sorted(candidates):
        bad_set = set(bad_maekawa_topology_vertices(G))
        if u not in bad_set or v not in bad_set:
            continue
        if non_border_degree(G, u) <= 2 or non_border_degree(G, v) <= 2:
            continue
        if not _would_remain_connected_after_remove(G, u, v):
            continue
        G.remove_edge(u, v)
        removed += 1
    return removed


def _try_remove_odd_boundary_edges(G: nx.Graph, bad: List[int]) -> int:
    bbox = _bbox(G)
    removed = 0
    for node in list(bad):
        if node not in bad_maekawa_topology_vertices(G):
            continue
        if non_border_degree(G, node) <= 2:
            continue

        candidates = []
        nx_, ny_ = _coords(G, node)
        for nb in G.neighbors(node):
            if G[node][nb].get("fold_type") == 1:
                continue
            if not _is_boundary_node(G, nb, bbox):
                continue
            bx, by = _coords(G, nb)
            candidates.append((math.hypot(nx_ - bx, ny_ - by), nb))

        for _, nb in sorted(candidates):
            if not G.has_edge(node, nb):
                continue
            if not _would_remain_connected_after_remove(G, node, nb):
                continue
            G.remove_edge(node, nb)
            removed += 1
            break
    return removed


def _try_add_odd_odd_edges(G: nx.Graph, bad: List[int], max_added: int) -> int:
    added = 0
    while added < max_added:
        bad = bad_maekawa_topology_vertices(G)
        if len(bad) < 2:
            break

        candidates = []
        for i, u in enumerate(bad):
            ux, uy = _coords(G, u)
            for v in bad[i + 1:]:
                if G.has_edge(u, v):
                    continue
                vx, vy = _coords(G, v)
                dist = math.hypot(ux - vx, uy - vy)
                candidates.append((dist, u, v))

        placed = False
        for _, u, v in sorted(candidates):
            if _edge_crosses_any(G, u, v):
                continue
            G.add_edge(u, v, fold_type=2)
            added += 1
            placed = True
            break

        if not placed:
            break
    return added


def _recompute_features(G: nx.Graph) -> nx.Graph:
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        cx, cy = _coords(G, node)
        angles = sorted(
            math.atan2(_coords(G, nb)[1] - cy, _coords(G, nb)[0] - cx)
            for nb in neighbors
        )
        G.nodes[node]["degree"] = len(neighbors)
        G.nodes[node]["angles"] = angles
    return G


def repair_even_nonborder_topology(
    G: nx.Graph,
    *,
    max_rounds: int = 4,
    max_added_edges: int = 24,
) -> tuple[nx.Graph, TopologyRepairStats]:
    repaired = copy.deepcopy(G)
    before_bad = len(bad_maekawa_topology_vertices(repaired))
    added = 0
    removed = 0
    attempts = 0

    for _ in range(max_rounds):
        bad = bad_maekawa_topology_vertices(repaired)
        if not bad:
            break
        attempts += 1
        removed_now = _try_remove_odd_odd_edges(repaired, bad)
        removed_now += _try_remove_odd_boundary_edges(
            repaired, bad_maekawa_topology_vertices(repaired))
        removed += removed_now

        bad = bad_maekawa_topology_vertices(repaired)
        if not bad:
            break
        added_now = _try_add_odd_odd_edges(
            repaired, bad, max_added=max_added_edges - added)
        added += added_now

        if removed_now == 0 and added_now == 0:
            break

    repaired = _recompute_features(repaired)
    after_bad = len(bad_maekawa_topology_vertices(repaired))
    status = "repaired" if after_bad == 0 else "partial"
    if before_bad == 0 and after_bad == 0:
        status = "skipped"

    reason = ""
    if after_bad:
        sample = bad_maekawa_topology_vertices(repaired)[:12]
        reason = ", ".join(
            f"{node}:d{non_border_degree(repaired, node)}" for node in sample
        )
        remaining = after_bad - len(sample)
        if remaining > 0:
            reason += f", ... +{remaining} more"

    return repaired, TopologyRepairStats(
        status=status,
        before_bad_vertices=before_bad,
        after_bad_vertices=after_bad,
        added_edges=added,
        removed_edges=removed,
        attempts=attempts,
        reason=reason,
    )
