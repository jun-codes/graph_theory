"""Direct Kawasaki sector-angle projection repair.

The gradient repair can reduce Kawasaki residuals, but it still searches in
coordinate space indirectly.  This module attacks the local theorem directly:
for each bad even-degree interior vertex, project its alternating sector gaps
onto two simplexes whose sums are both pi, then move mutable neighboring nodes
toward the projected rays.
"""

from __future__ import annotations

import copy
import math
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

import origami_constraints as oc


@dataclass
class KawasakiProjectionStats:
    accepted: bool
    attempted_vertices: int
    accepted_vertices: int
    before_kawasaki: float
    after_kawasaki: float
    before_objective: float
    after_objective: float
    crossing_rejections: int
    reason: str


def _project_simplex_with_lower(values: np.ndarray, total: float, lower: float) -> np.ndarray:
    """Project values onto sum(x)=total, x>=lower."""
    n = len(values)
    if n == 0:
        return values
    if total <= 0:
        return np.zeros(n, dtype=float)

    lower = min(lower, total / n * 0.95)
    shifted = values.astype(float) - lower
    target_sum = total - n * lower
    if target_sum <= 0:
        return np.full(n, total / n, dtype=float)

    u = np.sort(shifted)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - target_sum)
    if not np.any(rho_candidates):
        theta = 0.0
    else:
        rho = int(np.nonzero(rho_candidates)[0][-1])
        theta = (cssv[rho] - target_sum) / float(rho + 1)
    projected = np.maximum(shifted - theta, 0.0) + lower
    return projected


def project_kawasaki_gaps(
    gaps: Iterable[float],
    *,
    min_gap_deg: float = 8.0,
) -> List[float]:
    """Return same-length gaps whose even and odd alternating sums are pi."""
    gaps_arr = np.array(list(gaps), dtype=float)
    n = len(gaps_arr)
    if n < 4 or n % 2 == 1:
        return gaps_arr.tolist()

    min_gap = math.radians(min_gap_deg)
    target = np.array(gaps_arr, dtype=float)
    even_idx = np.arange(0, n, 2)
    odd_idx = np.arange(1, n, 2)
    target[even_idx] = _project_simplex_with_lower(target[even_idx], math.pi, min_gap)
    target[odd_idx] = _project_simplex_with_lower(target[odd_idx], math.pi, min_gap)

    scale = (2.0 * math.pi) / float(np.sum(target))
    return (target * scale).tolist()


def _angles_and_neighbors(G: nx.Graph, node: int) -> Tuple[List[int], List[float]]:
    neighbors = oc.ordered_neighbors(G, node, non_border_only=True)
    cx, cy = oc.coords(G, node)
    angles = [
        math.atan2(oc.coords(G, nb)[1] - cy, oc.coords(G, nb)[0] - cx)
        for nb in neighbors
    ]
    return neighbors, angles


def _unwrap_sorted_angles(angles: List[float]) -> List[float]:
    if not angles:
        return []
    out = [angles[0]]
    for angle in angles[1:]:
        while angle < out[-1]:
            angle += 2.0 * math.pi
        out.append(angle)
    return out


def _target_positions_for_vertex(
    G: nx.Graph,
    node: int,
    *,
    min_gap_deg: float,
) -> Dict[int, Tuple[float, float]]:
    neighbors, angles = _angles_and_neighbors(G, node)
    degree = len(neighbors)
    if degree < 4 or degree % 2 == 1:
        return {}

    gaps = oc.angle_gaps(G, node, non_border_only=True)
    projected = project_kawasaki_gaps(gaps, min_gap_deg=min_gap_deg)
    if len(projected) != degree:
        return {}

    unwrapped = _unwrap_sorted_angles(angles)
    relative = [0.0]
    for gap in projected[:-1]:
        relative.append(relative[-1] + gap)

    # Least-squares angular offset preserving cyclic neighbor order.
    offset = float(np.mean(np.array(unwrapped) - np.array(relative)))
    target_angles = [offset + rel for rel in relative]

    cx, cy = oc.coords(G, node)
    targets: Dict[int, Tuple[float, float]] = {}
    for nb, target_angle in zip(neighbors, target_angles):
        nx_, ny_ = oc.coords(G, nb)
        radius = max(5.0, math.hypot(nx_ - cx, ny_ - cy))
        targets[nb] = (
            cx + radius * math.cos(target_angle),
            cy + radius * math.sin(target_angle),
        )
    return targets


def _bounds(G: nx.Graph, inset: float = 20.0) -> Tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = oc.bbox(G)
    return min_x + inset, max_x - inset, min_y + inset, max_y - inset


def _apply_proposals(
    G: nx.Graph,
    proposals: Dict[int, List[Tuple[float, float]]],
    *,
    step: float,
    max_move: float,
) -> nx.Graph:
    out = copy.deepcopy(G)
    graph_bbox = oc.bbox(G)
    min_x, max_x, min_y, max_y = _bounds(G)

    for node, targets in proposals.items():
        if not targets or oc.is_boundary_node(G, node, graph_bbox):
            continue
        ox, oy = oc.coords(G, node)
        tx = float(np.mean([target[0] for target in targets]))
        ty = float(np.mean([target[1] for target in targets]))
        dx = (tx - ox) * step
        dy = (ty - oy) * step
        dist = math.hypot(dx, dy)
        if dist > max_move:
            scale = max_move / dist
            dx *= scale
            dy *= scale
        out.nodes[node]["x"] = float(np.clip(ox + dx, min_x, max_x))
        out.nodes[node]["y"] = float(np.clip(oy + dy, min_y, max_y))

    return oc.recompute_features(out)


def _objective(G: nx.Graph) -> float:
    return oc.kawasaki_penalty(G)


def _local_objective(G: nx.Graph, nodes: Iterable[int]) -> float:
    expanded = set(nodes)
    for node in list(expanded):
        if node in G:
            expanded.update(G.neighbors(node))
    return float(sum(oc.kawasaki_at(G, node) for node in expanded if node in G))


def repair_kawasaki_projection(
    G: nx.Graph,
    *,
    passes: int = 8,
    top_k: int = 10,
    step: float = 0.45,
    max_move: float = 14.0,
    min_gap_deg: float = 8.0,
    min_improvement: float = 1e-5,
    reject_crossings: bool = True,
) -> Tuple[nx.Graph, KawasakiProjectionStats]:
    before_kaw = oc.kawasaki_penalty(G)
    before_obj = _objective(G)
    if before_kaw <= min_improvement:
        return G, KawasakiProjectionStats(
            accepted=False,
            attempted_vertices=0,
            accepted_vertices=0,
            before_kawasaki=before_kaw,
            after_kawasaki=before_kaw,
            before_objective=before_obj,
            after_objective=before_obj,
            crossing_rejections=0,
            reason="already valid",
        )

    best = copy.deepcopy(G)
    best_obj = before_obj
    attempted_vertices = 0
    accepted_vertices = 0
    crossing_rejections = 0

    for _ in range(passes):
        worst = [
            (node, value)
            for node, value in oc.worst_vertices(best, k=top_k, metric="kawasaki")
            if value > min_improvement
        ]
        if not worst:
            break

        accepted_this_pass = False
        for node, _ in worst:
            targets = _target_positions_for_vertex(
                best,
                node,
                min_gap_deg=min_gap_deg,
            )
            if not targets:
                continue

            attempted_vertices += 1
            proposals = {nb: [target] for nb, target in targets.items()}
            affected = {node, *targets.keys()}
            before_local = _local_objective(best, affected)

            for shrink in (1.0, 0.5, 0.25, 0.125):
                trial = _apply_proposals(
                    best,
                    proposals,
                    step=step * shrink,
                    max_move=max_move * shrink,
                )
                if reject_crossings and oc.has_crossings(trial):
                    crossing_rejections += 1
                    continue
                trial_obj = _objective(trial)
                after_local = _local_objective(trial, affected)
                local_ok = after_local + min_improvement < before_local
                global_ok = trial_obj <= best_obj + 0.02
                if local_ok and global_ok:
                    best = trial
                    best_obj = trial_obj
                    accepted_vertices += 1
                    accepted_this_pass = True
                    break

        if not accepted_this_pass:
            break

    after_kaw = oc.kawasaki_penalty(best)
    accepted = after_kaw + min_improvement < before_kaw
    reason = "accepted" if accepted else "not improved"
    return best if accepted else G, KawasakiProjectionStats(
        accepted=accepted,
        attempted_vertices=attempted_vertices,
        accepted_vertices=accepted_vertices,
        before_kawasaki=before_kaw,
        after_kawasaki=after_kaw if accepted else before_kaw,
        before_objective=before_obj,
        after_objective=best_obj if accepted else before_obj,
        crossing_rejections=crossing_rejections,
        reason=reason,
    )


def _load_graphs(path: str):
    with open(path, "rb") as handle:
        item = pickle.load(handle)
    return item if isinstance(item, list) else [item]


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python kawasaki_projection_repair.py <graph-or-list.pkl> [...]")
        return 1
    for path in argv[1:]:
        graphs = _load_graphs(path)
        print(f"\n{path}: {len(graphs)} graph(s)")
        for i, graph in enumerate(graphs):
            repaired, stats = repair_kawasaki_projection(graph)
            mae = oc.maekawa_penalty(repaired)
            print(
                f"  [{i}] accepted={stats.accepted} "
                f"kaw={stats.before_kawasaki:.4f}->{stats.after_kawasaki:.4f} "
                f"mae={mae:.4f} attempted={stats.attempted_vertices} "
                f"cross_reject={stats.crossing_rejections} reason={stats.reason}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
