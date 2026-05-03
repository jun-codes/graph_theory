"""
End-of-pipeline crease-pattern flat-foldability fixer.

This script is meant to run after the GA produces a candidate.  It does not
try to explore; it edits the crease pattern directly:

1. Repair true-interior even non-border degree topology.
2. For bad Kawasaki vertices, remove one incident crease and draw a new exact
   Kawasaki-completing crease ray through the sheet, splitting crossed creases.
3. Repeat until the Kawasaki residual is under tolerance or no edit improves it.
4. Solve mountain/valley labels with Z3 using Maekawa plus Big-Little-Big.

The geometry repair is intentionally local and constructive, similar to drawing
a replacement flat-foldable line in an editor.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from cp_io import write_cp_file
try:
    from gradient_geometry_repair import gradient_repair_kawasaki_symmetry
except Exception:
    gradient_repair_kawasaki_symmetry = None
from topology_even_repair import (
    bad_maekawa_topology_vertices,
    repair_even_nonborder_topology,
)


BASE = Path(__file__).resolve().parent
FOLD_BORDER = 1
FOLD_MOUNTAIN = 2
FOLD_VALLEY = 3
BOUNDARY_TOL = 3.0
POINT_TOL = 1e-5
ANGLE_TOL = 1e-7


@dataclass
class FixStats:
    topology_repairs: int = 0
    topology_added: int = 0
    topology_removed: int = 0
    line_replacements: int = 0
    failed_vertices: int = 0
    mv_status: str = "not_run"
    mv_changed_edges: int = 0


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def coords(G: nx.Graph, node: int) -> Tuple[float, float]:
    return float(G.nodes[node]["x"]), float(G.nodes[node]["y"])


def bbox(G: nx.Graph) -> Tuple[float, float, float, float]:
    xs = [coords(G, n)[0] for n in G.nodes()]
    ys = [coords(G, n)[1] for n in G.nodes()]
    return min(xs), max(xs), min(ys), max(ys)


def is_boundary_node(G: nx.Graph, node: int, tol: float = BOUNDARY_TOL) -> bool:
    x, y = coords(G, node)
    min_x, max_x, min_y, max_y = bbox(G)
    return (
        abs(x - min_x) <= tol
        or abs(x - max_x) <= tol
        or abs(y - min_y) <= tol
        or abs(y - max_y) <= tol
    )


def non_border_neighbors(G: nx.Graph, node: int) -> List[int]:
    return [
        nb for nb in G.neighbors(node)
        if G[node][nb].get("fold_type") != FOLD_BORDER
    ]


def non_border_degree(G: nx.Graph, node: int) -> int:
    return len(non_border_neighbors(G, node))


def is_true_interior_vertex(G: nx.Graph, node: int) -> bool:
    if is_boundary_node(G, node):
        return False
    return non_border_degree(G, node) >= 2


def interior_vertices(G: nx.Graph) -> List[int]:
    return [n for n in G.nodes() if is_true_interior_vertex(G, n)]


def recompute_features(G: nx.Graph) -> nx.Graph:
    for node in G.nodes():
        x, y = coords(G, node)
        angles = sorted(
            math.atan2(coords(G, nb)[1] - y, coords(G, nb)[0] - x)
            for nb in G.neighbors(node)
        )
        G.nodes[node]["degree"] = len(list(G.neighbors(node)))
        G.nodes[node]["angles"] = angles
    return G


def overwrite_graph(dst: nx.Graph, src: nx.Graph) -> nx.Graph:
    dst.clear()
    dst.add_nodes_from((n, dict(data)) for n, data in src.nodes(data=True))
    dst.add_edges_from((u, v, dict(data)) for u, v, data in src.edges(data=True))
    dst.graph.update(src.graph)
    return dst


def angle_of(G: nx.Graph, node: int, nb: int) -> float:
    x, y = coords(G, node)
    nx_, ny_ = coords(G, nb)
    return math.atan2(ny_ - y, nx_ - x) % (2 * math.pi)


def angle_gaps_from_angles(angles: Sequence[float]) -> List[float]:
    if len(angles) < 2:
        return []
    ordered = sorted(a % (2 * math.pi) for a in angles)
    gaps = []
    for i, angle in enumerate(ordered):
        nxt = ordered[(i + 1) % len(ordered)]
        if i == len(ordered) - 1:
            nxt += 2 * math.pi
        gaps.append(nxt - angle)
    return gaps


def signed_kawasaki_from_angles(angles: Sequence[float]) -> float:
    gaps = angle_gaps_from_angles(angles)
    if not gaps or len(gaps) % 2 == 1:
        return math.pi
    return sum(gaps[0::2]) - math.pi


def kawasaki_at(G: nx.Graph, node: int) -> float:
    if not is_true_interior_vertex(G, node):
        return 0.0
    nbs = non_border_neighbors(G, node)
    if len(nbs) % 2 == 1:
        return math.pi
    angles = [angle_of(G, node, nb) for nb in nbs]
    gaps = angle_gaps_from_angles(angles)
    if not gaps:
        return 0.0
    return abs(sum(gaps[0::2]) - math.pi) + abs(sum(gaps[1::2]) - math.pi)


def kawasaki_stats(G: nx.Graph) -> Tuple[float, float, int, int]:
    verts = interior_vertices(G)
    if not verts:
        return 0.0, 0.0, 0, 0
    vals = [kawasaki_at(G, n) for n in verts]
    odd = sum(1 for n in verts if non_border_degree(G, n) % 2 == 1)
    return float(np.mean(vals)), float(np.max(vals)), odd, len(verts)


def kawasaki_objective(G: nx.Graph) -> float:
    mean_v, max_v, odd, total = kawasaki_stats(G)
    odd_frac = odd / max(1, total)
    return mean_v + 0.6 * max_v + 2.0 * odd_frac


def kawasaki_bad_vertices(G: nx.Graph, tolerance: float) -> List[int]:
    return [
        node for node in interior_vertices(G)
        if kawasaki_at(G, node) > tolerance
    ]


def point_close(a: Tuple[float, float], b: Tuple[float, float], tol: float = POINT_TOL) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def find_or_create_node(G: nx.Graph, x: float, y: float, tol: float = POINT_TOL) -> int:
    for node in G.nodes():
        nx_, ny_ = coords(G, node)
        if math.hypot(nx_ - x, ny_ - y) <= tol:
            return node
    node = max(G.nodes(), default=-1) + 1
    G.add_node(node, x=float(x), y=float(y))
    return node


def edge_length(G: nx.Graph, u: int, v: int) -> float:
    ux, uy = coords(G, u)
    vx, vy = coords(G, v)
    return math.hypot(ux - vx, uy - vy)


def ray_boundary_endpoint(
    G: nx.Graph,
    node: int,
    angle: float,
) -> Optional[Tuple[float, float]]:
    x, y = coords(G, node)
    dx, dy = math.cos(angle), math.sin(angle)
    min_x, max_x, min_y, max_y = bbox(G)
    candidates = []
    if abs(dx) > 1e-9:
        for bx in (min_x, max_x):
            t = (bx - x) / dx
            by = y + t * dy
            if t > 1e-6 and min_y - POINT_TOL <= by <= max_y + POINT_TOL:
                candidates.append((t, bx, by))
    if abs(dy) > 1e-9:
        for by in (min_y, max_y):
            t = (by - y) / dy
            bx = x + t * dx
            if t > 1e-6 and min_x - POINT_TOL <= bx <= max_x + POINT_TOL:
                candidates.append((t, bx, by))
    if not candidates:
        return None
    _, ex, ey = min(candidates, key=lambda item: item[0])
    return float(ex), float(ey)


def iter_points(geom: BaseGeometry) -> Iterable[Point]:
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if hasattr(geom, "geoms"):
        pts = []
        for part in geom.geoms:
            pts.extend(iter_points(part))
        return pts
    return []


def project_t(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p: Tuple[float, float],
) -> float:
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return 0.0
    return ((p[0] - p0[0]) * vx + (p[1] - p0[1]) * vy) / denom


def split_existing_edges_for_line(
    G: nx.Graph,
    start_node: int,
    end_xy: Tuple[float, float],
    min_segment_len: float = 4.0,
) -> Optional[List[Tuple[float, int]]]:
    start_xy = coords(G, start_node)
    new_line = LineString([start_xy, end_xy])
    if new_line.length < min_segment_len:
        return None

    intersections: List[Tuple[float, int]] = [(0.0, start_node)]
    edge_splits: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}

    for u, v, data in list(G.edges(data=True)):
        if start_node in (u, v):
            continue
        old_line = LineString([coords(G, u), coords(G, v)])
        inter = new_line.intersection(old_line)
        if inter.is_empty:
            continue
        if inter.geom_type in ("LineString", "MultiLineString"):
            return None
        for point in iter_points(inter):
            px, py = float(point.x), float(point.y)
            t_new = project_t(start_xy, end_xy, (px, py))
            if t_new <= 1e-6 or t_new >= 1.0 + 1e-6:
                continue
            t_old = project_t(coords(G, u), coords(G, v), (px, py))
            if t_old <= 1e-6:
                split_node = u
            elif t_old >= 1.0 - 1e-6:
                split_node = v
            else:
                split_node = find_or_create_node(G, px, py)
                key = (u, v)
                edge_splits.setdefault(key, []).append((t_old, split_node))
            intersections.append((max(0.0, min(1.0, t_new)), split_node))

    end_node = find_or_create_node(G, end_xy[0], end_xy[1])
    intersections.append((1.0, end_node))

    for key, split_nodes in edge_splits.items():
        u, v = key
        if not G.has_edge(u, v):
            continue
        fold_type = G[u][v].get("fold_type", FOLD_MOUNTAIN)
        ordered = [(0.0, u)] + sorted(split_nodes, key=lambda item: item[0]) + [(1.0, v)]
        G.remove_edge(u, v)
        for (_, a), (_, b) in zip(ordered, ordered[1:]):
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b, fold_type=fold_type)

    unique: List[Tuple[float, int]] = []
    seen = set()
    for t, node in sorted(intersections, key=lambda item: item[0]):
        if node in seen:
            continue
        unique.append((t, node))
        seen.add(node)
    return unique


def add_full_crease_to_boundary(
    G: nx.Graph,
    start_node: int,
    angle: float,
    fold_type: int = FOLD_MOUNTAIN,
) -> bool:
    end_xy = ray_boundary_endpoint(G, start_node, angle)
    if end_xy is None:
        return False
    nodes_on_line = split_existing_edges_for_line(G, start_node, end_xy)
    if not nodes_on_line or len(nodes_on_line) < 2:
        return False
    placed = 0
    for (_, a), (_, b) in zip(nodes_on_line, nodes_on_line[1:]):
        if a == b or edge_length(G, a, b) < 1e-6:
            continue
        if not G.has_edge(a, b):
            G.add_edge(a, b, fold_type=fold_type)
            placed += 1
    return placed > 0


def add_full_line_through_boundary(
    G: nx.Graph,
    start_node: int,
    angle: float,
    fold_type: int = FOLD_MOUNTAIN,
) -> bool:
    """Draw the whole infinite crease through a vertex, clipped to the square."""
    first = add_full_crease_to_boundary(G, start_node, angle, fold_type=fold_type)
    second = add_full_crease_to_boundary(
        G, start_node, angle + math.pi, fold_type=fold_type)
    return first and second


def completion_angles_adding_opposite_pair(
    G: nx.Graph,
    node: int,
    samples_if_needed: int = 24,
) -> List[float]:
    """Angles theta where adding theta and theta+pi satisfies Kawasaki."""
    existing = [angle_of(G, node, nb) for nb in non_border_neighbors(G, node)]
    if len(existing) % 2 == 1:
        return []

    def objective(theta: float) -> float:
        return signed_kawasaki_from_angles(
            existing + [theta % (2 * math.pi), (theta + math.pi) % (2 * math.pi)]
        )

    def valid_angle(theta: float) -> bool:
        theta = theta % (2 * math.pi)
        theta2 = (theta + math.pi) % (2 * math.pi)
        for angle in existing:
            if abs(math.atan2(math.sin(theta - angle), math.cos(theta - angle))) < math.radians(6):
                return False
            if abs(math.atan2(math.sin(theta2 - angle), math.cos(theta2 - angle))) < math.radians(6):
                return False
        return True

    breakpoints = {0.0, math.pi}
    for angle in existing:
        breakpoints.add(angle % math.pi)
    ordered = sorted(breakpoints)
    intervals = []
    for lo, hi in zip(ordered, ordered[1:]):
        if hi - lo > math.radians(7):
            intervals.append((lo, hi))
    if ordered[0] > 0 or ordered[-1] < math.pi:
        lo, hi = ordered[-1], ordered[0] + math.pi
        if hi - lo > math.radians(7):
            intervals.append((lo, hi))

    candidates = []
    for lo, hi in intervals:
        a = lo + ANGLE_TOL
        b = hi - ANGLE_TOL
        fa = objective(a)
        fb = objective(b)
        if abs(fa) < 1e-8 and valid_angle(a):
            candidates.append(a % math.pi)
            continue
        if abs(fb) < 1e-8 and valid_angle(b):
            candidates.append(b % math.pi)
            continue
        if fa * fb <= 0:
            left, right = a, b
            for _ in range(70):
                mid = 0.5 * (left + right)
                fm = objective(mid)
                if abs(fm) < 1e-12:
                    left = right = mid
                    break
                if fa * fm <= 0:
                    right = mid
                    fb = fm
                else:
                    left = mid
                    fa = fm
            theta = 0.5 * (left + right)
            if valid_angle(theta):
                candidates.append(theta % math.pi)

    if not candidates:
        best = []
        for i in range(samples_if_needed):
            theta = math.pi * (i + 0.5) / samples_if_needed
            if valid_angle(theta):
                best.append((abs(objective(theta)), theta))
        candidates = [theta for _, theta in sorted(best)[:4]]

    deduped = []
    for theta in candidates:
        if not any(abs(math.atan2(math.sin(theta - old), math.cos(theta - old))) < 1e-5 for old in deduped):
            deduped.append(theta)
    return deduped


def completion_angles_after_removing(
    G: nx.Graph,
    node: int,
    remove_nb: int,
    samples_if_needed: int = 12,
) -> List[float]:
    remaining = [
        angle_of(G, node, nb)
        for nb in non_border_neighbors(G, node)
        if nb != remove_nb
    ]
    if (len(remaining) + 1) % 2 == 1 or not remaining:
        return []
    ordered = sorted(a % (2 * math.pi) for a in remaining)
    candidates = []

    def valid_angle(theta: float) -> bool:
        theta = theta % (2 * math.pi)
        return min(abs(math.atan2(math.sin(theta - a), math.cos(theta - a))) for a in ordered) > math.radians(6)

    intervals = []
    for i, lo in enumerate(ordered):
        hi = ordered[(i + 1) % len(ordered)]
        if i == len(ordered) - 1:
            hi += 2 * math.pi
        intervals.append((lo, hi))

    for lo, hi in intervals:
        if hi - lo <= math.radians(7):
            continue
        a = lo + ANGLE_TOL
        b = hi - ANGLE_TOL
        fa = signed_kawasaki_from_angles(remaining + [a])
        fb = signed_kawasaki_from_angles(remaining + [b])
        if abs(fa) < 1e-6 and valid_angle(a):
            candidates.append(a % (2 * math.pi))
            continue
        if abs(fb) < 1e-6 and valid_angle(b):
            candidates.append(b % (2 * math.pi))
            continue
        if fa * fb <= 0:
            left, right = a, b
            for _ in range(60):
                mid = 0.5 * (left + right)
                fm = signed_kawasaki_from_angles(remaining + [mid])
                if abs(fm) < 1e-10:
                    left = right = mid
                    break
                if fa * fm <= 0:
                    right = mid
                    fb = fm
                else:
                    left = mid
                    fa = fm
            theta = 0.5 * (left + right)
            if valid_angle(theta):
                candidates.append(theta % (2 * math.pi))

    if not candidates:
        best = []
        for lo, hi in intervals:
            for k in range(1, samples_if_needed):
                theta = lo + (hi - lo) * k / samples_if_needed
                if valid_angle(theta):
                    err = abs(signed_kawasaki_from_angles(remaining + [theta]))
                    best.append((err, theta % (2 * math.pi)))
        candidates = [theta for _, theta in sorted(best)[:3]]

    deduped = []
    for theta in candidates:
        if not any(abs(math.atan2(math.sin(theta - old), math.cos(theta - old))) < 1e-5 for old in deduped):
            deduped.append(theta)
    return deduped


def candidate_score(G: nx.Graph, focus_node: int) -> float:
    mean_v, max_v, odd, total = kawasaki_stats(G)
    topo_bad = len(bad_maekawa_topology_vertices(G))
    focus = kawasaki_at(G, focus_node) if focus_node in G else 0.0
    return (
        mean_v
        + 0.8 * max_v
        + 0.6 * focus
        + 2.0 * odd / max(1, total)
        + 0.8 * topo_bad
    )


def remove_isolated_nonboundary_nodes(G: nx.Graph) -> None:
    for node in list(nx.isolates(G)):
        if not is_boundary_node(G, node):
            G.remove_node(node)


def replace_one_bad_kawasaki_line(
    G: nx.Graph,
    node: int,
    tolerance: float,
) -> bool:
    before_focus = kawasaki_at(G, node)
    before_score = candidate_score(G, node)
    best_graph = None
    best_key = (before_focus, before_score)

    incident = non_border_neighbors(G, node)
    if len(incident) < 2 or len(incident) % 2 == 1:
        return False

    # Preferred edit: draw a full crease through the vertex. This preserves
    # even degree at the focus vertex and often exactly fixes Kawasaki without
    # hurting a neighbor.
    for theta in completion_angles_adding_opposite_pair(G, node):
        H = copy.deepcopy(G)
        ok = add_full_line_through_boundary(H, node, theta)
        if not ok:
            continue
        remove_isolated_nonboundary_nodes(H)
        recompute_features(H)
        repaired, _ = repair_even_nonborder_topology(H, max_rounds=3, max_added_edges=18)
        H = recompute_features(repaired)
        focus = kawasaki_at(H, node) if node in H else math.inf
        score = candidate_score(H, node)
        key = (focus, score)
        if focus <= tolerance:
            overwrite_graph(G, H)
            recompute_features(G)
            return True
        if key < best_key:
            best_key = key
            best_graph = H

    # Fallback edit: remove one bad incident crease, then add one exact
    # completing ray to the boundary.
    for remove_nb in sorted(incident, key=lambda nb: edge_length(G, node, nb), reverse=True):
        angles = completion_angles_after_removing(G, node, remove_nb)
        for theta in angles:
            H = copy.deepcopy(G)
            if H.has_edge(node, remove_nb):
                H.remove_edge(node, remove_nb)
            ok = add_full_crease_to_boundary(H, node, theta)
            if not ok:
                continue
            remove_isolated_nonboundary_nodes(H)
            recompute_features(H)
            repaired, _ = repair_even_nonborder_topology(H, max_rounds=3, max_added_edges=18)
            H = recompute_features(repaired)
            focus = kawasaki_at(H, node) if node in H else math.inf
            score = candidate_score(H, node)
            if focus <= tolerance:
                overwrite_graph(G, H)
                recompute_features(G)
                return True
            key = (focus, score)
            if key < best_key:
                best_key = key
                best_graph = H

    if best_graph is None:
        return False
    if best_key[0] < before_focus - 1e-8 or best_key[1] < before_score - 1e-5:
        overwrite_graph(G, best_graph)
        recompute_features(G)
        return True
    return False


def repair_kawasaki_lines(
    G: nx.Graph,
    *,
    tolerance: float = 1e-4,
    max_rounds: int = 120,
    time_limit_s: float = 30.0,
    verbose: bool = True,
) -> Tuple[int, int]:
    replacements = 0
    failed = 0
    start = time.monotonic()
    for round_idx in range(1, max_rounds + 1):
        if time.monotonic() - start > time_limit_s:
            if verbose:
                print(f"  Kawasaki repair time limit hit after {round_idx - 1} rounds", flush=True)
            break
        bad = [
            (kawasaki_at(G, node), node)
            for node in interior_vertices(G)
            if kawasaki_at(G, node) > tolerance
        ]
        if not bad:
            break
        bad.sort(reverse=True)
        if verbose and (round_idx == 1 or round_idx % 10 == 0):
            mean_v, max_v, odd, total = kawasaki_stats(G)
            print(
                f"  Kaw round {round_idx}: bad={len(bad)}/{total} "
                f"KMean={mean_v:.6f} KMax={max_v:.6f} Odd={odd}",
                flush=True,
            )
        progressed = False
        for _, node in bad[:20]:
            if time.monotonic() - start > time_limit_s:
                break
            if replace_one_bad_kawasaki_line(G, node, tolerance):
                replacements += 1
                progressed = True
                break
            failed += 1
        if not progressed:
            break
    return replacements, failed


def relax_kawasaki_geometry(
    G: nx.Graph,
    *,
    tolerance: float,
    time_limit_s: float,
    verbose: bool = True,
) -> int:
    """Continuous final repair: move existing interior vertices, no new creases."""
    if gradient_repair_kawasaki_symmetry is None:
        if verbose:
            print("  Gradient repair unavailable; skipping coordinate relaxation", flush=True)
        return 0

    start = time.monotonic()
    accepted = 0
    schedules = [
        (260, 0.16, 10.0, 0.10, 1.8),
        (420, 0.11, 18.0, 0.08, 2.6),
        (650, 0.075, 30.0, 0.05, 3.4),
        (900, 0.050, 45.0, 0.03, 4.2),
    ]

    for attempt, (steps, lr, max_move, move_weight, worst_weight) in enumerate(schedules, start=1):
        if time.monotonic() - start > time_limit_s:
            break

        bad = kawasaki_bad_vertices(G, tolerance)
        mean_v, max_v, odd, total = kawasaki_stats(G)
        before_pass_obj = kawasaki_objective(G)
        if verbose:
            print(
                f"  Relax {attempt}: bad={len(bad)}/{total} "
                f"KMean={mean_v:.6f} KMax={max_v:.6f} Odd={odd}",
                flush=True,
            )
        if not bad and odd == 0:
            break

        mutable = [
            node for node in G.nodes()
            if not is_boundary_node(G, node)
        ]
        target = interior_vertices(G)
        if not mutable or not target:
            break

        before_obj = kawasaki_objective(G)
        before_max = max_v
        repaired, stats = gradient_repair_kawasaki_symmetry(
            G,
            steps=steps,
            lr=lr,
            kaw_weight=1.0,
            symmetry_weight=0.0,
            move_weight=move_weight,
            max_move=max_move,
            min_edge_length=5.0,
            reject_crossings=True,
            mutable_nodes=mutable,
            target_nodes=target,
            min_angle_gap_deg=5.0,
            angle_gap_weight=0.20,
            worst_weight=worst_weight,
            logsumexp_tau=0.08,
        )
        repaired = recompute_features(repaired)
        after_obj = kawasaki_objective(repaired)
        _, after_max, after_odd, _ = kawasaki_stats(repaired)

        if stats.accepted and (
            after_obj < before_obj - 1e-7
            or after_max < before_max - 1e-7
            or (after_max <= tolerance and after_odd == 0)
        ):
            overwrite_graph(G, repaired)
            accepted += 1
            if verbose:
                print(
                    f"    accepted: obj {before_obj:.6f}->{after_obj:.6f}, "
                    f"KMax {before_max:.6f}->{after_max:.6f}",
                    flush=True,
                )
        elif verbose:
            print(f"    no accept: {stats.reason}", flush=True)

    return accepted


def assign_mv_greedy_big_little_big(
    G: nx.Graph,
    *,
    blb_margin_deg: float = 1e-3,
    passes: int = 4,
) -> int:
    """Fast fallback when exact MV solving is too large for pure Python."""
    changed_edges = set()
    margin = math.radians(blb_margin_deg)

    for _ in range(passes):
        changed_this_pass = False
        for node in interior_vertices(G):
            incident = []
            for nb in non_border_neighbors(G, node):
                incident.append((angle_of(G, node, nb), edge_key(node, nb)))
            incident.sort(key=lambda item: item[0])
            degree = len(incident)
            if degree < 2:
                continue

            gaps = angle_gaps_from_angles([angle for angle, _ in incident])
            for i, gap in enumerate(gaps):
                prev_gap = gaps[i - 1]
                next_gap = gaps[(i + 1) % degree]
                if gap + margin < prev_gap and gap + margin < next_gap:
                    a = incident[i][1]
                    b = incident[(i + 1) % degree][1]
                    au, av = a
                    bu, bv = b
                    if G[au][av].get("fold_type") == G[bu][bv].get("fold_type"):
                        G[bu][bv]["fold_type"] = (
                            FOLD_VALLEY
                            if G[au][av].get("fold_type") == FOLD_MOUNTAIN
                            else FOLD_MOUNTAIN
                        )
                        changed_edges.add(b)
                        changed_this_pass = True

            mountain = [
                edge for _, edge in incident
                if G[edge[0]][edge[1]].get("fold_type") == FOLD_MOUNTAIN
            ]
            valley = [
                edge for _, edge in incident
                if G[edge[0]][edge[1]].get("fold_type") == FOLD_VALLEY
            ]
            if degree % 2 == 0 and mountain and valley:
                target_delta = 2 if len(mountain) >= len(valley) else -2
                current_delta = len(mountain) - len(valley)
                while current_delta != target_delta:
                    if current_delta > target_delta and mountain:
                        edge = mountain.pop()
                        G[edge[0]][edge[1]]["fold_type"] = FOLD_VALLEY
                        valley.append(edge)
                        current_delta -= 2
                    elif current_delta < target_delta and valley:
                        edge = valley.pop()
                        G[edge[0]][edge[1]]["fold_type"] = FOLD_MOUNTAIN
                        mountain.append(edge)
                        current_delta += 2
                    else:
                        break
                    changed_edges.add(edge)
                    changed_this_pass = True

        if not changed_this_pass:
            break
    return len(changed_edges)


def solve_mv_python_backtracking(
    G: nx.Graph,
    *,
    timeout_s: float = 8.0,
    blb_margin_deg: float = 1e-3,
    max_edges: int = 70,
) -> Tuple[str, int]:
    """Pure-Python fallback for Maekawa + Big-Little-Big when z3 is missing."""
    edges = sorted(
        edge_key(u, v)
        for u, v, data in G.edges(data=True)
        if data.get("fold_type") != FOLD_BORDER
    )
    if not edges:
        return "python_skipped:no_creases", 0
    if len(edges) > max_edges:
        greedy_changed = assign_mv_greedy_big_little_big(G, blb_margin_deg=blb_margin_deg)
        return f"python_greedy_too_many_edges:{len(edges)}", greedy_changed

    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    constraints = []
    blb_margin = math.radians(blb_margin_deg)

    for node in interior_vertices(G):
        incident = []
        for nb in non_border_neighbors(G, node):
            incident.append((angle_of(G, node, nb), edge_to_idx[edge_key(node, nb)]))
        incident.sort(key=lambda item: item[0])
        degree = len(incident)
        if degree < 2 or degree % 2 == 1:
            return f"python_topology_unsat:{node}:d{degree}", 0

        allowed = {degree // 2 - 1, degree // 2 + 1}
        pairs = []
        gaps = angle_gaps_from_angles([angle for angle, _ in incident])
        for i, gap in enumerate(gaps):
            prev_gap = gaps[i - 1]
            next_gap = gaps[(i + 1) % degree]
            if gap + blb_margin < prev_gap and gap + blb_margin < next_gap:
                pairs.append((incident[i][1], incident[(i + 1) % degree][1]))
        constraints.append({
            "incident": [idx for _, idx in incident],
            "allowed": allowed,
            "pairs": pairs,
        })

    if not constraints:
        return "python_skipped:no_interior_vertices", 0

    current = [
        G[u][v].get("fold_type") == FOLD_MOUNTAIN
        for u, v in edges
    ]
    assignments: List[Optional[bool]] = [None] * len(edges)

    var_degree = [0] * len(edges)
    for constraint in constraints:
        for idx in constraint["incident"]:
            var_degree[idx] += 1
        for a, b in constraint["pairs"]:
            var_degree[a] += 2
            var_degree[b] += 2
    order = sorted(range(len(edges)), key=lambda idx: var_degree[idx], reverse=True)
    start = time.monotonic()

    def constraint_possible(constraint) -> bool:
        m_count = 0
        unassigned = 0
        for idx in constraint["incident"]:
            value = assignments[idx]
            if value is None:
                unassigned += 1
            elif value:
                m_count += 1
        if not any(m_count <= target <= m_count + unassigned for target in constraint["allowed"]):
            return False
        if unassigned == 0 and m_count not in constraint["allowed"]:
            return False
        for a, b in constraint["pairs"]:
            av = assignments[a]
            bv = assignments[b]
            if av is not None and bv is not None and av == bv:
                return False
        return True

    def all_possible() -> bool:
        return all(constraint_possible(constraint) for constraint in constraints)

    def search(pos: int = 0) -> bool:
        if time.monotonic() - start > timeout_s:
            raise TimeoutError
        if pos >= len(order):
            return all_possible()
        idx = order[pos]
        preferred = current[idx]
        for value in (preferred, not preferred):
            assignments[idx] = value
            if all_possible() and search(pos + 1):
                return True
        assignments[idx] = None
        return False

    try:
        solved = search()
    except TimeoutError:
        return "python_timeout", 0
    if not solved:
        return "python_unsat", 0

    changed = 0
    for edge, value in zip(edges, assignments):
        u, v = edge
        new_fold = FOLD_MOUNTAIN if value else FOLD_VALLEY
        if G[u][v].get("fold_type") != new_fold:
            changed += 1
        G[u][v]["fold_type"] = new_fold
    return "python_sat", changed


def solve_mv_maekawa_big_little_big(
    G: nx.Graph,
    *,
    timeout_ms: int = 2000,
    blb_margin_deg: float = 1e-3,
) -> Tuple[str, int]:
    try:
        from z3 import Bool, If, Optimize, Or, Sum, is_true, sat
    except Exception as exc:
        status, changed = solve_mv_python_backtracking(
            G, timeout_s=max(2.0, timeout_ms / 1000.0 * 4.0),
            blb_margin_deg=blb_margin_deg,
        )
        return f"z3_unavailable:{exc};{status}", changed

    non_border_edges = sorted(
        edge_key(u, v)
        for u, v, data in G.edges(data=True)
        if data.get("fold_type") != FOLD_BORDER
    )
    if not non_border_edges:
        return "skipped:no_creases", 0

    variables = {edge: Bool(f"mv_{edge[0]}_{edge[1]}") for edge in non_border_edges}
    opt = Optimize()
    opt.set(timeout=timeout_ms)

    blb_margin = math.radians(blb_margin_deg)
    constrained = 0
    for node in interior_vertices(G):
        incident = []
        for nb in non_border_neighbors(G, node):
            incident.append((angle_of(G, node, nb), edge_key(node, nb)))
        incident.sort(key=lambda item: item[0])
        degree = len(incident)
        if degree < 2 or degree % 2 == 1:
            return f"topology_unsat:{node}:d{degree}", 0
        constrained += 1

        mountain_count = Sum([If(variables[edge], 1, 0) for _, edge in incident])
        opt.add(Or(
            mountain_count == degree // 2 - 1,
            mountain_count == degree // 2 + 1,
        ))

        gaps = angle_gaps_from_angles([angle for angle, _ in incident])
        for i, gap in enumerate(gaps):
            prev_gap = gaps[i - 1]
            next_gap = gaps[(i + 1) % degree]
            if gap + blb_margin < prev_gap and gap + blb_margin < next_gap:
                edge_a = incident[i][1]
                edge_b = incident[(i + 1) % degree][1]
                opt.add(variables[edge_a] != variables[edge_b])

    if constrained == 0:
        return "skipped:no_interior_vertices", 0

    change_terms = []
    for edge in non_border_edges:
        u, v = edge
        currently_mountain = G[u][v].get("fold_type") == FOLD_MOUNTAIN
        change_terms.append(If(variables[edge] == currently_mountain, 0, 1))
    opt.minimize(Sum(change_terms))

    result = opt.check()
    if result != sat:
        return str(result), 0

    model = opt.model()
    changed = 0
    for edge in non_border_edges:
        u, v = edge
        was = G[u][v].get("fold_type")
        is_mountain = is_true(model.evaluate(variables[edge], model_completion=True))
        new_fold = FOLD_MOUNTAIN if is_mountain else FOLD_VALLEY
        if was != new_fold:
            changed += 1
        G[u][v]["fold_type"] = new_fold
    return "sat", changed


def load_cp_graph(path: Path) -> nx.Graph:
    G = nx.Graph()
    next_node = 0

    def node_for(x: float, y: float) -> int:
        nonlocal next_node
        for node in G.nodes():
            nx_, ny_ = coords(G, node)
            if math.hypot(nx_ - x, ny_ - y) <= POINT_TOL:
                return node
        node = next_node
        next_node += 1
        G.add_node(node, x=float(x), y=float(y))
        return node

    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        fold_type = int(float(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])
        u = node_for(x1, y1)
        v = node_for(x2, y2)
        if u != v:
            G.add_edge(u, v, fold_type=fold_type)
    return recompute_features(G)


def load_graph(path: Path) -> nx.Graph:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            G = pickle.load(f)
        return recompute_features(G)
    if path.suffix.lower() == ".cp":
        return load_cp_graph(path)
    raise ValueError(f"Unsupported input file: {path}")


def save_graph_outputs(G: nx.Graph, output_prefix: Path, *, render_png: bool = True) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".pkl").open("wb") as f:
        pickle.dump(G, f)
    write_cp_file(G, str(output_prefix.with_suffix(".cp")))
    if render_png:
        visualise(G, output_prefix.with_suffix(".png"))


def visualise(G: nx.Graph, path: Path) -> None:
    if plt is None:
        print("matplotlib unavailable; skipped preview PNG")
        return
    pos = {n: coords(G, n) for n in G.nodes()}
    _, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get("fold_type") == FOLD_MOUNTAIN],
        edge_color="red", width=1.1, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get("fold_type") == FOLD_VALLEY],
        edge_color="blue", width=0.8, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get("fold_type") == FOLD_BORDER],
        edge_color="black", width=2.0, ax=ax,
    )
    nx.draw_networkx_nodes(G, pos, node_size=8, node_color="black", ax=ax)
    mean_v, max_v, odd, total = kawasaki_stats(G)
    ax.set_title(f"Kaw mean={mean_v:.6f} max={max_v:.6f} odd={odd}/{total}", fontsize=9)
    ax.axis("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def default_input_path() -> Path:
    for name in (
        "sym_topology_best_generated.pkl",
        "best_generated.pkl",
        "line_topology_z3_best_generated.pkl",
    ):
        path = BASE / name
        if path.exists():
            return path
    return BASE / "best_generated.pkl"


def fix_pattern(
    G: nx.Graph,
    *,
    kaw_tolerance: float,
    global_passes: int,
    kaw_rounds: int,
    time_limit_s: float,
    skip_mv: bool,
    allow_line_edits: bool,
    verbose: bool,
) -> FixStats:
    stats = FixStats()
    recompute_features(G)
    start = time.monotonic()

    for pass_idx in range(1, global_passes + 1):
        elapsed = time.monotonic() - start
        if elapsed > time_limit_s:
            if verbose:
                print(f"Global time limit hit before pass {pass_idx}", flush=True)
            break
        repaired, topo_stats = repair_even_nonborder_topology(
            G, max_rounds=5, max_added_edges=32)
        overwrite_graph(G, repaired)
        stats.topology_repairs += int(topo_stats.status in ("repaired", "partial"))
        stats.topology_added += topo_stats.added_edges
        stats.topology_removed += topo_stats.removed_edges

        mean_v, max_v, odd, total = kawasaki_stats(G)
        before_pass_obj = kawasaki_objective(G)
        if verbose:
            print(
                f"Pass {pass_idx}: KMean={mean_v:.6f} KMax={max_v:.6f} "
                f"Odd={odd}/{total} TopoBad={len(bad_maekawa_topology_vertices(G))}",
                flush=True,
            )

        remaining = max(1.0, time_limit_s - (time.monotonic() - start))
        relax_limit = min(remaining, max(6.0, 0.65 * time_limit_s / max(1, global_passes)))
        relax_kawasaki_geometry(
            G,
            tolerance=kaw_tolerance,
            time_limit_s=relax_limit,
            verbose=verbose,
        )

        added = failed = 0
        if allow_line_edits:
            remaining = max(1.0, time_limit_s - (time.monotonic() - start))
            pass_limit = min(remaining, max(3.0, 0.35 * time_limit_s / max(1, global_passes)))
            added, failed = repair_kawasaki_lines(
                G,
                tolerance=kaw_tolerance,
                max_rounds=kaw_rounds,
                time_limit_s=pass_limit,
                verbose=verbose,
            )
            stats.line_replacements += added
            stats.failed_vertices += failed

        mean_v, max_v, odd, _ = kawasaki_stats(G)
        after_pass_obj = kawasaki_objective(G)
        if max_v <= kaw_tolerance and odd == 0 and not bad_maekawa_topology_vertices(G):
            break
        if (added == 0 and topo_stats.added_edges == 0 and
                topo_stats.removed_edges == 0 and
                after_pass_obj >= before_pass_obj - 1e-7):
            break

    if skip_mv:
        stats.mv_status = "skipped"
        stats.mv_changed_edges = 0
    else:
        remaining = max(0.5, time_limit_s - (time.monotonic() - start))
        stats.mv_status, stats.mv_changed_edges = solve_mv_maekawa_big_little_big(
            G, timeout_ms=int(min(3000, remaining * 1000)))
    recompute_features(G)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fix a generated CP for local flat-foldability.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Input .pkl graph or .cp file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension. Defaults to <input>_fixed.",
    )
    parser.add_argument("--kaw-tol", type=float, default=1e-6)
    parser.add_argument("--global-passes", type=int, default=4)
    parser.add_argument("--kaw-rounds", type=int, default=80)
    parser.add_argument("--time-limit", type=float, default=90.0)
    parser.add_argument("--skip-mv", action="store_true")
    parser.add_argument(
        "--allow-line-edits",
        action="store_true",
        help="Also try topology-changing line edits. Off by default because it can explode intersections.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input
    if not input_path.is_absolute():
        input_path = BASE / input_path
    if args.output_prefix is None:
        output_prefix = input_path.with_name(f"{input_path.stem}_fixed")
    else:
        output_prefix = args.output_prefix
        if not output_prefix.is_absolute():
            output_prefix = BASE / output_prefix

    G = load_graph(input_path)
    before = kawasaki_stats(G)
    print(
        f"Loaded {input_path.name}: N={G.number_of_nodes()} E={G.number_of_edges()} "
        f"KMean={before[0]:.6f} KMax={before[1]:.6f} Odd={before[2]}"
    )

    stats = fix_pattern(
        G,
        kaw_tolerance=args.kaw_tol,
        global_passes=args.global_passes,
        kaw_rounds=args.kaw_rounds,
        time_limit_s=args.time_limit,
        skip_mv=args.skip_mv,
        allow_line_edits=args.allow_line_edits,
        verbose=not args.quiet,
    )
    after = kawasaki_stats(G)

    save_graph_outputs(G, output_prefix, render_png=not args.no_png)
    print(
        f"Fixed: KMean={after[0]:.6f} KMax={after[1]:.6f} Odd={after[2]} "
        f"TopoBad={len(bad_maekawa_topology_vertices(G))}"
    )
    if after[1] > args.kaw_tol or after[2] != 0:
        print(
            "WARNING: unresolved Kawasaki vertices remain. "
            "Try a larger --kaw-rounds/--global-passes or inspect the preview."
        )
    print(
        f"Edits: line_replacements={stats.line_replacements} "
        f"topology_added={stats.topology_added} topology_removed={stats.topology_removed} "
        f"mv_status={stats.mv_status} mv_changed={stats.mv_changed_edges}"
    )
    print(f"Saved {output_prefix.with_suffix('.pkl').name} and {output_prefix.with_suffix('.cp').name}")


if __name__ == "__main__":
    main()
