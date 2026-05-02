"""
Gradient-based geometry repair for generated crease-pattern graphs.

This module only changes node coordinates.  It does not add/remove creases or
change mountain/valley labels.  The repair objective targets the geometry-side
constraints that coordinates can affect:

- Kawasaki residual at interior vertices
- approximate left/right mirror symmetry
- short-edge degeneracy

Border nodes are kept fixed, and a candidate repair is rejected if it creates
crease crossings.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import torch
from shapely.geometry import LineString


SCALE = 200.0
BORDER = SCALE - 5.0
BOUNDARY_TOL = 3.0


@dataclass
class RepairStats:
    accepted: bool
    before_objective: float
    after_objective: float
    before_kawasaki: float
    after_kawasaki: float
    before_symmetry: float
    after_symmetry: float
    reason: str


def _coords(G: nx.Graph, node: int) -> Tuple[float, float]:
    return float(G.nodes[node]["x"]), float(G.nodes[node]["y"])


def is_boundary_node(
    G: nx.Graph,
    node: int,
    border: float = BORDER,
    tol: float = BOUNDARY_TOL,
) -> bool:
    x, y = _coords(G, node)
    return (
        abs(x - border) < tol
        or abs(x + border) < tol
        or abs(y - border) < tol
        or abs(y + border) < tol
    )


def is_interior_vertex(G: nx.Graph, node: int) -> bool:
    neighbors = list(G.neighbors(node))
    return len(neighbors) >= 2 and not all(
        G[node][nb].get("fold_type") == 1 for nb in neighbors
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


def kawasaki_at(G: nx.Graph, node: int) -> float:
    if not is_interior_vertex(G, node):
        return 0.0
    gaps = _angle_gaps(G, node)
    if not gaps:
        return 0.0
    even_sum = sum(gaps[i] for i in range(0, len(gaps), 2))
    odd_sum = sum(gaps[i] for i in range(1, len(gaps), 2))
    return float(abs(even_sum - math.pi) + abs(odd_sum - math.pi))


def kawasaki_mean(G: nx.Graph) -> float:
    values = [kawasaki_at(G, n) for n in G.nodes()]
    nonzero = [v for v in values if v > 0]
    return float(np.mean(nonzero)) if nonzero else 0.0


def symmetry_error(G: nx.Graph, border: float = BORDER, tol: float = BOUNDARY_TOL) -> float:
    mutable = [n for n in G.nodes() if not is_boundary_node(G, n, border, tol)]
    if len(mutable) < 2:
        return 0.0
    coords = np.array([_coords(G, n) for n in mutable], dtype=float)
    errors = []
    for x, y in coords:
        dists = np.sqrt((coords[:, 0] + x) ** 2 + (coords[:, 1] - y) ** 2)
        errors.append(float(np.min(dists) / SCALE))
    return float(np.mean(errors))


def geometry_objective(G: nx.Graph, symmetry_weight: float = 0.5) -> Tuple[float, float, float]:
    kaw = kawasaki_mean(G)
    sym = symmetry_error(G)
    return kaw + symmetry_weight * sym, kaw, sym


def has_crossings(G: nx.Graph) -> bool:
    edges = list(G.edges())
    segments = []
    for u, v in edges:
        segments.append(
            LineString([_coords(G, u), _coords(G, v)])
        )
    for i, (u1, v1) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            if len({u1, v1, u2, v2}) < 4:
                continue
            if segments[i].crosses(segments[j]):
                return True
    return False


def _neighbor_order_specs(G: nx.Graph, node_to_idx: Dict[int, int]):
    specs = []
    for node in G.nodes():
        if not is_interior_vertex(G, node):
            continue
        cx, cy = _coords(G, node)
        ordered = sorted(
            list(G.neighbors(node)),
            key=lambda nb: math.atan2(_coords(G, nb)[1] - cy, _coords(G, nb)[0] - cx),
        )
        if len(ordered) >= 2:
            specs.append((node_to_idx[node], [node_to_idx[nb] for nb in ordered]))
    return specs


def _symmetry_pairs(
    G: nx.Graph,
    nodes: List[int],
    mutable_nodes: List[int],
    node_to_idx: Dict[int, int],
    pair_tol: float,
) -> List[Tuple[int, int]]:
    left = [n for n in mutable_nodes if G.nodes[n]["x"] < 0]
    right = [n for n in mutable_nodes if G.nodes[n]["x"] >= 0]
    used_left = set()
    pairs = []

    for r in sorted(right, key=lambda n: abs(float(G.nodes[n]["x"])), reverse=True):
        rx, ry = _coords(G, r)
        candidates = [n for n in left if n not in used_left]
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda n: (_coords(G, n)[0] + rx) ** 2 + (_coords(G, n)[1] - ry) ** 2,
        )
        bx, by = _coords(G, best)
        dist = math.hypot(bx + rx, by - ry)
        if dist <= pair_tol:
            used_left.add(best)
            pairs.append((node_to_idx[r], node_to_idx[best]))

    return pairs


def _edge_index_pairs(G: nx.Graph, node_to_idx: Dict[int, int]) -> List[Tuple[int, int]]:
    return [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]


def _torch_loss(
    coords: torch.Tensor,
    original: torch.Tensor,
    mutable_idx: torch.Tensor,
    kaw_specs,
    sym_pairs,
    edge_pairs,
    kaw_weight: float,
    symmetry_weight: float,
    move_weight: float,
    min_edge_length: float,
    scale: float,
) -> torch.Tensor:
    device = coords.device
    dtype = coords.dtype
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    two_pi = torch.tensor(2 * math.pi, dtype=dtype, device=device)

    kaw_terms = []
    for center_idx, neighbor_idxs in kaw_specs:
        center = coords[center_idx]
        neighbors = coords[neighbor_idxs]
        vectors = neighbors - center
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])
        gaps = torch.remainder(torch.roll(angles, shifts=-1) - angles + two_pi, two_pi)
        even_sum = gaps[0::2].sum()
        odd_sum = gaps[1::2].sum()
        kaw_terms.append(((even_sum - pi) / pi) ** 2 + ((odd_sum - pi) / pi) ** 2)
    if kaw_terms:
        kaw_loss = torch.stack(kaw_terms).mean()
    else:
        kaw_loss = torch.zeros((), dtype=dtype, device=device)

    sym_terms = []
    for right_idx, left_idx in sym_pairs:
        right = coords[right_idx]
        left = coords[left_idx]
        sym_terms.append(((right[0] + left[0]) / scale) ** 2 + ((right[1] - left[1]) / scale) ** 2)
    if sym_terms:
        sym_loss = torch.stack(sym_terms).mean()
    else:
        sym_loss = torch.zeros((), dtype=dtype, device=device)

    short_terms = []
    for u_idx, v_idx in edge_pairs:
        length = torch.linalg.norm(coords[u_idx] - coords[v_idx])
        short_terms.append((torch.relu(torch.tensor(min_edge_length, dtype=dtype, device=device) - length) / scale) ** 2)
    if short_terms:
        short_loss = torch.stack(short_terms).mean()
    else:
        short_loss = torch.zeros((), dtype=dtype, device=device)

    if mutable_idx.numel() > 0:
        move_loss = (((coords[mutable_idx] - original[mutable_idx]) / scale) ** 2).mean()
    else:
        move_loss = torch.zeros((), dtype=dtype, device=device)

    return kaw_weight * kaw_loss + symmetry_weight * sym_loss + 0.25 * short_loss + move_weight * move_loss


def _apply_coords(G: nx.Graph, nodes: List[int], coords: np.ndarray) -> nx.Graph:
    out = copy.deepcopy(G)
    for i, node in enumerate(nodes):
        out.nodes[node]["x"] = float(coords[i, 0])
        out.nodes[node]["y"] = float(coords[i, 1])
    return out


def gradient_repair_kawasaki_symmetry(
    G: nx.Graph,
    *,
    steps: int = 25,
    lr: float = 0.45,
    kaw_weight: float = 1.0,
    symmetry_weight: float = 0.35,
    move_weight: float = 0.015,
    max_move: float = 12.0,
    min_edge_length: float = 4.0,
    pair_tol: float = 130.0,
    border: float = BORDER,
    boundary_tol: float = BOUNDARY_TOL,
    reject_crossings: bool = True,
) -> Tuple[nx.Graph, RepairStats]:
    before_obj, before_kaw, before_sym = geometry_objective(G, symmetry_weight=0.5)
    nodes = list(G.nodes())
    if not nodes:
        return G, RepairStats(False, before_obj, before_obj, before_kaw, before_kaw, before_sym, before_sym, "empty")

    mutable_nodes = [n for n in nodes if not is_boundary_node(G, n, border, boundary_tol)]
    if not mutable_nodes:
        return G, RepairStats(False, before_obj, before_obj, before_kaw, before_kaw, before_sym, before_sym, "no mutable nodes")

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    mutable_indices = [node_to_idx[n] for n in mutable_nodes]
    fixed_indices = [i for i, n in enumerate(nodes) if n not in mutable_nodes]

    kaw_specs = _neighbor_order_specs(G, node_to_idx)
    sym_pairs = _symmetry_pairs(G, nodes, mutable_nodes, node_to_idx, pair_tol)
    edge_pairs = _edge_index_pairs(G, node_to_idx)
    if not kaw_specs and not sym_pairs:
        return G, RepairStats(False, before_obj, before_obj, before_kaw, before_kaw, before_sym, before_sym, "no repair terms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_np = np.array([_coords(G, n) for n in nodes], dtype=np.float32)
    original = torch.tensor(original_np, dtype=torch.float32, device=device)
    coords = original.clone().detach().requires_grad_(True)
    mutable_idx = torch.tensor(mutable_indices, dtype=torch.long, device=device)
    fixed_idx = torch.tensor(fixed_indices, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([coords], lr=lr)

    limit = border - 20.0
    best_loss = float("inf")
    best_coords = original.detach().clone()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = _torch_loss(
            coords,
            original,
            mutable_idx,
            kaw_specs,
            sym_pairs,
            edge_pairs,
            kaw_weight,
            symmetry_weight,
            move_weight,
            min_edge_length,
            SCALE,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if fixed_idx.numel() > 0:
                coords[fixed_idx] = original[fixed_idx]

            coords[mutable_idx] = torch.clamp(coords[mutable_idx], -limit, limit)
            delta = coords[mutable_idx] - original[mutable_idx]
            delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_move / delta_norm, max=1.0)
            coords[mutable_idx] = original[mutable_idx] + delta * scale

            current = float(loss.detach().cpu().item())
            if current < best_loss:
                best_loss = current
                best_coords = coords.detach().clone()

    candidate = _apply_coords(G, nodes, best_coords.detach().cpu().numpy())
    candidate = _recompute_features(candidate)
    after_obj, after_kaw, after_sym = geometry_objective(candidate, symmetry_weight=0.5)

    if reject_crossings and has_crossings(candidate):
        return G, RepairStats(False, before_obj, after_obj, before_kaw, after_kaw, before_sym, after_sym, "crossing")

    improved = after_obj < before_obj or after_kaw < before_kaw * 0.98
    if not improved:
        return G, RepairStats(False, before_obj, after_obj, before_kaw, after_kaw, before_sym, after_sym, "not improved")

    return candidate, RepairStats(True, before_obj, after_obj, before_kaw, after_kaw, before_sym, after_sym, "accepted")


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
