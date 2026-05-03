"""
GA.py  —  Mori-style genetic algorithm for origami crease pattern generation v6
================================================================================
Key changes vs v5:
  • Analytical Kawasaki repair via numerical gradient descent (replaces random nudge)
  • Even-degree enforcement on interior nodes (Kawasaki requires 2n creases)
  • Dedicated kaw_repair mutation (30 % weight)
  • Maekawa-aware fold assignment during seeding (|M-V|=2)
  • Kawasaki pre-optimisation on all seeds before GA starts
  • Higher Kawasaki weight in fitness: 0.60 → 1.00 (was 0.35 → 0.80)
  • Population-wide Kawasaki repair every 5 generations
"""

import copy
import math
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from cp_io import write_cp_collection, write_cp_file
from gradient_geometry_repair import gradient_repair_kawasaki_symmetry
from line_graph_ga_filter import LineGraphGAFilter
try:
    from maekawa_z3_repair import repair_maekawa_z3
except Exception:
    repair_maekawa_z3 = None
from topology_even_repair import (
    bad_maekawa_topology_vertices,
    repair_even_nonborder_topology,
)
from shapely.geometry import LineString
from shapely.strtree import STRtree
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

USE_SYMMETRY = True
SYMMETRY_MODE = "vertical"   # options: "vertical", "diagonal"

ODD_KAW_PENALTY = math.pi
MIN_INCIDENT_ANGLE = math.radians(8.0)
KAW_BAD_THRESHOLD = 0.05
KAW_TARGET_THRESHOLD = 0.02
KAW_PATCH_HOPS = 1
KAW_PATCH_MAX_NODES = 12

BASE    = r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project"
MAX_GEN = 80
IN_CHAN = 10          # must match trained model
SCALE   = 200.0
BORDER  = SCALE - 5  # = 195.0  — coordinate threshold for border

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (identical to GNN_Classifier.py v4 — do not diverge)
# ─────────────────────────────────────────────────────────────────────────────

def _angle_gaps(G, node):
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return []
    cx, cy = G.nodes[node]['x'], G.nodes[node]['y']
    angles = sorted(
        math.atan2(G.nodes[nb]['y'] - cy, G.nodes[nb]['x'] - cx)
        for nb in neighbors
    )
    n    = len(angles)
    gaps = [angles[(i + 1) % n] - angles[i] for i in range(n)]
    gaps = [g + 2 * math.pi if g < 0 else g for g in gaps]
    return gaps


def _kaw_at(G, node):
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
        return 0.0
    gaps = _angle_gaps(G, node)
    if not gaps:
        return 0.0
    n = len(gaps)
    return (abs(sum(gaps[i] for i in range(0, n, 2)) - math.pi) +
            abs(sum(gaps[i] for i in range(1, n, 2)) - math.pi))


def _mae_at(G, node):
    if is_border_node(G, node):
        return 0.0
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
        return 0.0
    m = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 2)
    v = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 3)
    return float(abs(abs(m - v) - 2))

def reflect_point(x, y, mode):
    if mode == "vertical":
        return -x, y
    elif mode == "diagonal":   # y = x
        return y, x
    else:
        return x, y


def configure_symmetry_from_input():
    global USE_SYMMETRY, SYMMETRY_MODE
    try:
        use = input("Use enforced symmetry? [y/n, default y]: ").strip().lower()
    except EOFError:
        use = ""
    USE_SYMMETRY = use not in ("n", "no", "false", "0")
    if not USE_SYMMETRY:
        print("Symmetry disabled")
        return

    try:
        mode = input("Symmetry axis? [vertical/diagonal, default vertical]: ").strip().lower()
    except EOFError:
        mode = ""
    if mode in ("d", "diag", "diagonal"):
        SYMMETRY_MODE = "diagonal"
    else:
        SYMMETRY_MODE = "vertical"
    print(f"Symmetry enabled: {SYMMETRY_MODE}")


def extract_node_features(G, scale=SCALE):
    nodes = list(G.nodes())
    feats = []
    for node in nodes:
        x = G.nodes[node]['x'] / scale
        y = G.nodes[node]['y'] / scale

        neighbors = list(G.neighbors(node))
        degree    = len(neighbors)
        is_border = 1.0 if (degree > 0 and
                             all(G[node][nb].get('fold_type') == 1
                                 for nb in neighbors)) else 0.0

        gaps = _angle_gaps(G, node)
        if gaps:
            a_mean = float(np.mean(gaps))
            a_std  = float(np.std(gaps))
            a_min  = float(np.min(gaps))
            a_max  = float(np.max(gaps))
        else:
            a_mean = a_std = a_min = a_max = 0.0

        kaw = _kaw_at(G, node)
        mae = _mae_at(G, node)

        feats.append([x, y, float(degree), is_border,
                      a_mean, a_std, a_min, a_max,
                      kaw, mae])
    return feats

# ─────────────────────────────────────────────────────────────────────────────
# GNN
# ─────────────────────────────────────────────────────────────────────────────

class GINClassifier(torch.nn.Module):
    def __init__(self, in_channels=IN_CHAN, hidden=64, num_classes=2):
        super().__init__()

        def mlp(a, b):
            return Sequential(Linear(a, b), BatchNorm1d(b), ReLU(),
                               Linear(b, b), ReLU())

        self.conv1 = GINConv(mlp(in_channels, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.conv3 = GINConv(mlp(hidden, hidden))
        self.classifier = Sequential(
            Linear(hidden * 2, hidden), ReLU(),
            torch.nn.Dropout(0.3), Linear(hidden, 2))

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return self.classifier(torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GINClassifier().to(device)
model.load_state_dict(torch.load(f"{BASE}\\best_model.pt", weights_only=False))
model.eval()
print(f"GNN loaded on {device}  (in_channels={IN_CHAN})")

with open(f"{BASE}\\graphs.pkl", 'rb') as f:
    real_graphs = pickle.load(f)
print(f"Loaded {len(real_graphs)} real CPs for novelty reference")
print("Z3 Maekawa repair " + ("enabled" if repair_maekawa_z3 else "unavailable; using heuristic folds"))
line_filter = LineGraphGAFilter(min_valid_prob=0.50, penalty_weight=1.0)
print("Line-GNN filter loaded (threshold=0.50, weight=1.0)")

# ─────────────────────────────────────────────────────────────────────────────
# Border detection — coordinate-based only (never touch these nodes)
# ─────────────────────────────────────────────────────────────────────────────

TOL = 3.0   # pixels — node is on border if within TOL of ±BORDER

def is_border_node(G, node):
    """True if the node sits on the square boundary (by coordinates)."""
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    return (abs(x - BORDER) < TOL or abs(x + BORDER) < TOL or
            abs(y - BORDER) < TOL or abs(y + BORDER) < TOL)


def is_interior(G, node):
    if is_border_node(G, node):
        return False
    nbs = list(G.neighbors(node))
    return len(nbs) >= 2 and not all(
        G[node][nb].get('fold_type') == 1 for nb in nbs)


def is_mutable(G, node):
    """Mutable = interior node NOT on the coordinate boundary."""
    return not is_border_node(G, node)

# ─────────────────────────────────────────────────────────────────────────────
# Fast crossing check with STRtree spatial index
# ─────────────────────────────────────────────────────────────────────────────

def _build_strtree(G, exclude_nodes=()):
    """Build a shapely STRtree of all edges not incident to exclude_nodes."""
    segs, meta = [], []
    for u, v in G.edges():
        if u in exclude_nodes or v in exclude_nodes:
            continue
        seg = LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                          (G.nodes[v]['x'], G.nodes[v]['y'])])
        segs.append(seg)
        meta.append((u, v))
    tree = STRtree(segs)
    return tree, segs, meta


def edge_crosses_any(G, u, v, tree=None, segs=None, meta=None):
    """Return True if edge (u,v) crosses any existing edge (excluding shared endpoints)."""
    p1 = (G.nodes[u]['x'], G.nodes[u]['y'])
    p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
    nl = LineString([p1, p2])

    if tree is None:
        # Fallback: linear scan (only used for small ad-hoc checks)
        for a, b in G.edges():
            if a in (u, v) or b in (u, v):
                continue
            if nl.crosses(LineString([(G.nodes[a]['x'], G.nodes[a]['y']),
                                       (G.nodes[b]['x'], G.nodes[b]['y'])])):
                return True
        return False

    candidates = tree.query(nl)
    for idx in candidates:
        a, b = meta[idx]
        if a in (u, v) or b in (u, v):
            continue
        if nl.crosses(segs[idx]):
            return True
    return False


def any_incident_crosses(G, node, tree=None, segs=None, meta=None):
    return any(
        edge_crosses_any(G, node, nb, tree, segs, meta)
        for nb in G.neighbors(node)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_min_gap(G, node, other):
    """Minimum angular gap at node if edge (node, other) were present."""
    cx, cy = G.nodes[node]['x'], G.nodes[node]['y']
    angles = [
        math.atan2(G.nodes[nb]['y'] - cy, G.nodes[nb]['x'] - cx)
        for nb in G.neighbors(node)
    ]
    angles.append(math.atan2(G.nodes[other]['y'] - cy,
                             G.nodes[other]['x'] - cx))
    if len(angles) < 2:
        return 2 * math.pi
    angles.sort()
    gaps = [angles[(i + 1) % len(angles)] - angles[i]
            for i in range(len(angles))]
    gaps = [g + 2 * math.pi if g < 0 else g for g in gaps]
    return min(gaps)


def respects_min_incident_angle(G, u, v, min_gap=MIN_INCIDENT_ANGLE):
    """Reject edges that create nearly duplicate rays at either endpoint."""
    return (_candidate_min_gap(G, u, v) >= min_gap and
            _candidate_min_gap(G, v, u) >= min_gap)


def _kaw_penalty_at(G, node):
    """GA-facing Kawasaki penalty for one vertex."""
    if not is_interior(G, node):
        return 0.0
    penalty = _kaw_at(G, node)
    if G.degree(node) % 2 != 0:
        penalty += ODD_KAW_PENALTY
    return penalty


def _local_kaw_objective(G, node):
    affected = {node}
    affected.update(nb for nb in G.neighbors(node) if is_interior(G, nb))
    return sum(_kaw_penalty_at(G, n) for n in affected)


def kawasaki_stats(G):
    interior = [n for n in G.nodes() if is_interior(G, n)]
    if not interior:
        return 0.0, 0.0, 0, 0
    vals = [_kaw_at(G, n) for n in interior]
    odd_count = sum(1 for n in interior if G.degree(n) % 2 != 0)
    return float(np.mean(vals)), float(np.max(vals)), odd_count, len(interior)


def recompute_features(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        cx = G.nodes[node]['x']
        cy = G.nodes[node]['y']
        angles = sorted(
            math.atan2(G.nodes[nb]['y'] - cy, G.nodes[nb]['x'] - cx)
            for nb in neighbors)
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles
    return G


def _overwrite_graph(dst, src):
    dst.clear()
    dst.add_nodes_from((n, dict(data)) for n, data in src.nodes(data=True))
    dst.add_edges_from((u, v, dict(data)) for u, v, data in src.edges(data=True))
    return dst


def topology_bad_count(G):
    return len(bad_maekawa_topology_vertices(G))


def _tag_topology_stats(G, stats):
    G.graph['topology_status'] = stats.status
    G.graph['topology_before_bad_vertices'] = stats.before_bad_vertices
    G.graph['topology_after_bad_vertices'] = stats.after_bad_vertices
    G.graph['topology_added_edges'] = stats.added_edges
    G.graph['topology_removed_edges'] = stats.removed_edges
    G.graph['topology_attempts'] = stats.attempts
    G.graph['topology_reason'] = stats.reason
    return G


def _tag_z3_stats(G, stats):
    G.graph['z3_status'] = stats.status
    G.graph['z3_changed_edges'] = stats.changed_edges
    G.graph['z3_before_mae'] = stats.before_penalty
    G.graph['z3_after_mae'] = stats.after_penalty
    G.graph['z3_constrained_vertices'] = stats.constrained_vertices
    G.graph['z3_odd_degree_vertices'] = stats.odd_degree_vertices
    G.graph['z3_unsat_reason'] = stats.unsat_reason
    return G


def apply_discrete_flatfold_repairs(
    G, *, density=True, max_added_edges=28, target_edges=None,
    solve_maekawa=True
):
    """Repair parity and fold labels before continuous Kawasaki geometry."""
    target_edges = MIN_INTERIOR_EDGES if target_edges is None else target_edges
    if density:
        add_density_edges(G, target_edges=target_edges, max_added=max_added_edges // 2)

    enforce_even_degree(G)
    repaired, topology_stats = repair_even_nonborder_topology(
        G, max_rounds=4, max_added_edges=max_added_edges)
    _overwrite_graph(G, repaired)
    _tag_topology_stats(G, topology_stats)

    if density and topology_bad_count(G) == 0:
        add_density_edges(G, target_edges=target_edges, max_added=max_added_edges // 2)
        enforce_even_degree(G)

    if not solve_maekawa:
        recompute_features(G)
        return G

    if repair_maekawa_z3 is not None:
        repaired, z3_stats = repair_maekawa_z3(G, timeout_ms=500)
        _overwrite_graph(G, repaired)
        _tag_z3_stats(G, z3_stats)
    else:
        assign_folds_maekawa(G)
        G.graph['z3_status'] = 'unavailable'
        G.graph['z3_changed_edges'] = 0
        G.graph['z3_after_mae'] = maekawa_penalty(G)

    recompute_features(G)
    return G


def _bad_kawasaki_nodes(G, threshold=KAW_BAD_THRESHOLD):
    return [
        n for n in G.nodes()
        if is_interior(G, n) and _kaw_penalty_at(G, n) > threshold
    ]


def _expand_interior_patch(G, seed_nodes, hops=KAW_PATCH_HOPS):
    visited = {n for n in seed_nodes if n in G and is_interior(G, n)}
    frontier = set(visited)
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            nxt.update(nb for nb in G.neighbors(node) if is_interior(G, nb))
        frontier = nxt - visited
        visited.update(frontier)
        if not frontier:
            break
    return visited


def _trim_patch_nodes(G, core_nodes, patch_nodes, max_nodes=KAW_PATCH_MAX_NODES):
    core = [n for n in core_nodes if n in patch_nodes]
    if len(patch_nodes) <= max_nodes:
        return set(patch_nodes)
    if len(core) >= max_nodes:
        keep = sorted(core, key=lambda n: _kaw_penalty_at(G, n), reverse=True)
        return set(keep[:max_nodes])

    def dist2_to_core(node):
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        return min(
            (G.nodes[c]['x'] - x) ** 2 + (G.nodes[c]['y'] - y) ** 2
            for c in core
        )

    keep = set(core)
    extras = sorted(
        (n for n in patch_nodes if n not in keep),
        key=lambda n: (dist2_to_core(n), -_kaw_penalty_at(G, n)),
    )
    for node in extras:
        if len(keep) >= max_nodes:
            break
        keep.add(node)
    return keep


def repair_kawasaki_patch(
    G,
    core_nodes,
    *,
    steps=45,
    lr=0.22,
    hops=KAW_PATCH_HOPS,
    max_nodes=KAW_PATCH_MAX_NODES,
    max_move=14.0,
):
    core = [n for n in core_nodes if n in G and is_interior(G, n)]
    if not core:
        return False

    patch_nodes = _expand_interior_patch(G, core, hops=hops)
    patch_nodes = _trim_patch_nodes(G, core, patch_nodes, max_nodes=max_nodes)
    mutable_nodes = [n for n in patch_nodes if is_mutable(G, n)]
    target_nodes = [n for n in patch_nodes if is_interior(G, n)]
    if not mutable_nodes or not target_nodes:
        return False

    repaired, stats = gradient_repair_kawasaki_symmetry(
        G,
        steps=steps,
        lr=lr,
        kaw_weight=1.0,
        symmetry_weight=0.0,
        move_weight=0.006,
        max_move=max_move,
        min_edge_length=6.0,
        reject_crossings=True,
        mutable_nodes=mutable_nodes,
        target_nodes=target_nodes,
        min_angle_gap_deg=8.0,
        angle_gap_weight=0.90,
        worst_weight=0.70,
        logsumexp_tau=0.20,
    )
    if stats.accepted:
        _overwrite_graph(G, repaired)
        recompute_features(G)
        return True
    return False


def repair_candidate_constraints(
    G, *, trim_edges=True, kaw_passes=1, kaw_steps=45, density=True,
    target_edges=None, solve_maekawa=True
):
    if trim_edges:
        trim_long_edges(G)
    rebuild_square_border(G)
    remove_crossing_edges(G, max_rounds=2)
    rebuild_square_border(G)
    apply_discrete_flatfold_repairs(
        G, density=density, target_edges=target_edges, solve_maekawa=solve_maekawa)
    recompute_features(G)
    full_kaw_repair(G, max_passes=kaw_passes, patch_steps=kaw_steps)
    apply_discrete_flatfold_repairs(
        G, density=False, solve_maekawa=solve_maekawa)
    recompute_features(G)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Analytical Kawasaki repair via numerical gradient descent
# ─────────────────────────────────────────────────────────────────────────────

def analytical_kaw_repair(G, node, max_iter=50, lr=3.0, tree=None, segs=None, meta=None):
    """Gradient descent on Kawasaki violation at a single interior node.
    Computes numerical gradient of the Kawasaki error w.r.t. node (x,y)
    and takes steps in the negative-gradient direction."""
    if not is_interior(G, node):
        return

    lim = BORDER - 5
    eps = 0.5  # finite-difference step for gradient

    if tree is None:
        tree, segs, meta = _build_strtree(G, exclude_nodes=(node,))

    best_x = G.nodes[node]['x']
    best_y = G.nodes[node]['y']
    best_self = _kaw_penalty_at(G, node)
    best_obj = _local_kaw_objective(G, node)
    cur_lr = lr

    for _ in range(max_iter):
        if best_self < 0.005 and best_obj < 0.02:
            break

        cx, cy = G.nodes[node]['x'], G.nodes[node]['y']

        # ∂kaw/∂x  (numerical)
        G.nodes[node]['x'] = cx + eps
        vx = _local_kaw_objective(G, node)
        G.nodes[node]['x'] = cx - eps
        vx2 = _local_kaw_objective(G, node)
        G.nodes[node]['x'] = cx

        # ∂kaw/∂y  (numerical)
        G.nodes[node]['y'] = cy + eps
        vy = _local_kaw_objective(G, node)
        G.nodes[node]['y'] = cy - eps
        vy2 = _local_kaw_objective(G, node)
        G.nodes[node]['y'] = cy

        gx = (vx - vx2) / (2 * eps)
        gy = (vy - vy2) / (2 * eps)
        norm = math.sqrt(gx * gx + gy * gy) + 1e-8

        # Step in negative-gradient direction
        nx_ = np.clip(cx - cur_lr * gx / norm, -lim, lim)
        ny_ = np.clip(cy - cur_lr * gy / norm, -lim, lim)

        G.nodes[node]['x'] = nx_
        G.nodes[node]['y'] = ny_

        if any_incident_crosses(G, node, tree, segs, meta):
            G.nodes[node]['x'] = cx
            G.nodes[node]['y'] = cy
            cur_lr *= 0.5
            continue

        obj_new = _local_kaw_objective(G, node)
        self_new = _kaw_penalty_at(G, node)
        if (obj_new < best_obj - 1e-6 or
                (abs(obj_new - best_obj) <= 1e-6 and self_new < best_self)):
            best_obj = obj_new
            best_self = self_new
            best_x = nx_
            best_y = ny_
        else:
            # Restore and shrink step
            G.nodes[node]['x'] = cx
            G.nodes[node]['y'] = cy
            cur_lr *= 0.6

    G.nodes[node]['x'] = best_x
    G.nodes[node]['y'] = best_y


def full_kaw_repair(G, max_passes=2, patch_steps=45):
    """Repair connected bad Kawasaki regions with frozen topology."""
    recompute_features(G)
    for _ in range(max_passes):
        bad = _bad_kawasaki_nodes(G)
        if not bad:
            break

        bad_sub = G.subgraph(bad).copy()
        components = list(nx.connected_components(bad_sub))
        components.sort(
            key=lambda comp: max(_kaw_penalty_at(G, n) for n in comp),
            reverse=True,
        )

        any_accept = False
        for comp in components:
            worst_here = max(_kaw_penalty_at(G, n) for n in comp)
            if worst_here <= KAW_TARGET_THRESHOLD:
                continue
            step_budget = patch_steps + 6 * max(0, min(3, len(comp) - 1))
            move_budget = 12.0 + 1.5 * min(4, len(comp))
            accepted = repair_kawasaki_patch(
                G,
                list(comp),
                steps=step_budget,
                lr=0.20,
                max_move=move_budget,
            )
            any_accept = any_accept or accepted

        if not any_accept:
            ranked = sorted(
                [(_kaw_penalty_at(G, node), node) for node in bad],
                reverse=True,
            )
            if not ranked:
                break
            before = ranked[0][0]
            analytical_kaw_repair(G, ranked[0][1])
            any_accept = _kaw_penalty_at(G, ranked[0][1]) < before - 1e-6

        worst_v = max(
            (_kaw_penalty_at(G, node) for node in G.nodes() if is_interior(G, node)),
            default=0.0,
        )
        if worst_v < KAW_TARGET_THRESHOLD or not any_accept:
            break
    recompute_features(G)


# ─────────────────────────────────────────────────────────────────────────────
# Even-degree enforcement  (Kawasaki requires 2n creases at each vertex)
# ─────────────────────────────────────────────────────────────────────────────

MAX_EDGE_LEN = 270.0   # only penalise truly extreme edges (square diagonal ~550)
MAX_EDGE_ADD = 220.0   # allow medium-length edges when adding


def _edge_len(G, u, v):
    """Euclidean length of edge (u, v)."""
    dx = G.nodes[u]['x'] - G.nodes[v]['x']
    dy = G.nodes[u]['y'] - G.nodes[v]['y']
    return math.sqrt(dx * dx + dy * dy)


def _legacy_enforce_even_degree(G):
    """Ensure all interior (non-border) nodes have even degree.
    Kawasaki's theorem requires 2n creases meeting at a vertex.
    For odd-degree nodes: try to add one SHORT edge to a nearby non-neighbor,
    or remove the longest interior edge as fallback."""
    for node in list(G.nodes()):
        if is_border_node(G, node):
            continue
        nbs = list(G.neighbors(node))
        if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
            continue
        if len(nbs) % 2 == 0:
            continue

        # ── Strategy 1: add one SHORT edge to nearest non-neighbour ──
        nx_, ny_ = G.nodes[node]['x'], G.nodes[node]['y']
        candidates = []
        for cand in G.nodes():
            if cand == node or cand in nbs:
                continue
            if is_border_node(G, cand):
                bucket = 2
            elif G.degree(cand) % 2 != 0:
                bucket = 0
            else:
                bucket = 1
            dist2 = ((G.nodes[cand]['x'] - nx_) ** 2 +
                     (G.nodes[cand]['y'] - ny_) ** 2)
            candidates.append((bucket, dist2, cand))
        candidates.sort()
        added = False
        for _, _, cand in candidates[:25]:
            dist = math.sqrt((G.nodes[cand]['x'] - nx_)**2 +
                             (G.nodes[cand]['y'] - ny_)**2)
            if dist > MAX_EDGE_ADD:
                continue
            if not respects_min_incident_angle(G, node, cand):
                continue
            if not edge_crosses_any(G, node, cand):
                m = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 2)
                v = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 3)
                ft = 3 if m > v else 2
                G.add_edge(node, cand, fold_type=ft)
                added = True
                break
        if added:
            continue

        # ── Strategy 2: remove longest interior edge ──
        int_edges = [(node, nb) for nb in nbs
                     if G[node][nb].get('fold_type') != 1]
        if int_edges:
            longest = max(int_edges, key=lambda e: _edge_len(G, *e))
            G.remove_edge(*longest)


def odd_interior_nodes(G):
    return [
        n for n in G.nodes()
        if is_interior(G, n) and not is_border_node(G, n) and G.degree(n) % 2 != 0
    ]


def _legal_new_edge(G, u, v):
    return (
        u != v
        and not G.has_edge(u, v)
        and _edge_len(G, u, v) <= MAX_EDGE_ADD
        and respects_min_incident_angle(G, u, v)
        and not edge_crosses_any(G, u, v)
    )


def enforce_even_degree(G, max_rounds=8):
    """Make true interior vertices even-valence before Kawasaki geometry repair."""
    for _ in range(max_rounds):
        odd = odd_interior_nodes(G)
        if not odd:
            break

        # Best case: add one legal crease between two odd true-interior vertices.
        candidates = []
        for i, u in enumerate(odd):
            for v in odd[i + 1:]:
                if not G.has_edge(u, v):
                    candidates.append((_edge_len(G, u, v), u, v))

        changed = False
        for dist, u, v in sorted(candidates):
            if not _legal_new_edge(G, u, v):
                continue
            G.add_edge(u, v, fold_type=random.choice([2, 3]))
            changed = True
            break
        if changed:
            continue

        # If the remaining odd count cannot be paired without crossings, attach
        # one odd interior vertex to the square boundary. Boundary vertices are
        # excluded from flat-foldability constraints, so this fixes one interior
        # parity violation without creating a new constrained odd vertex.
        boundary_nodes = [n for n in G.nodes() if is_border_node(G, n)]
        boundary_candidates = []
        for u in odd:
            for v in boundary_nodes:
                if not G.has_edge(u, v):
                    boundary_candidates.append((_edge_len(G, u, v), u, v))
        for _, u, v in sorted(boundary_candidates):
            if not _legal_new_edge(G, u, v):
                continue
            G.add_edge(u, v, fold_type=random.choice([2, 3]))
            changed = True
            break
        if changed:
            continue

        # Last resort: remove a non-border crease between two odd vertices.
        removable = []
        odd_set = set(odd)
        for u, v, d in G.edges(data=True):
            if d.get('fold_type') == 1 or u not in odd_set or v not in odd_set:
                continue
            if G.degree(u) <= 2 or G.degree(v) <= 2:
                continue
            removable.append((_edge_len(G, u, v), u, v, dict(d)))

        before_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
        for _, u, v, attrs in sorted(removable, reverse=True):
            G.remove_edge(u, v)
            after_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
            if after_components <= before_components:
                changed = True
                break
            G.add_edge(u, v, **attrs)
        if not changed:
            break


def interior_crease_edges(G):
    return [(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') != 1]


def add_density_edges(G, *, target_edges=36, max_added=12, attempts=220):
    """Add short planar crease cycles while preserving true-interior parity."""
    added = 0
    if len(interior_crease_edges(G)) >= target_edges or max_added <= 0:
        return 0

    enforce_even_degree(G)
    mutable = [n for n in G.nodes() if is_interior(G, n) and not is_border_node(G, n)]
    if len(mutable) < 3:
        return 0

    for _ in range(attempts):
        if len(interior_crease_edges(G)) >= target_edges or added >= max_added:
            break
        if topology_bad_count(G) != 0:
            enforce_even_degree(G)
            if topology_bad_count(G) != 0:
                break

        anchor = random.choice(mutable)
        ax, ay = G.nodes[anchor]['x'], G.nodes[anchor]['y']
        nearby = sorted(
            (n for n in mutable if n != anchor),
            key=lambda n: ((G.nodes[n]['x'] - ax) ** 2 +
                           (G.nodes[n]['y'] - ay) ** 2),
        )[:10]
        if len(nearby) < 2:
            continue

        a, b = random.sample(nearby, 2)
        cycle_edges = [(anchor, a), (a, b), (b, anchor)]
        new_edges = [(u, v) for u, v in cycle_edges if not G.has_edge(u, v)]
        if not new_edges:
            continue

        before_bad = topology_bad_count(G)
        placed = []
        legal = True
        for u, v in new_edges:
            if not _legal_new_edge(G, u, v):
                legal = False
                break
            G.add_edge(u, v, fold_type=random.choice([2, 3]))
            placed.append((u, v))

        if legal and topology_bad_count(G) == before_bad:
            added += len(placed)
            continue

        for u, v in reversed(placed):
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    recompute_features(G)
    return added


def trim_long_edges(G, max_len=MAX_EDGE_LEN):
    """Remove all interior edges longer than max_len. Keeps graph connected."""
    to_remove = []
    for u, v, d in G.edges(data=True):
        if d.get('fold_type') == 1:
            continue
        if _edge_len(G, u, v) > max_len:
            to_remove.append((u, v))
    # Sort longest first, remove greedily while keeping connectivity
    to_remove.sort(key=lambda e: _edge_len(G, *e), reverse=True)
    for u, v in to_remove:
        if G.has_edge(u, v) and nx.is_connected(G):
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))


# ─────────────────────────────────────────────────────────────────────────────
# Maekawa-aware fold assignment  (|M - V| = 2 at each interior vertex)
# ─────────────────────────────────────────────────────────────────────────────

def assign_folds_maekawa(G):
    """Assign mountain/valley fold types globally to best satisfy |M-V|=2
    at each interior vertex. Does a single pass, never overwrites already-set edges."""
    # First: mark all interior edges as unassigned
    unassigned = set()
    for u, v, d in G.edges(data=True):
        if d.get('fold_type') != 1:
            unassigned.add((min(u, v), max(u, v)))

    # Process interior nodes in order of degree (most constrained first)
    int_nodes = []
    for node in G.nodes():
        if is_border_node(G, node):
            continue
        int_nbs = [nb for nb in G.neighbors(node)
                   if G[node][nb].get('fold_type') != 1]
        if len(int_nbs) >= 2:
            int_nodes.append((len(int_nbs), node))
    int_nodes.sort(reverse=True)

    for _, node in int_nodes:
        int_nbs = [nb for nb in G.neighbors(node)
                   if G[node][nb].get('fold_type') != 1]
        n = len(int_nbs)
        if n < 2:
            continue

        # Count already-assigned edges at this node
        m_assigned = sum(1 for nb in int_nbs if G[node][nb]['fold_type'] == 2)
        v_assigned = sum(1 for nb in int_nbs if G[node][nb]['fold_type'] == 3)
        unset_nbs = [nb for nb in int_nbs
                     if (min(node, nb), max(node, nb)) in unassigned]

        if not unset_nbs:
            continue

        # Target: |M - V| = 2, prefer M > V, so M = (n+2)//2
        target_m = (n + 2) // 2
        need_m = max(0, target_m - m_assigned)
        need_v = max(0, len(unset_nbs) - need_m)

        random.shuffle(unset_nbs)
        for i, nb in enumerate(unset_nbs):
            ft = 2 if i < need_m else 3
            G[node][nb]['fold_type'] = ft
            key = (min(node, nb), max(node, nb))
            unassigned.discard(key)

# ─────────────────────────────────────────────────────────────────────────────
# Penalties / scores
# ─────────────────────────────────────────────────────────────────────────────

def kawasaki_penalty(G):
    mean_v, max_v, odd_count, n_interior = kawasaki_stats(G)
    if n_interior == 0:
        return 0.0
    odd_frac = odd_count / n_interior
    max_excess = max(0.0, max_v - KAW_MAX_TARGET)
    mean_excess = max(0.0, mean_v - KAW_MEAN_TARGET)
    return mean_v + 0.85 * max_v + 1.70 * max_excess + 0.80 * mean_excess + 0.75 * odd_frac


def maekawa_penalty(G):
    vals = [_mae_at(G, n) for n in G.nodes()]
    nonz = [v for v in vals if v > 0]
    return float(np.mean(nonz)) if nonz else 0.0


def symmetry_penalty(G, tol=25.0):
    if not USE_SYMMETRY:
        return 0.0
    mutable = [n for n in G.nodes() if is_mutable(G, n)]
    if not mutable:
        return 0.0
    coords  = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in mutable])
    penalty = 0.0
    for x, y in coords:
        rx, ry = reflect_point(x, y, SYMMETRY_MODE)
        dists = np.sqrt((coords[:, 0] - rx) ** 2 + (coords[:, 1] - ry) ** 2)
        if dists.min() > tol:
            penalty += 1.0
    return penalty / len(mutable)


def complexity_bonus(G):
    """Reward graphs that stay structurally rich, not just locally valid."""
    interior_edges = interior_crease_edges(G)
    n_edges = len(interior_edges)
    n_nodes = G.number_of_nodes()
    n_interior = sum(1 for n in G.nodes() if is_interior(G, n))
    if n_edges < 8:
        return 0.0

    edge_term = min(0.80, 0.80 * n_edges / TARGET_INTERIOR_EDGES)
    node_term = min(0.45, 0.45 * n_nodes / TARGET_TOTAL_NODES)
    interior_term = min(0.25, 0.25 * n_interior / max(1, TARGET_TOTAL_NODES - 8))
    return float(edge_term + node_term + interior_term)


def edge_length_penalty(G):
    """Penalise interior edges that are too long. Returns 0 if all edges < MAX_EDGE_LEN."""
    lengths = []
    for u, v, d in G.edges(data=True):
        if d.get('fold_type') == 1:
            continue
        lengths.append(_edge_len(G, u, v))
    if not lengths:
        return 0.0
    # Penalty = fraction of edges above threshold + mean overshoot
    over = [max(0.0, l - MAX_EDGE_LEN) / SCALE for l in lengths]
    frac_over = sum(1 for o in over if o > 0) / len(over)
    mean_over = float(np.mean(over))
    return frac_over + mean_over


def edge_length_stats(G):
    """Return (count, mean_len, max_len) of interior edges for diagnostics."""
    lengths = []
    for u, v, d in G.edges(data=True):
        if d.get('fold_type') == 1:
            continue
        lengths.append(_edge_len(G, u, v))
    if not lengths:
        return 0, 0.0, 0.0
    return len(lengths), float(np.mean(lengths)), float(np.max(lengths))


def clump_penalty(G, min_node_gap=18.0, min_edge_len=18.0, min_angle_deg=12.0):
    """Penalise crowded vertices, tiny creases, and nearly parallel rays."""
    mutable = [n for n in G.nodes() if is_interior(G, n) and not is_border_node(G, n)]
    if not mutable:
        return 0.0

    node_hits = 0.0
    checked = 0
    for i, u in enumerate(mutable):
        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
        for v in mutable[i + 1:]:
            vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
            d = math.hypot(ux - vx, uy - vy)
            if d < min_node_gap:
                node_hits += (min_node_gap - d) / min_node_gap
            checked += 1
    node_term = node_hits / max(1, checked)

    short_hits = []
    for u, v in interior_crease_edges(G):
        d = _edge_len(G, u, v)
        if d < min_edge_len:
            short_hits.append((min_edge_len - d) / min_edge_len)
    short_term = float(np.mean(short_hits)) if short_hits else 0.0

    min_angle = math.radians(min_angle_deg)
    angle_hits = []
    for n in mutable:
        gaps = _angle_gaps(G, n)
        if gaps:
            small = [max(0.0, min_angle - g) / min_angle for g in gaps]
            angle_hits.extend(v for v in small if v > 0.0)
    angle_term = float(np.mean(angle_hits)) if angle_hits else 0.0

    return float(2.0 * node_term + 0.8 * short_term + 1.2 * angle_term)


def node_spacing_ok(G, node, min_gap=14.0):
    if is_border_node(G, node):
        return True
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    for other in G.nodes():
        if other == node or is_border_node(G, other):
            continue
        ox, oy = G.nodes[other]['x'], G.nodes[other]['y']
        if math.hypot(x - ox, y - oy) < min_gap:
            return False
    return True


def compute_similarity(G1, G2):
    def fv(G):
        degs = sorted([d for _, d in G.degree()], reverse=True)
        n, e = G.number_of_nodes(), G.number_of_edges()
        return np.array([n, e, e / max(1, n * (n - 1) / 2)] +
                        (degs + [0] * 20)[:20], dtype=float)
    v1, v2 = fv(G1), fv(G2)
    norm   = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


def novelty_penalty(G, real_gs, sample_k=20):
    sample = random.sample(real_gs, min(sample_k, len(real_gs)))
    return max(0.0, max(compute_similarity(G, r) for r in sample) - 0.80)


def gnn_score(G):
    nodes = list(G.nodes())
    if len(nodes) < 2 or G.number_of_edges() < 2:
        return 0.0
    node_to_idx   = {n: i for i, n in enumerate(nodes)}
    node_features = extract_node_features(G)
    x_tensor      = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index    = []
    for u, v in G.edges():
        edge_index += [[node_to_idx[u], node_to_idx[v]],
                       [node_to_idx[v], node_to_idx[u]]]
    if not edge_index:
        return 0.0
    ei    = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    batch = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x_tensor, ei, batch), dim=1)
    return prob[0][1].item()


MIN_TOTAL_NODES = 36
TARGET_TOTAL_NODES = 54
SEED_MIN_INTERIOR_EDGES = 28
MIN_INTERIOR_EDGES = 42
TARGET_INTERIOR_EDGES = 66
KAW_MAX_TARGET = 0.045
KAW_MEAN_TARGET = 0.018
CLUMP_REJECT_THRESHOLD = 0.38


def projection_steps_for_gen(gen, max_gen):
    t = gen / max(1, max_gen)
    return int(22 + 26 * t)


def fitness(G, gen=1, max_gen=MAX_GEN):
    interior_edges = interior_crease_edges(G)
    kaw_mean, kaw_max, odd_count, n_interior = kawasaki_stats(G)
    topo_bad = topology_bad_count(G)
    clump = clump_penalty(G)
    if (len(interior_edges) < MIN_INTERIOR_EDGES or
            G.number_of_nodes() < MIN_TOTAL_NODES or
            odd_count > 0 or
            topo_bad > 0 or
            clump > CLUMP_REJECT_THRESHOLD):
        return -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, clump, 2.0, kaw_max

    gnn  = gnn_score(G)
    kaw  = kawasaki_penalty(G)
    mae  = maekawa_penalty(G)
    sym  = symmetry_penalty(G) if USE_SYMMETRY else 0.0
    nov  = novelty_penalty(G, real_graphs)
    comp = complexity_bonus(G)
    elen = edge_length_penalty(G)
    line_prob = line_filter.valid_probability(G)
    line_pen = line_filter.penalty(G)
    z3_status = G.graph.get('z3_status', 'missing')
    z3_pen = 0.0 if z3_status in (
        'sat', 'skipped', 'missing', 'unavailable', 'not_run'
    ) else 1.25
    topology_pen = 0.0 if G.graph.get('topology_status', 'missing') in ('repaired', 'skipped') else 1.25

    t     = gen / max_gen
    kaw_w = 0.95 + 0.85 * t
    kmax_w = 1.25 + 1.35 * t
    if kaw_max < KAW_MAX_TARGET and kaw_mean < KAW_MEAN_TARGET:
        kaw_w *= 0.55
        kmax_w *= 0.65
    kmax_excess = max(0.0, kaw_max - KAW_MAX_TARGET)
    kmean_excess = max(0.0, kaw_mean - KAW_MEAN_TARGET)

    score = (gnn
             + 0.80 * line_prob
             + 0.42 * comp
             - kaw_w * kaw
             - kmax_w * kmax_excess
             - 0.85 * kmean_excess
             - 0.35 * mae
             - 0.35 * sym
             - 0.30 * nov
             - 0.20 * elen
             - 1.10 * clump
             - 0.70 * line_pen
             - z3_pen
             - topology_pen)
    return score, gnn, kaw, mae, sym, nov, comp, elen, line_prob, clump, line_pen, kaw_max

# ─────────────────────────────────────────────────────────────────────────────
# Random planar graph seed (fully connected, symmetric, clean border)
# ─────────────────────────────────────────────────────────────────────────────

def _coord_key(x, y, ndigits=5):
    return round(float(x), ndigits), round(float(y), ndigits)


def rebuild_square_border(G):
    """Keep fold_type=1 only on the outer square and rebuild it in order."""
    s = BORDER
    next_id = max(G.nodes(), default=-1) + 1
    by_coord = {
        _coord_key(G.nodes[n]['x'], G.nodes[n]['y']): n
        for n in G.nodes()
    }
    for x, y in [(-s, -s), (s, -s), (s, s), (-s, s)]:
        key = _coord_key(x, y)
        if key not in by_coord:
            G.add_node(next_id, x=float(x), y=float(y))
            by_coord[key] = next_id
            next_id += 1

    for u, v in list(G.edges()):
        if G[u][v].get('fold_type') == 1:
            G.remove_edge(u, v)

    def side_nodes(side):
        nodes = []
        for n in G.nodes():
            x, y = G.nodes[n]['x'], G.nodes[n]['y']
            if side == 'bottom' and abs(y + s) < TOL:
                nodes.append((x, n))
            elif side == 'right' and abs(x - s) < TOL:
                nodes.append((y, n))
            elif side == 'top' and abs(y - s) < TOL:
                nodes.append((x, n))
            elif side == 'left' and abs(x + s) < TOL:
                nodes.append((y, n))
        return [n for _, n in sorted(nodes)]

    for side in ['bottom', 'right', 'top', 'left']:
        nodes = side_nodes(side)
        for u, v in zip(nodes, nodes[1:]):
            if u != v:
                G.add_edge(u, v, fold_type=1)
    return G


def remove_crossing_edges(G, max_rounds=3):
    """Drop crossing interior creases, preferring longer non-border edges."""
    for _ in range(max_rounds):
        edges = list(G.edges(data=True))
        before_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
        segments = [
            LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                        (G.nodes[v]['x'], G.nodes[v]['y'])])
            for u, v, _ in edges
        ]
        removed = False
        for i, (u1, v1, d1) in enumerate(edges):
            for j in range(i + 1, len(edges)):
                u2, v2, d2 = edges[j]
                if len({u1, v1, u2, v2}) < 4:
                    continue
                if not segments[i].crosses(segments[j]):
                    continue

                if d1.get('fold_type') == 1 and d2.get('fold_type') == 1:
                    continue
                if d1.get('fold_type') == 1:
                    choice = (u2, v2)
                elif d2.get('fold_type') == 1:
                    choice = (u1, v1)
                else:
                    choice = (
                        (u1, v1) if _edge_len(G, u1, v1) >= _edge_len(G, u2, v2)
                        else (u2, v2)
                    )

                if not G.has_edge(*choice):
                    continue
                attrs = dict(G[choice[0]][choice[1]])
                G.remove_edge(*choice)
                after_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
                if after_components > before_components:
                    G.add_edge(*choice, **attrs)
                    continue
                removed = True
                break
            if removed:
                break
        if not removed:
            break
    return G


def reflect_graph(G, mode=None):
    mode = SYMMETRY_MODE if mode is None else mode
    if not USE_SYMMETRY:
        G2 = G.copy()
        rebuild_square_border(G2)
        remove_crossing_edges(G2, max_rounds=3)
        rebuild_square_border(G2)
        apply_discrete_flatfold_repairs(G2, density=True)
        recompute_features(G2)
        return nx.convert_node_labels_to_integers(G2)

    G2 = G.copy()
    mapping = {}
    coord_to_node = {
        _coord_key(G2.nodes[n]['x'], G2.nodes[n]['y']): n
        for n in G2.nodes()
    }
    next_id = max(G2.nodes(), default=-1) + 1

    for n in list(G.nodes()):
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        rx, ry = reflect_point(x, y, mode)
        key = _coord_key(rx, ry)
        if key in coord_to_node:
            mapping[n] = coord_to_node[key]
            continue
        G2.add_node(next_id, x=float(rx), y=float(ry))
        mapping[n] = next_id
        coord_to_node[key] = next_id
        next_id += 1

    for u, v, d in G.edges(data=True):
        mu, mv = mapping.get(u), mapping.get(v)
        if mu is not None and mv is not None and mu != mv:
            G2.add_edge(mu, mv, fold_type=d.get('fold_type', random.choice([2, 3])))

    rebuild_square_border(G2)
    remove_crossing_edges(G2, max_rounds=3)
    rebuild_square_border(G2)
    apply_discrete_flatfold_repairs(G2, density=False, solve_maekawa=False)
    recompute_features(G2)
    return nx.convert_node_labels_to_integers(G2)

def make_random_planar_graph(scale=SCALE):
    """
    Build a symmetric planar CP seed from scratch with a guaranteed clean
    rectangular border and connected interior.  Used as GA seed.
    """
    s      = BORDER          # 195
    margin = 30
    tol    = 1e-6

    # ── Corner nodes (always present) ────────────────────────────────────────
    corner_coords = [(-s, -s), (s, -s), (s, s), (-s, s)]

    # ── Border nodes (symmetric pairs on bottom/top/left-right) ──────────────
    border_coords = []
    for _ in range(random.randint(3, 7)):
        side = random.choice(['bottom', 'top', 'leftright'])
        if side == 'bottom':
            x = random.uniform(-s + 20, -10)
            border_coords += [(x, -s), (-x, -s)]
        elif side == 'top':
            x = random.uniform(-s + 20, -10)
            border_coords += [(x, s), (-x, s)]
        else:
            y = random.uniform(-s + 20, s - 20)
            border_coords += [(-s, y), (s, y)]

    # ── Interior nodes (well-spread, symmetric pairs) ──────────────────────
    n_int  = random.randint(12, 18) if USE_SYMMETRY else random.randint(24, 34)
    min_spacing = 28.0  # smaller spacing so seeds can start richer
    int_half = []
    for _ in range(n_int * 10):  # oversample then filter for spacing
        if len(int_half) >= n_int:
            break
        if USE_SYMMETRY and SYMMETRY_MODE == "vertical":
            x = random.uniform(-s + margin, 0)
            y = random.uniform(-s + margin, s - margin)
        elif USE_SYMMETRY and SYMMETRY_MODE == "diagonal":
            x = random.uniform(-s + margin, s - margin)
            y = random.uniform(-s + margin, min(x, s - margin))
        else:
            x = random.uniform(-s + margin, s - margin)
            y = random.uniform(-s + margin, s - margin)
        # Check spacing against existing interior nodes
        too_close = False
        for ex, ey in int_half:
            if math.sqrt((x - ex)**2 + (y - ey)**2) < min_spacing:
                too_close = True
                break
        if not too_close:
            int_half.append((x, y))
    interior_coords = [(x, y) for x, y in int_half]

    all_coords   = corner_coords + border_coords + interior_coords
    corner_ids   = list(range(4))
    border_ids   = list(range(4, 4 + len(border_coords)))
    interior_ids = list(range(4 + len(border_coords), len(all_coords)))

    G = nx.Graph()
    for i, (x, y) in enumerate(all_coords):
        G.add_node(i, x=float(x), y=float(y))

    placed_segs = []

    def _line(u, v):
        return LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                           (G.nodes[v]['x'], G.nodes[v]['y'])])

    def _crosses(u, v):
        nl = _line(u, v)
        for seg in placed_segs:
            if nl.crosses(seg):
                return True
        return False

    def try_add(u, v, ft):
        if G.has_edge(u, v) or _crosses(u, v):
            return False
        if not respects_min_incident_angle(G, u, v):
            return False
        G.add_edge(u, v, fold_type=ft)
        placed_segs.append(_line(u, v))
        return True

    # ── Border edges: walk each side in order ─────────────────────────────────
    def on_side(side):
        pts = []
        for i in corner_ids + border_ids:
            x, y = G.nodes[i]['x'], G.nodes[i]['y']
            if side == 'bottom' and abs(y - (-s)) < tol: pts.append((x, i))
            elif side == 'right'  and abs(x - s)  < tol: pts.append((x, i))
            elif side == 'top'    and abs(y - s)   < tol: pts.append((x, i))
            elif side == 'left'   and abs(x - (-s)) < tol: pts.append((x, i))
        return pts

    for side, rev in [('bottom', False), ('right', False),
                      ('top', True), ('left', True)]:
        pts = sorted(on_side(side), key=lambda xi: xi[0], reverse=rev)
        for k in range(len(pts) - 1):
            _, u = pts[k]; _, v = pts[k + 1]
            if not G.has_edge(u, v):
                G.add_edge(u, v, fold_type=1)
                placed_segs.append(_line(u, v))

    # Guarantee corners are linked
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        if not G.has_edge(u, v):
            G.add_edge(u, v, fold_type=1)
            placed_segs.append(_line(u, v))

    # ── Interior connectivity — ensure every interior node has ≥2 crease edges ──
    all_ids = corner_ids + border_ids + interior_ids

    for u in interior_ids:
        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
        cands  = sorted([v for v in all_ids if v != u],
                        key=lambda v: (G.nodes[v]['x'] - ux)**2 +
                                      (G.nodes[v]['y'] - uy)**2)
        placed = 0
        target = random.randint(3, 4)   # richer seeds without going fully dense
        for v in cands:
            if placed >= target:
                break
            if try_add(u, v, random.choice([2, 3])):
                placed += 1

    # Ensure minimum interior edges — connect any isolated interior pairs
    interior_edge_count = sum(
        1 for u, v, d in G.edges(data=True) if d.get('fold_type') != 1)
    if interior_edge_count < SEED_MIN_INTERIOR_EDGES:
        for u in interior_ids:
            for v in interior_ids:
                if u >= v: continue
                if interior_edge_count >= SEED_MIN_INTERIOR_EDGES: break
                if try_add(u, v, random.choice([2, 3])):
                    interior_edge_count += 1

    # ── Ensure the graph is connected ────────────────────────────────────────
    # Find the largest connected component and hook up stragglers
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        main = max(comps, key=len)
        for comp in comps:
            if comp is main:
                continue
            candidates = []
            for u in comp:
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                for v in main:
                    dist2 = ((G.nodes[v]['x'] - ux) ** 2 +
                             (G.nodes[v]['y'] - uy) ** 2)
                    candidates.append((dist2, u, v))
            for _, u, v in sorted(candidates):
                if try_add(u, v, random.choice([2, 3])):
                    break
            main = main | comp

    G = nx.convert_node_labels_to_integers(G)

    # ── v6: Trim long edges, enforce even degree, Maekawa folds, Kawasaki repair ──
    repair_candidate_constraints(
        G,
        trim_edges=True,
        kaw_passes=1,
        kaw_steps=20,
        density=True,
        target_edges=SEED_MIN_INTERIOR_EDGES if USE_SYMMETRY else MIN_INTERIOR_EDGES,
    )
    G = reflect_graph(G)
    repair_candidate_constraints(
        G,
        trim_edges=False,
        kaw_passes=0 if USE_SYMMETRY else 1,
        kaw_steps=0 if USE_SYMMETRY else 22,
        density=not USE_SYMMETRY,
        solve_maekawa=True,
    )
    G = recompute_features(G)
    return G


def visualise_initial_population(population, filename="ga_initial_seeds.png"):
    """Show 6 random seeds from the initial population for debugging."""
    sample = random.sample(population, min(6, len(population)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(sample):
        n_e, avg_e, max_e = edge_length_stats(G)
        kaw = kawasaki_penalty(G)
        mae = maekawa_penalty(G)
        m_count = sum(1 for u, v, d in G.edges(data=True) if d.get('fold_type') == 2)
        v_count = sum(1 for u, v, d in G.edges(data=True) if d.get('fold_type') == 3)
        visualise(G,
                  title=(f"Seed {i+1}: N={G.number_of_nodes()} E={n_e} "
                         f"M={m_count} V={v_count}\n"
                         f"Kaw={kaw:.3f} Mae={mae:.3f} "
                         f"avgL={avg_e:.0f} maxL={max_e:.0f}"),
                  ax=axes.flatten()[i])
    plt.suptitle("Initial Population Seeds (Debug)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{BASE}\\{filename}", dpi=150)
    plt.show()
    print(f"Saved initial population visualisation to {filename}")

# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────────────────

def get_half_graph(G):
    if not USE_SYMMETRY:
        return G.copy()
    H = nx.Graph()
    for n in G.nodes():
        x, y = G.nodes[n]['x'], G.nodes[n]['y']

        if SYMMETRY_MODE == "vertical" and x <= 0:
            H.add_node(n, **G.nodes[n])
        elif SYMMETRY_MODE == "diagonal" and y <= x:
            H.add_node(n, **G.nodes[n])

    for u, v, d in G.edges(data=True):
        if u in H.nodes() and v in H.nodes():
            H.add_edge(u, v, **d)

    return H


def repair_symmetric_candidate(
    G, *, kaw_passes=1, kaw_steps=35, solve_maekawa=False
):
    if not USE_SYMMETRY:
        repair_candidate_constraints(
            G,
            trim_edges=True,
            kaw_passes=kaw_passes,
            kaw_steps=kaw_steps,
            solve_maekawa=solve_maekawa,
        )
        rebuild_square_border(G)
        remove_crossing_edges(G, max_rounds=3)
        rebuild_square_border(G)
        apply_discrete_flatfold_repairs(
            G, density=True, solve_maekawa=solve_maekawa)
        recompute_features(G)
        return G

    H = get_half_graph(G)
    repair_candidate_constraints(
        H,
        trim_edges=True,
        kaw_passes=kaw_passes,
        kaw_steps=kaw_steps,
        density=True,
        target_edges=SEED_MIN_INTERIOR_EDGES,
        solve_maekawa=solve_maekawa,
    )
    repaired = reflect_graph(H)
    repair_candidate_constraints(
        repaired,
        trim_edges=False,
        kaw_passes=0,
        kaw_steps=0,
        density=False,
        solve_maekawa=solve_maekawa,
    )
    _overwrite_graph(G, repaired)
    recompute_features(G)
    return G


def _projection_objective(kmean, kmax, odd, topo, clump):
    return (
        kmean
        + 2.2 * max(0.0, kmax - KAW_MAX_TARGET)
        + 0.8 * kmax
        + 2.5 * odd
        + 2.5 * topo
        + 0.75 * clump
    )


def project_candidate(G, *, gen=1, max_gen=MAX_GEN, label="candidate"):
    """Lightly project each candidate back toward topology + Kawasaki validity."""
    original = copy.deepcopy(G)
    before_mean, before_max, before_odd, _ = kawasaki_stats(G)
    before_topo = topology_bad_count(G)
    before_clump = clump_penalty(G)
    before_obj = _projection_objective(
        before_mean, before_max, before_odd, before_topo, before_clump)
    steps = projection_steps_for_gen(gen, max_gen)
    passes = 2 if before_max > 0.18 else 1

    repair_symmetric_candidate(
        G, kaw_passes=passes, kaw_steps=steps, solve_maekawa=False)
    apply_discrete_flatfold_repairs(G, density=False, solve_maekawa=False)
    recompute_features(G)

    after_mean, after_max, after_odd, _ = kawasaki_stats(G)
    after_topo = topology_bad_count(G)
    after_clump = clump_penalty(G)
    after_obj = _projection_objective(
        after_mean, after_max, after_odd, after_topo, after_clump)
    reverted = False
    if after_obj > before_obj + 1e-6 and before_topo == 0 and before_odd == 0:
        _overwrite_graph(G, original)
        recompute_features(G)
        after_mean, after_max, after_odd, _ = kawasaki_stats(G)
        after_topo = topology_bad_count(G)
        after_clump = clump_penalty(G)
        reverted = True

    improved = (
        after_max < before_max - 1e-6 or
        after_mean < before_mean - 1e-6 or
        after_odd < before_odd or
        after_topo < before_topo
    )
    return {
        "label": label,
        "improved": improved,
        "before_mean": before_mean,
        "before_max": before_max,
        "before_odd": before_odd,
        "before_topo": before_topo,
        "before_clump": before_clump,
        "after_mean": after_mean,
        "after_max": after_max,
        "after_odd": after_odd,
        "after_topo": after_topo,
        "after_clump": after_clump,
        "reverted": reverted,
        "z3": G.graph.get("z3_status", "missing"),
    }


def summarize_projection_stats(stats, prefix):
    if not stats:
        print(f"{prefix}: no projection stats", flush=True)
        return
    improved = sum(1 for s in stats if s["improved"])
    reverted = sum(1 for s in stats if s.get("reverted"))
    z3_counts = {}
    for s in stats:
        z3_counts[s["z3"]] = z3_counts.get(s["z3"], 0) + 1
    print(
        f"{prefix}: improved={improved}/{len(stats)} "
        f"reverted={reverted} "
        f"KMax {np.mean([s['before_max'] for s in stats]):.3f}->"
        f"{np.mean([s['after_max'] for s in stats]):.3f} "
        f"KMean {np.mean([s['before_mean'] for s in stats]):.3f}->"
        f"{np.mean([s['after_mean'] for s in stats]):.3f} "
        f"Clump {np.mean([s['before_clump'] for s in stats]):.3f}->"
        f"{np.mean([s['after_clump'] for s in stats]):.3f} "
        f"Z3={z3_counts}",
        flush=True,
    )


LEGACY_MUTATION_WEIGHTS = [
    ('flip_fold',   22),
    ('move_node',   24),
    ('kaw_repair',  22),
    ('trim_long',    5),    # v6b: much lower — only trim truly extreme edges
    ('add_edge',    22),    # v6b: higher — encourage adding structure
    ('remove_edge',  8),    # v6b: lower — discourage deleting edges
]
MUTATION_WEIGHTS = [
    ('flip_fold',   22),
    ('move_node',   24),
    ('kaw_repair',  22),
    ('trim_long',    4),
    ('add_edge',    16),
    ('remove_edge',  6),
]
_MUT_NAMES, _MUT_W = zip(*MUTATION_WEIGHTS)
_MUT_CUM = []
_c = 0
for w in _MUT_W:
    _c += w
    _MUT_CUM.append(_c)


def _choose_mutation():
    r = random.randint(1, _MUT_CUM[-1])
    for name, cum in zip(_MUT_NAMES, _MUT_CUM):
        if r <= cum:
            return name


def mutate(G, gen=1, max_gen=MAX_GEN):
    G    = copy.deepcopy(G)
    G = get_half_graph(G)   # NEW
    t    = gen / max_gen
    step = 15.0 * (1 - t) + 2.0 * t   # 15 → 2
    lim  = BORDER - 20

    edges          = list(G.edges(data=True))
    nodes          = list(G.nodes())
    interior_edges = [(u, v, d) for u, v, d in edges
                      if d.get('fold_type') != 1]
    # Mutable = not on the coordinate boundary
    mutable_nodes  = [n for n in nodes if is_mutable(G, n)]

    mutation = _choose_mutation()
    structural_cleanup = False

    # Build STRtree once per mutation for speed
    tree, segs, meta = _build_strtree(G)

    # ── flip_fold ─────────────────────────────────────────────────────────────
    if mutation == 'flip_fold' and interior_edges:
        for u, v, d in random.sample(interior_edges, min(5, len(interior_edges))):
            ot = d['fold_type']
            nt = 3 if ot == 2 else 2

            def mv_d(node, ot=ot, nt=nt):
                m  = sum(1 for nb in G.neighbors(node)
                         if G[node][nb].get('fold_type') == 2)
                vv = sum(1 for nb in G.neighbors(node)
                         if G[node][nb].get('fold_type') == 3)
                m2  = m  + (1 if nt == 2 else 0) - (1 if ot == 2 else 0)
                vv2 = vv + (1 if nt == 3 else 0) - (1 if ot == 3 else 0)
                return abs(m2 - vv2) - abs(m - vv)

            if mv_d(u) + mv_d(v) <= 0:
                G[u][v]['fold_type'] = nt
                break

    # ── move_node ─────────────────────────────────────────────────────────────
    elif mutation == 'move_node' and mutable_nodes:
        node  = random.choice(mutable_nodes)
        old_x = G.nodes[node]['x']
        old_y = G.nodes[node]['y']
        nx_   = np.clip(old_x + random.uniform(-step, step), -lim, lim)
        ny_   = np.clip(old_y + random.uniform(-step, step), -lim, lim)
        G.nodes[node]['x'] = nx_
        G.nodes[node]['y'] = ny_

        # Rebuild tree excluding this node's incident edges
        tree2, segs2, meta2 = _build_strtree(G, exclude_nodes=(node,))
        if (not node_spacing_ok(G, node, min_gap=14.0) or
                any_incident_crosses(G, node, tree2, segs2, meta2)):
            G.nodes[node]['x'] = old_x
            G.nodes[node]['y'] = old_y
        else:
            repair_kawasaki_patch(
                G,
                [node],
                steps=25,
                lr=0.24,
                max_move=max(10.0, step * 1.5),
            )

    # ── kaw_repair (v6) ───────────────────────────────────────────────────────
    elif mutation == 'kaw_repair':
        # Pick the node with the WORST Kawasaki violation and fix it
        worst_node, worst_v = None, 0.0
        for n in mutable_nodes:
            v = _kaw_penalty_at(G, n)
            if v > worst_v:
                worst_v = v
                worst_node = n
        if worst_node is not None and worst_v > 0.01:
            repaired = repair_kawasaki_patch(
                G,
                [worst_node],
                steps=35,
                lr=0.20,
                max_move=12.0,
            )
            if not repaired:
                analytical_kaw_repair(G, worst_node)

    # ── add_edge (only short edges) ────────────────────────────────────────────
    elif mutation == 'add_edge' and len(mutable_nodes) > 2:
        odd  = [n for n in mutable_nodes if G.degree(n) % 2 != 0]
        pool = odd if len(odd) >= 2 else mutable_nodes
        for _ in range(20):
            u, v = random.sample(pool, 2)
            if _edge_len(G, u, v) > MAX_EDGE_ADD:
                continue  # don't create long spaghetti edges
            if not respects_min_incident_angle(G, u, v):
                continue
            if not G.has_edge(u, v) and not edge_crosses_any(G, u, v, tree, segs, meta):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))
                structural_cleanup = True
                break

    # ── trim_long (only if edge is TRULY extreme and graph has plenty of edges) ─
    elif mutation == 'trim_long' and len(interior_edges) > MIN_INTERIOR_EDGES + 8:
        longest = max(interior_edges, key=lambda e: _edge_len(G, e[0], e[1]))
        u, v, _ = longest
        if _edge_len(G, u, v) > MAX_EDGE_LEN and G.has_edge(u, v):
            G.remove_edge(u, v)
            structural_cleanup = True

    # ── remove_edge (only when graph has MANY edges, prefer long ones) ─────────
    elif mutation == 'remove_edge' and len(interior_edges) > MIN_INTERIOR_EDGES + 8:
        by_len = sorted(interior_edges, key=lambda e: _edge_len(G, e[0], e[1]), reverse=True)
        pick = random.choice(by_len[:min(5, len(by_len))])
        u, v, _ = pick
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            structural_cleanup = True

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    if structural_cleanup:
        repair_candidate_constraints(
            G,
            trim_edges=True,
            kaw_passes=1,
            kaw_steps=24,
            density=True,
            target_edges=SEED_MIN_INTERIOR_EDGES if USE_SYMMETRY else MIN_INTERIOR_EDGES,
            solve_maekawa=False,
        )
    G = recompute_features(G)
    G = reflect_graph(G)    # rebuild full symmetric graph
    repair_candidate_constraints(
        G,
        trim_edges=False,
        kaw_passes=0,
        kaw_steps=0,
        density=not USE_SYMMETRY,
        solve_maekawa=False,
    )
    G = recompute_features(G)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Symmetry repair
# ─────────────────────────────────────────────────────────────────────────────

def repair_symmetry(G, tol=30.0):
    if not USE_SYMMETRY:
        return
    if SYMMETRY_MODE != "vertical":
        return
    lim     = BORDER - 20
    mutable = [n for n in G.nodes() if is_mutable(G, n)]
    for n in mutable:
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        if x <= 0:
            continue
        has_mirror = any(
            abs(G.nodes[m]['x'] + x) < tol and abs(G.nodes[m]['y'] - y) < tol
            for m in mutable if m != n
        )
        if has_mirror:
            continue
        left = [m for m in mutable if m != n and G.nodes[m]['x'] < 0]
        if not left:
            continue
        best_m = min(left,
                     key=lambda m: (G.nodes[m]['x'] + x)**2 +
                                   (G.nodes[m]['y'] - y)**2)
        dist = math.sqrt((G.nodes[best_m]['x'] + x)**2 +
                         (G.nodes[best_m]['y'] - y)**2)
        if dist < SCALE * 0.5:
            old_x = G.nodes[best_m]['x']
            old_y = G.nodes[best_m]['y']
            G.nodes[best_m]['x'] = np.clip(-x, -lim, lim)
            G.nodes[best_m]['y'] = np.clip(y, -lim, lim)
            if any_incident_crosses(G, best_m):
                G.nodes[best_m]['x'] = old_x
                G.nodes[best_m]['y'] = old_y

# ─────────────────────────────────────────────────────────────────────────────
# Diversity helpers
# ─────────────────────────────────────────────────────────────────────────────

def shared_fitness(population, raw_fitnesses, sigma=0.85):
    shared = []
    for i, (gi, fi) in enumerate(zip(population, raw_fitnesses)):
        s = sum((compute_similarity(gi, gj) - sigma) / (1 - sigma)
                for j, gj in enumerate(population)
                if i != j and compute_similarity(gi, gj) > sigma)
        shared.append(fi / (1 + s))
    return shared


def select_diverse_top(population, fitnesses, k=6, min_d=0.15):
    ranked   = sorted(zip(fitnesses, population), reverse=True, key=lambda x: x[0])
    selected = []
    for _, c in ranked:
        if len(selected) >= k:
            break
        if not any(compute_similarity(c, s) > (1 - min_d) for s in selected):
            selected.append(c)
    if len(selected) < k:
        for _, c in ranked:
            if c not in selected:
                selected.append(c)
            if len(selected) >= k:
                break
    return selected


def finalize_candidate_for_export(G, label):
    print(f"Final projection for {label}...", flush=True)
    for pass_idx, (kaw_passes, kaw_steps) in enumerate(((2, 60), (2, 90)), start=1):
        before_mean, before_max, before_odd, _ = kawasaki_stats(G)
        before_mae = maekawa_penalty(G)
        before_topo = topology_bad_count(G)

        repair_symmetric_candidate(
            G,
            kaw_passes=kaw_passes,
            kaw_steps=kaw_steps,
            solve_maekawa=False,
        )
        apply_discrete_flatfold_repairs(G, density=False, solve_maekawa=True)
        recompute_features(G)

        after_mean, after_max, after_odd, _ = kawasaki_stats(G)
        after_mae = maekawa_penalty(G)
        after_topo = topology_bad_count(G)
        print(
            f"  {label} pass {pass_idx}: "
            f"KMean {before_mean:.4f}->{after_mean:.4f} "
            f"KMax {before_max:.4f}->{after_max:.4f} "
            f"Odd {before_odd}->{after_odd} "
            f"TopoBad {before_topo}->{after_topo} "
            f"Mae {before_mae:.4f}->{after_mae:.4f} "
            f"Z3={G.graph.get('z3_status', 'missing')}",
            flush=True,
        )
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise(G, title="", ax=None):
    pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_edges(G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 2],
        edge_color='red', width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 3],
        edge_color='blue', width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 1],
        edge_color='black', width=2.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=8, node_color='black', ax=ax)
    ax.set_title(title, fontsize=8)
    ax.axis('equal')
    ax.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# GA
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(population_size=30, generations=MAX_GEN,
           elite_keep=8, mutations_per=3):
    print(f"\nStarting GA — pop={population_size}, gen={generations}")
    print("Seeding from random planar graphs (genuine exploration mode)")

    # ── Build initial population from scratch ─────────────────────────────────
    population = []
    seed_projection_stats = []
    for idx in range(population_size):
        G = make_random_planar_graph()
        seed_projection_stats.append(project_candidate(
            G, gen=1, max_gen=generations, label=f"seed-{idx + 1}"))
        # Guarantee seed meets minimum interior edge count
        interior = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('fold_type') != 1]
        _, _, odd_count, _ = kawasaki_stats(G)
        topo_bad = topology_bad_count(G)
        attempts = 0
        while ((len(interior) < MIN_INTERIOR_EDGES or
                G.number_of_nodes() < MIN_TOTAL_NODES or
                odd_count > 0 or
                topo_bad > 0) and
               attempts < 10):
            G = make_random_planar_graph()
            seed_projection_stats[-1] = project_candidate(
                G, gen=1, max_gen=generations, label=f"seed-{idx + 1}-retry")
            interior = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('fold_type') != 1]
            _, _, odd_count, _ = kawasaki_stats(G)
            topo_bad = topology_bad_count(G)
            attempts += 1
        population.append(G)
        if (idx + 1) % 5 == 0 or idx == 0 or idx + 1 == population_size:
            print(f"  Seeded {idx + 1}/{population_size}", flush=True)
    print(f"Seeded {population_size} random planar graphs")
    summarize_projection_stats(seed_projection_stats, "Initial projection")

    # ── Debug: visualise initial seeds ─────────────────────────────────
    visualise_initial_population(population)

    best_scores, mean_scores = [], []
    kaw_scores, kmax_scores, sym_scores, gnn_scores = [], [], [], []
    line_scores, clump_scores = [], []
    best_ever, best_ever_score = None, -999.0

    for gen in range(1, generations + 1):
        scored   = []
        raw_fits = []
        for G in population:
            result = fitness(G, gen=gen, max_gen=generations)
            f, gnn, kaw, mae, sym, nov, comp, elen, line_prob, clump, line_pen, kmax = result
            scored.append((
                f, gnn, kaw, mae, sym, nov, comp, elen,
                line_prob, clump, line_pen, kmax, G,
            ))
            raw_fits.append(f)

        shared        = shared_fitness(population, raw_fits)
        shared_scored = sorted(zip(shared, scored),
                               reverse=True, key=lambda x: x[0])
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[0]
        best_scores.append(top[0])
        mean_scores.append(float(np.mean(raw_fits)))
        kaw_scores.append(top[2])
        kmax_scores.append(top[11])
        sym_scores.append(top[4])
        gnn_scores.append(top[1])
        line_scores.append(top[8])
        clump_scores.append(top[9])

        if top[0] > best_ever_score:
            best_ever_score = top[0]
            best_ever       = copy.deepcopy(top[12])

        t = gen / generations
        if gen % 5 == 0 or gen == 1:
            # Diagnostic: edge stats for the best individual
            n_e, avg_e, max_e = edge_length_stats(top[12])
            n_nodes = top[12].number_of_nodes()
            kaw_mean, kaw_max, odd_count, _ = kawasaki_stats(top[12])
            topo_bad = topology_bad_count(top[12])
            kaw_w = 0.95 + 0.85 * t
            kmax_w = 1.25 + 1.35 * t
            if kaw_max < KAW_MAX_TARGET and kaw_mean < KAW_MEAN_TARGET:
                kaw_w *= 0.55
                kmax_w *= 0.65
            print(f"Gen {gen:03d} | fit={top[0]:.4f} GNN={top[1]:.3f} "
                  f"Line={top[8]:.3f} Kaw={top[2]:.3f}[w={kaw_w:.2f}/{kmax_w:.2f}] "
                  f"KMean={kaw_mean:.3f} KMax={kaw_max:.3f} "
                  f"Odd={odd_count} TopoBad={topo_bad} "
                  f"Mae={top[3]:.3f} Sym={top[4]:.3f} "
                  f"ELen={top[7]:.3f} Clump={top[9]:.3f} "
                  f"| N={n_nodes} E={n_e} avgL={avg_e:.0f} maxL={max_e:.0f} "
                  f"| Mean={np.mean(raw_fits):.4f}")

        # ── Build next generation ──────────────────────────────────────────
        new_pop  = [copy.deepcopy(s[12]) for s in scored[:elite_keep]]
        top_half = [s[1][12] for s in shared_scored[:population_size // 2]]

        projection_stats = []
        while len(new_pop) < population_size:
            parent = random.choice(top_half)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child, gen=gen, max_gen=generations)
            projection_stats.append(project_candidate(
                child, gen=gen, max_gen=generations, label=f"gen-{gen}-child"))
            new_pop.append(child)

        # ── Inject fresh diversity every 20 generations (10 % of pop) ─────
        if gen % 20 == 0:
            n_inject = max(2, population_size // 10)
            for i in range(n_inject):
                injected = make_random_planar_graph()
                projection_stats.append(project_candidate(
                    injected, gen=gen, max_gen=generations, label=f"gen-{gen}-inject"))
                new_pop[-(i + 1)] = injected
            print(f"  [Gen {gen}] Injected {n_inject} fresh random seeds")

        # ── Symmetry repair every 10 generations ──────────────────────────
    

        # ── v6: Population-wide Kawasaki repair every 5 generations ───────
        if gen >= 10 and gen % 10 == 0:
            elite_projection_stats = []
            for G in new_pop[:elite_keep]:
                elite_projection_stats.append(project_candidate(
                    G, gen=gen, max_gen=generations, label=f"gen-{gen}-elite"))
            summarize_projection_stats(
                elite_projection_stats, f"  Elite projection gen {gen}")

        summarize_projection_stats(projection_stats, f"  Child projection gen {gen}")

        population = new_pop

    print(f"\nGA complete — best fitness: {best_ever_score:.4f}")

    # ── Convergence plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(best_scores, 'b', label='best')
    axes[0, 0].plot(mean_scores, 'orange', linestyle='--', label='mean')
    axes[0, 0].set_title('Fitness'); axes[0, 0].legend()
    axes[0, 1].plot(gnn_scores, 'purple', label='node GNN')
    axes[0, 1].plot(line_scores, 'black', linestyle='--', label='line GNN')
    axes[0, 1].set_title('Classifier Scores (best)'); axes[0, 1].legend()
    axes[1, 0].plot(kaw_scores, 'r', label='kaw score')
    axes[1, 0].plot(kmax_scores, 'darkred', linestyle='--', label='KMax')
    axes[1, 0].set_title('Kawasaki (best)'); axes[1, 0].legend()
    axes[1, 1].plot(sym_scores, 'g', label='symmetry')
    axes[1, 1].plot(clump_scores, 'brown', linestyle='--', label='clump')
    axes[1, 1].set_title('Shape Penalties (best)'); axes[1, 1].legend()
    for ax in axes.flatten():
        ax.set_xlabel('Generation')
    plt.tight_layout()
    plt.savefig(f"{BASE}\\sym_topology_ga_convergence.png", dpi=150)
    plt.show()

    # ── Top-6 diverse results ──────────────────────────────────────────────
    final_pop  = [s[12] for s in scored]
    final_fits = [s[0] for s in scored]
    diverse6   = select_diverse_top(final_pop, final_fits, k=6)

    best_ever = finalize_candidate_for_export(best_ever, "best")
    diverse6 = [
        finalize_candidate_for_export(G, f"rank {i}")
        for i, G in enumerate(diverse6, start=1)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(diverse6):
        f, gnn, kaw, mae, sym, nov, comp, elen, line_prob, clump, line_pen, kmax = fitness(
            G, gen=generations, max_gen=generations)
        n_e, avg_e, max_e = edge_length_stats(G)
        visualise(G,
                  title=(f"Rank {i+1} fit={f:.3f} GNN={gnn:.3f} "
                         f"Line={line_prob:.3f} Kaw={kaw:.3f} KMax={kmax:.3f} "
                         f"Mae={mae:.3f}\n"
                         f"Clump={clump:.3f} N={G.number_of_nodes()} "
                         f"E={n_e} avgL={avg_e:.0f} maxL={max_e:.0f}"),
                  ax=axes.flatten()[i])
    plt.suptitle("Top 6 Generated Crease Patterns (Diverse)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{BASE}\\sym_topology_ga_top6.png", dpi=150)
    plt.show()

    with open(f"{BASE}\\sym_topology_best_generated.pkl", 'wb') as f:
        pickle.dump(best_ever, f)
    with open(f"{BASE}\\sym_topology_diverse_top6.pkl", 'wb') as f:
        pickle.dump(diverse6, f)
    write_cp_file(best_ever, f"{BASE}\\sym_topology_best_generated.cp")
    write_cp_collection(diverse6, f"{BASE}\\sym_topology_diverse_top6_cp", prefix="rank")
    print("Saved sym_topology_best_generated.pkl + sym_topology_diverse_top6.pkl + editable .cp exports")
    return best_ever, scored, diverse6


if __name__ == "__main__":
    configure_symmetry_from_input()
    best, results, diverse = run_ga(
        population_size=30,
        generations=80,
        elite_keep=8,
        mutations_per=3,
    )
