"""
GA.py  —  Mori-style genetic algorithm for origami crease pattern generation v5
================================================================================
Key changes vs v4:
  • Seeds from make_random_planar_graph() (genuine exploration, not CP repair)
  • Border nodes defined by COORDINATES (x≈±195 or y≈±195), never features
  • Spatial index (RTee via shapely STRtree) for O(log E) crossing checks
  • Kawasaki repair budget reduced for speed
  • Minimum interior edges enforced at seed time
  • Population diversity injection every 20 gens (10 % fresh random seeds)
"""

import copy
import math
import pickle
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from shapely.geometry import LineString
from shapely.strtree import STRtree
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from gradient_geometry_repair import gradient_repair_kawasaki_symmetry
from kawasaki_projection_repair import repair_kawasaki_projection
from cp_io import write_cp_collection, write_cp_file
from line_graph_ga_filter import LineGraphGAFilter
try:
    from maekawa_z3_repair import repair_maekawa_z3
    Z3_IMPORT_ERROR = None
except Exception as exc:
    repair_maekawa_z3 = None
    Z3_IMPORT_ERROR = exc
import origami_constraints as oc
from topology_even_repair import repair_even_nonborder_topology

PROJECT_ROOT = Path(__file__).resolve().parent
BASE    = str(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "z3_symmetry_kawasaki"
MAX_GEN = 80
IN_CHAN = 10          # must match trained model
SCALE   = 200.0
BORDER  = SCALE - 5  # = 195.0  — coordinate threshold for border

USE_SYMMETRY = True
SYMMETRY_MODE = "vertical"   # options: "vertical", "diagonal"


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def reflect_point(x, y, mode=None):
    mode = SYMMETRY_MODE if mode is None else mode
    if mode == "vertical":
        return -x, y
    if mode == "diagonal":
        return y, x
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
    SYMMETRY_MODE = "diagonal" if mode in ("d", "diag", "diagonal") else "vertical"
    print(f"Symmetry enabled: {SYMMETRY_MODE}")

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
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
        return 0.0
    m = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 2)
    v = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 3)
    return float(abs(abs(m - v) - 2))


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
model.load_state_dict(
    torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
)
model.eval()
print(f"GNN loaded on {device}  (in_channels={IN_CHAN})")

with (DATA_DIR / "graphs.pkl").open('rb') as f:
    real_graphs = pickle.load(f)
print(f"Loaded {len(real_graphs)} real CPs for novelty reference")

line_filter = LineGraphGAFilter(min_valid_prob=0.50, penalty_weight=1.0)
print("Line-GNN filter loaded (threshold=0.50, weight=1.0)")
print("Gradient geometry repair enabled (Kawasaki + symmetry)")
print("Z3 Maekawa repair " + ("enabled" if repair_maekawa_z3 else f"unavailable: {Z3_IMPORT_ERROR}"))
print("Even-degree topology repair enabled")
print("Direct Kawasaki projection repair enabled")

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


def _coord_key(x, y, ndigits=5):
    return round(float(x), ndigits), round(float(y), ndigits)


def _overwrite_graph(dst, src):
    graph_attrs = dict(src.graph)
    dst.clear()
    dst.graph.update(graph_attrs)
    dst.add_nodes_from((n, dict(data)) for n, data in src.nodes(data=True))
    dst.add_edges_from((u, v, dict(data)) for u, v, data in src.edges(data=True))
    return dst


def rebuild_square_border(G):
    """Keep fold_type=1 only on the square boundary and rebuild side chains."""
    s = BORDER
    by_coord = {
        _coord_key(G.nodes[n]['x'], G.nodes[n]['y']): n
        for n in G.nodes()
    }
    next_id = max(G.nodes(), default=-1) + 1
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
    recompute_features(G)
    return G


def remove_crossing_edges(G, max_rounds=3):
    for _ in range(max_rounds):
        removed = False
        edges = list(G.edges(data=True))
        for i, (u1, v1, d1) in enumerate(edges):
            seg1 = LineString([
                (G.nodes[u1]['x'], G.nodes[u1]['y']),
                (G.nodes[v1]['x'], G.nodes[v1]['y']),
            ])
            for u2, v2, d2 in edges[i + 1:]:
                if len({u1, v1, u2, v2}) < 4:
                    continue
                seg2 = LineString([
                    (G.nodes[u2]['x'], G.nodes[u2]['y']),
                    (G.nodes[v2]['x'], G.nodes[v2]['y']),
                ])
                if not seg1.crosses(seg2):
                    continue
                e1_border = d1.get('fold_type') == 1
                e2_border = d2.get('fold_type') == 1
                if e1_border and not e2_border:
                    victim = (u2, v2, d2)
                elif e2_border and not e1_border:
                    victim = (u1, v1, d1)
                else:
                    victim = max(
                        [(u1, v1, d1), (u2, v2, d2)],
                        key=lambda e: math.hypot(
                            G.nodes[e[0]]['x'] - G.nodes[e[1]]['x'],
                            G.nodes[e[0]]['y'] - G.nodes[e[1]]['y'],
                        ),
                    )
                a, b, attrs = victim
                if G.has_edge(a, b):
                    G.remove_edge(a, b)
                    removed = True
                break
            if removed:
                break
        if not removed:
            break
    recompute_features(G)
    return G


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
    for u, v, data in G.edges(data=True):
        if u in H.nodes() and v in H.nodes():
            H.add_edge(u, v, **data)
    return H


def reflect_graph(G, mode=None):
    if not USE_SYMMETRY:
        out = G.copy()
        rebuild_square_border(out)
        remove_crossing_edges(out, max_rounds=3)
        return nx.convert_node_labels_to_integers(recompute_features(out))

    mode = SYMMETRY_MODE if mode is None else mode
    G2 = G.copy()
    coord_to_node = {
        _coord_key(G2.nodes[n]['x'], G2.nodes[n]['y']): n
        for n in G2.nodes()
    }
    mapping = {}
    next_id = max(G2.nodes(), default=-1) + 1

    for n in list(G.nodes()):
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        rx, ry = reflect_point(x, y, mode)
        key = _coord_key(rx, ry)
        if key in coord_to_node:
            mapping[n] = coord_to_node[key]
        else:
            G2.add_node(next_id, x=float(rx), y=float(ry))
            mapping[n] = next_id
            coord_to_node[key] = next_id
            next_id += 1

    for u, v, data in G.edges(data=True):
        mu, mv = mapping.get(u), mapping.get(v)
        if mu is not None and mv is not None and mu != mv:
            G2.add_edge(mu, mv, fold_type=data.get('fold_type', random.choice([2, 3])))

    rebuild_square_border(G2)
    remove_crossing_edges(G2, max_rounds=3)
    rebuild_square_border(G2)
    return nx.convert_node_labels_to_integers(recompute_features(G2))

# ─────────────────────────────────────────────────────────────────────────────
# Local Kawasaki repair  (reduced budget for speed)
# ─────────────────────────────────────────────────────────────────────────────

def local_kaw_repair(G, node, n_tries=15, step=4.0):
    """Nudge one interior (non-border) node to reduce Kawasaki violation."""
    if is_border_node(G, node):
        return
    lim = BORDER - 5

    def local_v():
        return _kaw_at(G, node) + sum(
            _kaw_at(G, nb)
            for nb in G.neighbors(node)
            if not is_border_node(G, nb))

    bv  = local_v()
    bx  = G.nodes[node]['x']
    by_ = G.nodes[node]['y']

    tree, segs, meta = _build_strtree(G, exclude_nodes=(node,))

    for _ in range(n_tries):
        nx_ = np.clip(bx + random.uniform(-step, step), -lim, lim)
        ny_ = np.clip(by_ + random.uniform(-step, step), -lim, lim)
        G.nodes[node]['x'] = nx_
        G.nodes[node]['y'] = ny_
        if any_incident_crosses(G, node, tree, segs, meta):
            G.nodes[node]['x'] = bx
            G.nodes[node]['y'] = by_
            continue
        v = local_v()
        if v < bv:
            bv = v; bx = nx_; by_ = ny_
    G.nodes[node]['x'] = bx
    G.nodes[node]['y'] = by_

# ─────────────────────────────────────────────────────────────────────────────
# Penalties / scores
# ─────────────────────────────────────────────────────────────────────────────

def kawasaki_penalty(G):
    return oc.kawasaki_penalty(G)


def maekawa_penalty(G):
    return oc.maekawa_penalty(G)


def symmetry_penalty(G, tol=25.0):
    if not USE_SYMMETRY:
        return 0.0
    mutable = [n for n in G.nodes() if is_mutable(G, n)]
    if not mutable:
        return 0.0
    coords  = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in mutable])
    penalty = 0.0
    for x, y in coords:
        rx, ry = reflect_point(x, y)
        dists = np.sqrt((coords[:, 0] - rx) ** 2 + (coords[:, 1] - ry) ** 2)
        if dists.min() > tol:
            penalty += 1.0
    return penalty / len(mutable)


def complexity_bonus(G):
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') != 1]
    n = len(interior_edges)
    if n < 4:
        return 0.0
    return float(min(1.0, n / 20.0))


def compute_similarity(G1, G2):
    return oc.graph_similarity(G1, G2)


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


MIN_INTERIOR_EDGES = 8
Z3_FAILURE_PENALTY = 2.0
TOPOLOGY_FAILURE_PENALTY = 2.0
KAWASAKI_GATE_START = 1.25
KAWASAKI_GATE_END = 0.30
SYMMETRY_WEIGHT_START = 0.45
SYMMETRY_WEIGHT_END = 1.10
SYMMETRY_REPAIR_INTERVAL = 5
MIRROR_MOVE_PROB = 0.90
MIRROR_NODE_TOL = 45.0
MIRROR_EDGE_TOL = 55.0


def symmetry_weight(gen=1, max_gen=MAX_GEN):
    if not USE_SYMMETRY:
        return 0.0
    t = gen / max_gen
    return SYMMETRY_WEIGHT_START * (1 - t) + SYMMETRY_WEIGHT_END * t


def invalid_fitness_tuple(
    *,
    score=-999.0,
    gnn=0.0,
    kaw=999.0,
    mae=999.0,
    sym=1.0,
    nov=1.0,
    comp=0.0,
    line_prob=0.0,
    line_pen=1.0,
    z3_pen=Z3_FAILURE_PENALTY,
    topology_pen=TOPOLOGY_FAILURE_PENALTY,
):
    return (
        score,
        gnn,
        kaw,
        mae,
        sym,
        nov,
        comp,
        line_prob,
        line_pen,
        z3_pen,
        topology_pen,
    )


def _tag_z3_stats(G, stats):
    G.graph['z3_status'] = stats.status
    G.graph['z3_changed_edges'] = stats.changed_edges
    G.graph['z3_before_mae'] = stats.before_penalty
    G.graph['z3_after_mae'] = stats.after_penalty
    G.graph['z3_constrained_vertices'] = stats.constrained_vertices
    G.graph['z3_odd_degree_vertices'] = stats.odd_degree_vertices
    G.graph['z3_unsat_reason'] = stats.unsat_reason
    return G


def _tag_topology_stats(G, stats):
    G.graph['topology_status'] = stats.status
    G.graph['topology_before_bad_vertices'] = stats.before_bad_vertices
    G.graph['topology_after_bad_vertices'] = stats.after_bad_vertices
    G.graph['topology_added_edges'] = stats.added_edges
    G.graph['topology_removed_edges'] = stats.removed_edges
    G.graph['topology_attempts'] = stats.attempts
    G.graph['topology_reason'] = stats.reason
    return G


def _tag_kawasaki_projection_stats(G, stats):
    G.graph['kawasaki_projection_accepted'] = bool(stats.accepted)
    G.graph['kawasaki_projection_attempted_vertices'] = stats.attempted_vertices
    G.graph['kawasaki_projection_accepted_vertices'] = stats.accepted_vertices
    G.graph['kawasaki_projection_before'] = stats.before_kawasaki
    G.graph['kawasaki_projection_after'] = stats.after_kawasaki
    G.graph['kawasaki_projection_crossing_rejections'] = stats.crossing_rejections
    G.graph['kawasaki_projection_reason'] = stats.reason
    return G


def z3_status_penalty(G):
    status = G.graph.get('z3_status', 'missing')
    return 0.0 if status in ('sat', 'skipped') else Z3_FAILURE_PENALTY


def topology_status_penalty(G):
    status = G.graph.get('topology_status', 'missing')
    return 0.0 if status in ('repaired', 'skipped') else TOPOLOGY_FAILURE_PENALTY


def topology_z3_then_gradient_repair(
    G,
    *,
    projection_passes=1,
    projection_top_k=3,
    projection_step=0.60,
    projection_max_move=8.0,
    grad_steps=25,
    grad_lr=0.45,
    grad_max_move=12.0,
    grad_symmetry_weight=0.85,
):
    effective_symmetry_weight = (
        grad_symmetry_weight
        if USE_SYMMETRY and SYMMETRY_MODE == "vertical"
        else 0.0
    )

    def discrete_repair(base):
        if USE_SYMMETRY:
            repair_symmetry(base)
            H = get_half_graph(base)
            H, topology_stats = repair_even_nonborder_topology(H)
            H = _tag_topology_stats(H, topology_stats)
            H, z3_stats = repair_maekawa_z3(H, timeout_ms=1000)
            H = _tag_z3_stats(H, z3_stats)
            out = reflect_graph(H)
        else:
            out, topology_stats = repair_even_nonborder_topology(base)
            out = _tag_topology_stats(out, topology_stats)
            out, z3_stats = repair_maekawa_z3(out, timeout_ms=1000)
            out = _tag_z3_stats(out, z3_stats)
        out = _tag_topology_stats(out, topology_stats)
        out = _tag_z3_stats(out, z3_stats)
        return out, topology_stats, z3_stats

    G, topology_stats, z3_stats = discrete_repair(G)
    G, projection_stats = repair_kawasaki_projection(
        G,
        passes=projection_passes,
        top_k=projection_top_k,
        step=projection_step,
        max_move=projection_max_move,
    )
    G = _tag_kawasaki_projection_stats(G, projection_stats)
    G, grad_stats = gradient_repair_kawasaki_symmetry(
        G,
        steps=grad_steps,
        lr=grad_lr,
        max_move=grad_max_move,
        symmetry_weight=effective_symmetry_weight,
    )
    if USE_SYMMETRY:
        repair_symmetry(G)
    G, topology_stats, z3_stats = discrete_repair(G)
    G = _tag_topology_stats(G, topology_stats)
    G = _tag_z3_stats(G, z3_stats)
    G = _tag_kawasaki_projection_stats(G, projection_stats)
    return G, topology_stats, z3_stats, projection_stats, grad_stats


def legacy_soft_fitness(G, gen=1, max_gen=MAX_GEN):
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') != 1]
    if len(interior_edges) < MIN_INTERIOR_EDGES:
        return -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Z3_FAILURE_PENALTY, TOPOLOGY_FAILURE_PENALTY

    gnn  = gnn_score(G)
    kaw  = kawasaki_penalty(G)
    mae  = maekawa_penalty(G)
    sym  = symmetry_penalty(G)
    nov  = novelty_penalty(G, real_graphs)
    comp = complexity_bonus(G)
    line_prob = line_filter.valid_probability(G)
    line_pen  = line_filter.penalty(G)
    z3_pen = z3_status_penalty(G)
    topology_pen = topology_status_penalty(G)

    t     = gen / max_gen
    kaw_w = 0.35 + 0.45 * t   # 0.35 → 0.80

    sym_w = symmetry_weight(gen, max_gen)

    score = (gnn
             + 0.20 * comp
             - kaw_w * kaw
             - 0.25 * mae
             - sym_w * sym
             - 0.30 * nov
             - line_pen
             - z3_pen
             - topology_pen)
    return score, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen

# ─────────────────────────────────────────────────────────────────────────────
# Random planar graph seed (fully connected, symmetric, clean border)
# ─────────────────────────────────────────────────────────────────────────────

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
    for _ in range(random.randint(2, 6)):
        side = random.choice(['bottom', 'top', 'left', 'right'])
        if side == 'bottom':
            x, y = random.uniform(-s + 20, s - 20), -s
        elif side == 'top':
            x, y = random.uniform(-s + 20, s - 20), s
        elif side == 'left':
            x, y = -s, random.uniform(-s + 20, s - 20)
        else:
            x, y = s, random.uniform(-s + 20, s - 20)

        border_coords.append((x, y))
        if USE_SYMMETRY:
            rx, ry = reflect_point(x, y)
            if math.hypot(rx - x, ry - y) > 1e-6:
                border_coords.append((rx, ry))

    # ── Interior nodes (symmetric pairs) ─────────────────────────────────────
    n_int = random.randint(4, 12) if USE_SYMMETRY else random.randint(10, 24)
    interior_coords = []
    for _ in range(n_int):
        if USE_SYMMETRY and SYMMETRY_MODE == "vertical":
            x = random.uniform(-s + margin, -margin)
            y = random.uniform(-s + margin, s - margin)
        elif USE_SYMMETRY and SYMMETRY_MODE == "diagonal":
            x = random.uniform(-s + margin, s - margin)
            y = random.uniform(-s + margin, min(x, s - margin))
        else:
            x = random.uniform(-s + margin, s - margin)
            y = random.uniform(-s + margin, s - margin)
        interior_coords.append((x, y))
        if USE_SYMMETRY:
            rx, ry = reflect_point(x, y)
            if math.hypot(rx - x, ry - y) > 1e-6:
                interior_coords.append((rx, ry))

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
        target = random.randint(2, 5)
        for v in cands:
            if placed >= target:
                break
            if try_add(u, v, random.choice([2, 3])):
                placed += 1

    # Ensure minimum interior edges — connect any isolated interior pairs
    interior_edge_count = sum(
        1 for u, v, d in G.edges(data=True) if d.get('fold_type') != 1)
    if interior_edge_count < MIN_INTERIOR_EDGES:
        for u in interior_ids:
            for v in interior_ids:
                if u >= v: continue
                if interior_edge_count >= MIN_INTERIOR_EDGES: break
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
            # Pick one node from each straggler and nearest node in main
            u = next(iter(comp))
            ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
            best_v = min(main,
                         key=lambda v: (G.nodes[v]['x'] - ux)**2 +
                                       (G.nodes[v]['y'] - uy)**2)
            if not try_add(u, best_v, random.choice([2, 3])):
                # Force it even if crossing (better connected than isolated)
                G.add_edge(u, best_v, fold_type=random.choice([2, 3]))
            main = main | comp

    G = nx.convert_node_labels_to_integers(G)
    if USE_SYMMETRY:
        G = reflect_graph(get_half_graph(G))
    else:
        rebuild_square_border(G)
        remove_crossing_edges(G, max_rounds=3)
    return recompute_features(G)

# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────────────────

MUTATION_WEIGHTS = [
    ('flip_fold',   40),
    ('move_node',   40),
    ('add_edge',    15),
    ('remove_edge',  5),
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


def find_mirror_node(G, node, candidates=None, max_dist=MIRROR_NODE_TOL):
    if not USE_SYMMETRY:
        return None
    if node not in G:
        return None
    if candidates is None:
        candidates = list(G.nodes())
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    pool = [n for n in candidates if n != node]
    if not pool:
        return None
    rx, ry = reflect_point(x, y)
    best = min(
        pool,
        key=lambda n: (G.nodes[n]['x'] - rx) ** 2 + (G.nodes[n]['y'] - ry) ** 2,
    )
    dist = math.hypot(G.nodes[best]['x'] - rx, G.nodes[best]['y'] - ry)
    return best if dist <= max_dist else None


def mirror_edge_nodes(G, u, v, max_dist=MIRROR_EDGE_TOL):
    mu = find_mirror_node(G, u, max_dist=max_dist)
    mv = find_mirror_node(G, v, max_dist=max_dist)
    if mu is None or mv is None or mu == mv:
        return None
    if {mu, mv} == {u, v}:
        return None
    return mu, mv


def try_add_mirror_edge(G, u, v, fold_type):
    if not USE_SYMMETRY:
        return False
    mirrored = mirror_edge_nodes(G, u, v)
    if mirrored is None:
        return False
    mu, mv = mirrored
    if G.has_edge(mu, mv):
        G[mu][mv]['fold_type'] = fold_type
        return True
    if edge_crosses_any(G, mu, mv):
        return False
    G.add_edge(mu, mv, fold_type=fold_type)
    return True


def mutate(G, gen=1, max_gen=MAX_GEN):
    G    = copy.deepcopy(G)
    if USE_SYMMETRY:
        G = get_half_graph(G)
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
                mirrored = None if USE_SYMMETRY else mirror_edge_nodes(G, u, v)
                if mirrored and G.has_edge(*mirrored):
                    G[mirrored[0]][mirrored[1]]['fold_type'] = nt
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
        if any_incident_crosses(G, node, tree2, segs2, meta2):
            G.nodes[node]['x'] = old_x
            G.nodes[node]['y'] = old_y
        else:
            repair_step = max(2.0, step * 0.5)
            affected = [node] + [nb for nb in G.neighbors(node)
                                  if is_mutable(G, nb)]
            for n in affected:
                local_kaw_repair(G, n, step=repair_step)

    # ── add_edge ──────────────────────────────────────────────────────────────
    elif mutation == 'add_edge' and len(mutable_nodes) > 2:
        odd  = [n for n in mutable_nodes if G.degree(n) % 2 != 0]
        pool = odd if len(odd) >= 2 else mutable_nodes
        for _ in range(20):
            u, v = random.sample(pool, 2)
            if not G.has_edge(u, v) and not edge_crosses_any(G, u, v, tree, segs, meta):
                fold_type = random.choice([2, 3])
                G.add_edge(u, v, fold_type=fold_type)
                if USE_SYMMETRY:
                    try_add_mirror_edge(G, u, v, fold_type)
                break

    # ── remove_edge ───────────────────────────────────────────────────────────
    elif mutation == 'remove_edge' and len(interior_edges) > MIN_INTERIOR_EDGES + 4:
        odd_e = [(u, v, d) for u, v, d in interior_edges
                 if G.degree(u) % 2 != 0 and G.degree(v) % 2 != 0]
        pool  = odd_e if odd_e else interior_edges
        u, v, _ = random.choice(pool)
        mirrored = None if USE_SYMMETRY else mirror_edge_nodes(G, u, v)
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        if (mirrored and G.has_edge(*mirrored) and
                len(interior_edges) > MIN_INTERIOR_EDGES + 5):
            G.remove_edge(*mirrored)

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    if USE_SYMMETRY:
        G = reflect_graph(G)
    G = recompute_features(G)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Symmetry repair
# ─────────────────────────────────────────────────────────────────────────────

def repair_symmetry(G, tol=30.0):
    if not USE_SYMMETRY:
        return G
    repaired = reflect_graph(get_half_graph(G))
    _overwrite_graph(G, repaired)
    recompute_features(G)
    return G

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

def fitness(G, gen=1, max_gen=MAX_GEN):
    """Boundary-aware, hard-gated fitness for the Kawasaki projection run."""
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') != 1]
    if len(interior_edges) < MIN_INTERIOR_EDGES:
        return invalid_fitness_tuple()

    kaw = kawasaki_penalty(G)
    mae = maekawa_penalty(G)
    z3_pen = z3_status_penalty(G)
    topology_pen = topology_status_penalty(G)

    if oc.has_crossings(G):
        return invalid_fitness_tuple(
            score=-999.0 - kaw - mae - 10.0,
            kaw=kaw,
            mae=mae,
            z3_pen=z3_pen,
            topology_pen=topology_pen,
        )

    if topology_pen > 0.0 or z3_pen > 0.0:
        return invalid_fitness_tuple(
            score=-999.0 - kaw - mae - topology_pen - z3_pen,
            kaw=kaw,
            mae=mae,
            z3_pen=z3_pen,
            topology_pen=topology_pen,
        )

    if mae > 0.01:
        return invalid_fitness_tuple(
            score=-999.0 - kaw - 5.0 * mae,
            kaw=kaw,
            mae=mae,
            z3_pen=z3_pen,
            topology_pen=topology_pen,
        )

    t = gen / max_gen
    kaw_gate = KAWASAKI_GATE_START * (1 - t) + KAWASAKI_GATE_END * t
    if kaw > kaw_gate:
        return invalid_fitness_tuple(
            score=-999.0 - kaw,
            kaw=kaw,
            mae=mae,
            z3_pen=z3_pen,
            topology_pen=topology_pen,
        )

    gnn = gnn_score(G)
    sym = symmetry_penalty(G)
    nov = novelty_penalty(G, real_graphs)
    comp = complexity_bonus(G)
    line_prob = line_filter.valid_probability(G)
    line_pen = line_filter.penalty(G)
    kaw_w = 0.35 + 0.45 * t

    sym_w = symmetry_weight(gen, max_gen)

    score = (gnn
             + 0.20 * comp
             - kaw_w * kaw
             - 0.25 * mae
             - sym_w * sym
             - 0.30 * nov
             - line_pen
             - z3_pen
             - topology_pen)
    return score, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen


def run_ga(population_size=50, generations=MAX_GEN,
           elite_keep=10, mutations_per=3):
    if repair_maekawa_z3 is None:
        raise RuntimeError(
            "Z3 Maekawa repair is required for this runner. Install it with: "
            "python -m pip install z3-solver"
        )
    print(f"\nStarting GA — pop={population_size}, gen={generations}")
    print(f"Symmetry: {'off' if not USE_SYMMETRY else SYMMETRY_MODE}")
    print("Seeding from random planar graphs (genuine exploration mode)")
    run_start = time.perf_counter()
    repair_accepts = 0
    repair_attempts = 0
    initial_topology_counts = {}
    initial_topology_removed = []
    initial_topology_added = []
    initial_z3_counts = {}
    initial_z3_changed = []
    initial_projection_accepts = 0
    initial_projection_attempts = 0
    initial_projection_before = []
    initial_projection_after = []

    # ── Build initial population from scratch ─────────────────────────────────
    population = []
    seed_start = time.perf_counter()
    for seed_idx in range(population_size):
        item_start = time.perf_counter()
        G = make_random_planar_graph()
        # Guarantee seed meets minimum interior edge count
        interior = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('fold_type') != 1]
        attempts = 0
        while len(interior) < MIN_INTERIOR_EDGES and attempts < 10:
            G = make_random_planar_graph()
            interior = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('fold_type') != 1]
            attempts += 1
        G, topology_stats, z3_stats, projection_stats, stats = topology_z3_then_gradient_repair(
            G,
            grad_steps=4,
            grad_lr=0.45,
            grad_max_move=10.0,
            grad_symmetry_weight=0.90,
        )
        initial_topology_counts[topology_stats.status] = (
            initial_topology_counts.get(topology_stats.status, 0) + 1)
        initial_topology_removed.append(topology_stats.removed_edges)
        initial_topology_added.append(topology_stats.added_edges)
        initial_z3_counts[z3_stats.status] = initial_z3_counts.get(z3_stats.status, 0) + 1
        initial_z3_changed.append(z3_stats.changed_edges)
        initial_projection_attempts += 1
        initial_projection_accepts += int(projection_stats.accepted)
        initial_projection_before.append(projection_stats.before_kawasaki)
        initial_projection_after.append(projection_stats.after_kawasaki)
        repair_attempts += 1
        repair_accepts += int(stats.accepted)
        population.append(G)
        if ((seed_idx + 1) % 3 == 0 or seed_idx == 0 or
                seed_idx + 1 == population_size):
            elapsed = time.perf_counter() - seed_start
            avg = elapsed / max(1, seed_idx + 1)
            remaining = avg * (population_size - seed_idx - 1)
            print(
                f"  Seed {seed_idx + 1}/{population_size} "
                f"done in {format_duration(time.perf_counter() - item_start)} "
                f"| seed elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(remaining)} "
                f"| KProj={projection_stats.before_kawasaki:.3f}->"
                f"{projection_stats.after_kawasaki:.3f} "
                f"Z3={z3_stats.status} Topo={topology_stats.status}",
                flush=True,
            )
    print(f"Seeded {population_size} random planar graphs")
    print(f"Initial gradient repairs accepted: {repair_accepts}/{repair_attempts}")
    print(f"Initial topology repairs: {initial_topology_counts} "
          f"avg_removed={np.mean(initial_topology_removed):.2f} "
          f"avg_added={np.mean(initial_topology_added):.2f}")
    print(f"Initial Z3 repairs: {initial_z3_counts} "
          f"avg_changed={np.mean(initial_z3_changed):.2f}")
    print(f"Initial Kawasaki projections accepted: "
          f"{initial_projection_accepts}/{initial_projection_attempts} "
          f"avg_kaw={np.mean(initial_projection_before):.3f}->"
          f"{np.mean(initial_projection_after):.3f}")

    best_scores, mean_scores, kaw_scores, sym_scores, gnn_scores, line_scores = [], [], [], [], [], []
    best_ever, best_ever_score = None, -999.0

    for gen in range(1, generations + 1):
        gen_start = time.perf_counter()
        scored   = []
        raw_fits = []
        gen_repair_accepts = 0
        gen_repair_attempts = 0
        gen_topology_counts = {}
        gen_topology_removed = []
        gen_topology_added = []
        gen_z3_counts = {}
        gen_z3_changed = []
        gen_projection_accepts = 0
        gen_projection_attempts = 0
        gen_projection_before = []
        gen_projection_after = []
        for G in population:
            result = fitness(G, gen=gen, max_gen=generations)
            f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen = result
            scored.append((f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen, G))
            raw_fits.append(f)

        shared        = shared_fitness(population, raw_fits)
        shared_scored = sorted(zip(shared, scored),
                               reverse=True, key=lambda x: x[0])
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[0]
        best_scores.append(top[0])
        mean_scores.append(float(np.mean(raw_fits)))
        kaw_scores.append(top[2])
        sym_scores.append(top[4])
        gnn_scores.append(top[1])
        line_scores.append(top[7])

        if top[0] > best_ever_score:
            best_ever_score = top[0]
            best_ever       = copy.deepcopy(top[11])

        t = gen / generations
        if gen % 5 == 0 or gen == 1:
            print(f"Gen {gen:03d} | fit={top[0]:.4f} GNN={top[1]:.3f} "
                  f"Kaw={top[2]:.3f}[w={0.35 + 0.45*t:.2f}] "
                  f"Mae={top[3]:.3f} Sym={top[4]:.3f}[w={symmetry_weight(gen, generations):.2f}] "
                  f"Nov={top[5]:.3f} Comp={top[6]:.2f} "
                  f"Line={top[7]:.3f} Lpen={top[8]:.3f} "
                  f"Z3pen={top[9]:.1f} Z3={top[11].graph.get('z3_status', '?')} "
                  f"Topo={top[11].graph.get('topology_status', '?')} "
                  f"KProj={top[11].graph.get('kawasaki_projection_reason', '?')} "
                  f"| Mean={np.mean(raw_fits):.4f}")

        # ── Build next generation ──────────────────────────────────────────
        new_pop  = [copy.deepcopy(s[11]) for s in scored[:elite_keep]]
        top_half = [s[1][11] for s in shared_scored[:population_size // 2]]

        children_total = population_size - len(new_pop)
        children_done = 0
        build_start = time.perf_counter()
        while len(new_pop) < population_size:
            child_start = time.perf_counter()
            parent = random.choice(top_half)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child, gen=gen, max_gen=generations)
            child, topology_stats, z3_stats, projection_stats, repair_stats = topology_z3_then_gradient_repair(
                child,
                grad_steps=4,
                grad_lr=0.45,
                grad_max_move=12.0,
                grad_symmetry_weight=1.00,
            )
            gen_topology_counts[topology_stats.status] = (
                gen_topology_counts.get(topology_stats.status, 0) + 1)
            gen_topology_removed.append(topology_stats.removed_edges)
            gen_topology_added.append(topology_stats.added_edges)
            gen_z3_counts[z3_stats.status] = gen_z3_counts.get(z3_stats.status, 0) + 1
            gen_z3_changed.append(z3_stats.changed_edges)
            gen_projection_attempts += 1
            gen_projection_accepts += int(projection_stats.accepted)
            gen_projection_before.append(projection_stats.before_kawasaki)
            gen_projection_after.append(projection_stats.after_kawasaki)
            gen_repair_attempts += 1
            gen_repair_accepts += int(repair_stats.accepted)
            new_pop.append(child)
            children_done += 1
            if (children_done % 5 == 0 or children_done == 1 or
                    children_done == children_total):
                elapsed = time.perf_counter() - build_start
                avg = elapsed / max(1, children_done)
                remaining = avg * (children_total - children_done)
                print(
                    f"  Gen {gen:03d} child {children_done}/{children_total} "
                    f"done in {format_duration(time.perf_counter() - child_start)} "
                    f"| build elapsed={format_duration(elapsed)} "
                    f"ETA={format_duration(remaining)} "
                    f"| KProj={projection_stats.before_kawasaki:.3f}->"
                    f"{projection_stats.after_kawasaki:.3f} "
                    f"Z3={z3_stats.status} Topo={topology_stats.status}",
                    flush=True,
                )

        # ── Inject fresh diversity every 20 generations (10 % of pop) ─────
        if gen % 20 == 0:
            n_inject = max(2, population_size // 10)
            for i in range(n_inject):
                fresh = make_random_planar_graph()
                fresh, topology_stats, z3_stats, projection_stats, repair_stats = topology_z3_then_gradient_repair(
                    fresh,
                    grad_steps=4,
                    grad_lr=0.45,
                    grad_max_move=10.0,
                    grad_symmetry_weight=0.90,
                )
                gen_topology_counts[topology_stats.status] = (
                    gen_topology_counts.get(topology_stats.status, 0) + 1)
                gen_topology_removed.append(topology_stats.removed_edges)
                gen_topology_added.append(topology_stats.added_edges)
                gen_z3_counts[z3_stats.status] = gen_z3_counts.get(z3_stats.status, 0) + 1
                gen_z3_changed.append(z3_stats.changed_edges)
                gen_projection_attempts += 1
                gen_projection_accepts += int(projection_stats.accepted)
                gen_projection_before.append(projection_stats.before_kawasaki)
                gen_projection_after.append(projection_stats.after_kawasaki)
                gen_repair_attempts += 1
                gen_repair_accepts += int(repair_stats.accepted)
                new_pop[-(i + 1)] = fresh
            print(f"  [Gen {gen}] Injected {n_inject} repaired random seeds")

        # ── Symmetry repair every 10 generations ──────────────────────────
        if USE_SYMMETRY and SYMMETRY_REPAIR_INTERVAL and gen % SYMMETRY_REPAIR_INTERVAL == 0:
            repaired_pop = []
            sym_start = time.perf_counter()
            for repair_idx, G in enumerate(new_pop, start=1):
                repair_symmetry(G)
                repaired, topology_stats, z3_stats, projection_stats, repair_stats = topology_z3_then_gradient_repair(
                    G,
                    grad_steps=6,
                    grad_lr=0.35,
                    grad_max_move=8.0,
                    grad_symmetry_weight=1.30,
                )
                gen_topology_counts[topology_stats.status] = (
                    gen_topology_counts.get(topology_stats.status, 0) + 1)
                gen_topology_removed.append(topology_stats.removed_edges)
                gen_topology_added.append(topology_stats.added_edges)
                gen_z3_counts[z3_stats.status] = gen_z3_counts.get(z3_stats.status, 0) + 1
                gen_z3_changed.append(z3_stats.changed_edges)
                gen_projection_attempts += 1
                gen_projection_accepts += int(projection_stats.accepted)
                gen_projection_before.append(projection_stats.before_kawasaki)
                gen_projection_after.append(projection_stats.after_kawasaki)
                gen_repair_attempts += 1
                gen_repair_accepts += int(repair_stats.accepted)
                repaired_pop.append(repaired)
                if (repair_idx % 5 == 0 or repair_idx == 1 or
                        repair_idx == len(new_pop)):
                    elapsed = time.perf_counter() - sym_start
                    avg = elapsed / max(1, repair_idx)
                    remaining = avg * (len(new_pop) - repair_idx)
                    print(
                        f"  Gen {gen:03d} symmetry repair "
                        f"{repair_idx}/{len(new_pop)} "
                        f"| elapsed={format_duration(elapsed)} "
                        f"ETA={format_duration(remaining)} "
                        f"| KProj={projection_stats.before_kawasaki:.3f}->"
                        f"{projection_stats.after_kawasaki:.3f} "
                        f"Z3={z3_stats.status}",
                        flush=True,
                    )
            new_pop = repaired_pop

        if gen % 5 == 0 or gen == 1:
            avg_changed = float(np.mean(gen_z3_changed)) if gen_z3_changed else 0.0
            avg_removed = float(np.mean(gen_topology_removed)) if gen_topology_removed else 0.0
            avg_added = float(np.mean(gen_topology_added)) if gen_topology_added else 0.0
            print(f"  Gradient repairs accepted this gen: "
                  f"{gen_repair_accepts}/{gen_repair_attempts}")
            print(f"  Topology repairs this gen: {gen_topology_counts} "
                  f"avg_removed={avg_removed:.2f} avg_added={avg_added:.2f}")
            print(f"  Z3 repairs this gen: {gen_z3_counts} "
                  f"avg_changed={avg_changed:.2f}")
            if gen_projection_attempts:
                print(f"  Kawasaki projections this gen: "
                      f"{gen_projection_accepts}/{gen_projection_attempts} "
                      f"avg_kaw={np.mean(gen_projection_before):.3f}->"
                      f"{np.mean(gen_projection_after):.3f}")

        gen_elapsed = time.perf_counter() - gen_start
        total_elapsed = time.perf_counter() - run_start
        avg_gen = total_elapsed / max(1, gen)
        eta = avg_gen * (generations - gen)
        print(
            f"Gen {gen:03d}/{generations} complete in "
            f"{format_duration(gen_elapsed)} | total="
            f"{format_duration(total_elapsed)} ETA={format_duration(eta)}",
            flush=True,
        )

        population = new_pop

    final_scored = []
    for G in population:
        result = fitness(G, gen=generations, max_gen=generations)
        f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen = result
        final_scored.append((f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen, G))
    final_scored.sort(key=lambda x: x[0], reverse=True)
    scored = final_scored
    if scored and scored[0][0] > best_ever_score:
        best_ever_score = scored[0][0]
        best_ever = copy.deepcopy(scored[0][11])

    print(f"\nGA complete — best fitness: {best_ever_score:.4f}")

    # ── Convergence plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(best_scores, 'b', label='best')
    axes[0, 0].plot(mean_scores, 'orange', linestyle='--', label='mean')
    axes[0, 0].set_title('Fitness'); axes[0, 0].legend()
    axes[0, 1].plot(gnn_scores, 'purple'); axes[0, 1].set_title('GNN Score (best)')
    axes[1, 0].plot(kaw_scores, 'r');      axes[1, 0].set_title('Kawasaki (best)')
    axes[1, 1].plot(sym_scores, 'g', label='symmetry')
    axes[1, 1].plot(line_scores, 'black', linestyle='--', label='line valid prob')
    axes[1, 1].set_title('Symmetry / Line-GNN')
    axes[1, 1].legend()
    for ax in axes.flatten():
        ax.set_xlabel('Generation')
    plt.tight_layout()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "z3_symmetry_kawasaki_ga_convergence.png", dpi=150)
    plt.close(fig)

    # ── Top-6 diverse results ──────────────────────────────────────────────
    final_pop  = [s[11] for s in scored]
    final_fits = [s[0] for s in scored]
    diverse6   = select_diverse_top(final_pop, final_fits, k=6, min_d=0.35)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(diverse6):
        f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen = fitness(
            G, gen=generations, max_gen=generations)
        visualise(G,
                  title=(f"Rank {i+1} fit={f:.3f} GNN={gnn:.3f} "
                         f"Kaw={kaw:.3f} Mae={mae:.3f} Z3={G.graph.get('z3_status', '?')}"),
                  ax=axes.flatten()[i])
    plt.suptitle("Top 6 Topology + Z3 + Kawasaki-Projected Crease Patterns", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "z3_symmetry_kawasaki_ga_top6.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(diverse6):
        f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, z3_pen, topology_pen = fitness(
            G, gen=generations, max_gen=generations)
        oc.visualise_with_violations(
            G,
            title=(f"Rank {i+1} Kaw={kaw:.3f} Mae={mae:.3f} "
                   f"crossing_free={not oc.has_crossings(G)}"),
            ax=axes.flatten()[i])
    plt.suptitle("Top 6 Local Constraint Violation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "z3_symmetry_kawasaki_violation_top6.png", dpi=150)
    plt.close(fig)

    with (output_dir / "z3_symmetry_kawasaki_best_generated.pkl").open('wb') as f:
        pickle.dump(best_ever, f)
    with (output_dir / "z3_symmetry_kawasaki_diverse_top6.pkl").open('wb') as f:
        pickle.dump(diverse6, f)
    write_cp_file(best_ever, output_dir / "z3_symmetry_kawasaki_best_generated.cp")
    write_cp_collection(
        diverse6, output_dir / "z3_symmetry_kawasaki_diverse_top6_cp", prefix="rank")
    print("Saved z3_symmetry_kawasaki_best_generated.pkl + z3_symmetry_kawasaki_diverse_top6.pkl + editable .cp exports")
    return best_ever, scored, diverse6


if __name__ == "__main__":
    configure_symmetry_from_input()
    best, results, diverse = run_ga(
        population_size=30,
        generations=30,
        elite_keep=6,
        mutations_per=3,
    )
