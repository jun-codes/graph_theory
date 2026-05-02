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
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from shapely.geometry import LineString
from shapely.strtree import STRtree
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from line_graph_ga_filter import LineGraphGAFilter

BASE    = str(Path(__file__).resolve().parent)
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
model.load_state_dict(torch.load(f"{BASE}\\best_model.pt", weights_only=False))
model.eval()
print(f"GNN loaded on {device}  (in_channels={IN_CHAN})")

with open(f"{BASE}\\graphs.pkl", 'rb') as f:
    real_graphs = pickle.load(f)
print(f"Loaded {len(real_graphs)} real CPs for novelty reference")

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
    vals = [_kaw_at(G, n) for n in G.nodes()]
    nonz = [v for v in vals if v > 0]
    return float(np.mean(nonz)) if nonz else 0.0


def maekawa_penalty(G):
    vals = [_mae_at(G, n) for n in G.nodes()]
    nonz = [v for v in vals if v > 0]
    return float(np.mean(nonz)) if nonz else 0.0


def symmetry_penalty(G, tol=25.0):
    mutable = [n for n in G.nodes() if is_mutable(G, n)]
    if not mutable:
        return 0.0
    coords  = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in mutable])
    penalty = 0.0
    for x, y in coords:
        dists = np.sqrt((coords[:, 0] - (-x)) ** 2 + (coords[:, 1] - y) ** 2)
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


MIN_INTERIOR_EDGES = 8


def fitness(G, gen=1, max_gen=MAX_GEN):
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') != 1]
    if len(interior_edges) < MIN_INTERIOR_EDGES:
        return -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    gnn  = gnn_score(G)
    kaw  = kawasaki_penalty(G)
    mae  = maekawa_penalty(G)
    sym  = symmetry_penalty(G)
    nov  = novelty_penalty(G, real_graphs)
    comp = complexity_bonus(G)
    line_prob = line_filter.valid_probability(G)
    line_pen  = line_filter.penalty(G)

    t     = gen / max_gen
    kaw_w = 0.35 + 0.45 * t   # 0.35 → 0.80

    score = (gnn
             + 0.20 * comp
             - kaw_w * kaw
             - 0.25 * mae
             - 0.35 * sym
             - 0.30 * nov
             - line_pen)
    return score, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen

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

    # ── Interior nodes (symmetric pairs) ─────────────────────────────────────
    n_int  = random.randint(4, 12)
    int_left = [
        (random.uniform(-s + margin, -margin),
         random.uniform(-s + margin, s - margin))
        for _ in range(n_int)
    ]
    interior_coords = int_left + [(-x, y) for x, y in int_left]

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


def mutate(G, gen=1, max_gen=MAX_GEN):
    G    = copy.deepcopy(G)
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

            # Mirror move (60 % chance) for symmetry
            if random.random() < 0.6:
                mx, my = -nx_, ny_
                best_d, mirror_node = float('inf'), None
                for n in mutable_nodes:
                    if n == node:
                        continue
                    d = (G.nodes[n]['x'] - mx)**2 + (G.nodes[n]['y'] - my)**2
                    if d < best_d:
                        best_d = d; mirror_node = n
                if mirror_node and best_d < (lim * 0.9)**2:
                    mox = G.nodes[mirror_node]['x']
                    moy = G.nodes[mirror_node]['y']
                    G.nodes[mirror_node]['x'] = np.clip(-nx_, -lim, lim)
                    G.nodes[mirror_node]['y'] = np.clip(ny_, -lim, lim)
                    tree3, segs3, meta3 = _build_strtree(
                        G, exclude_nodes=(mirror_node,))
                    if any_incident_crosses(G, mirror_node, tree3, segs3, meta3):
                        G.nodes[mirror_node]['x'] = mox
                        G.nodes[mirror_node]['y'] = moy
                    else:
                        local_kaw_repair(G, mirror_node, step=repair_step)

    # ── add_edge ──────────────────────────────────────────────────────────────
    elif mutation == 'add_edge' and len(mutable_nodes) > 2:
        odd  = [n for n in mutable_nodes if G.degree(n) % 2 != 0]
        pool = odd if len(odd) >= 2 else mutable_nodes
        for _ in range(20):
            u, v = random.sample(pool, 2)
            if not G.has_edge(u, v) and not edge_crosses_any(G, u, v, tree, segs, meta):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))
                break

    # ── remove_edge ───────────────────────────────────────────────────────────
    elif mutation == 'remove_edge' and len(interior_edges) > MIN_INTERIOR_EDGES + 4:
        odd_e = [(u, v, d) for u, v, d in interior_edges
                 if G.degree(u) % 2 != 0 and G.degree(v) % 2 != 0]
        pool  = odd_e if odd_e else interior_edges
        u, v, _ = random.choice(pool)
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = recompute_features(G)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Symmetry repair
# ─────────────────────────────────────────────────────────────────────────────

def repair_symmetry(G, tol=30.0):
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

def run_ga(population_size=50, generations=MAX_GEN,
           elite_keep=10, mutations_per=3):
    print(f"\nStarting GA — pop={population_size}, gen={generations}")
    print("Seeding from random planar graphs (genuine exploration mode)")

    # ── Build initial population from scratch ─────────────────────────────────
    population = []
    for _ in range(population_size):
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
        population.append(G)
    print(f"Seeded {population_size} random planar graphs")

    best_scores, mean_scores, kaw_scores, sym_scores, gnn_scores, line_scores = [], [], [], [], [], []
    best_ever, best_ever_score = None, -999.0

    for gen in range(1, generations + 1):
        scored   = []
        raw_fits = []
        for G in population:
            result = fitness(G, gen=gen, max_gen=generations)
            f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen = result
            scored.append((f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen, G))
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
            best_ever       = copy.deepcopy(top[9])

        t = gen / generations
        if gen % 5 == 0 or gen == 1:
            print(f"Gen {gen:03d} | fit={top[0]:.4f} GNN={top[1]:.3f} "
                  f"Kaw={top[2]:.3f}[w={0.35 + 0.45*t:.2f}] "
                  f"Mae={top[3]:.3f} Sym={top[4]:.3f} "
                  f"Nov={top[5]:.3f} Comp={top[6]:.2f} "
                  f"Line={top[7]:.3f} Lpen={top[8]:.3f} "
                  f"| Mean={np.mean(raw_fits):.4f}")

        # ── Build next generation ──────────────────────────────────────────
        new_pop  = [copy.deepcopy(s[9]) for s in scored[:elite_keep]]
        top_half = [s[1][9] for s in shared_scored[:population_size // 2]]

        while len(new_pop) < population_size:
            parent = random.choice(top_half)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child, gen=gen, max_gen=generations)
            new_pop.append(child)

        # ── Inject fresh diversity every 20 generations (10 % of pop) ─────
        if gen % 20 == 0:
            n_inject = max(2, population_size // 10)
            for i in range(n_inject):
                new_pop[-(i + 1)] = make_random_planar_graph()
            print(f"  [Gen {gen}] Injected {n_inject} fresh random seeds")

        # ── Symmetry repair every 10 generations ──────────────────────────
        if gen % 10 == 0:
            for G in new_pop:
                repair_symmetry(G)

        population = new_pop

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
    plt.savefig(f"{BASE}\\line_ga_convergence.png", dpi=150)
    plt.close(fig)

    # ── Top-6 diverse results ──────────────────────────────────────────────
    final_pop  = [s[9] for s in scored]
    final_fits = [s[0] for s in scored]
    diverse6   = select_diverse_top(final_pop, final_fits, k=6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(diverse6):
        f, gnn, kaw, mae, sym, nov, comp, line_prob, line_pen = fitness(
            G, gen=generations, max_gen=generations)
        visualise(G,
                  title=(f"Rank {i+1} fit={f:.3f} GNN={gnn:.3f} "
                         f"Kaw={kaw:.3f} Line={line_prob:.3f}"),
                  ax=axes.flatten()[i])
    plt.suptitle("Top 6 Line-Filtered Generated Crease Patterns", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{BASE}\\line_ga_top6.png", dpi=150)
    plt.close(fig)

    with open(f"{BASE}\\line_filtered_best_generated.pkl", 'wb') as f:
        pickle.dump(best_ever, f)
    with open(f"{BASE}\\line_filtered_diverse_top6.pkl", 'wb') as f:
        pickle.dump(diverse6, f)
    print("Saved line_filtered_best_generated.pkl + line_filtered_diverse_top6.pkl")
    return best_ever, scored, diverse6


if __name__ == "__main__":
    best, results, diverse = run_ga(
        population_size=30,
        generations=20,
        elite_keep=6,
        mutations_per=3,
    )
