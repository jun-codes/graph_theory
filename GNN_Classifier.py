"""
GNN_Classifier.py  —  Mori-style retrain v4
============================================
Key changes vs v3:
  • Negatives are now CORRUPTED real CPs (hard negatives) instead of random
    planar graphs.  Three corruption modes:
      - fold_flip   : randomly flip mountain/valley assignments (breaks Maekawa)
      - node_jitter : perturb interior node positions (breaks Kawasaki)
      - edge_delete : remove random interior edges (breaks connectivity pattern)
    50 % of negatives use a single corruption, 50 % stack two corruptions.
    This forces the GNN to learn origami-specific structure rather than
    "does it look like a grid of lines".

  • Two extra node features (IN_CHANNELS now 10):
      8  kaw_violation  — |Σ_even gaps − π| + |Σ_odd gaps − π| at this node
      9  mae_violation  — |#mountain − #valley| − 2  at this node
    These bake the origami constraints directly into the node representation
    so the GNN has an explicit signal to work with.

  • Raw corrupted NetworkX graphs saved to negatives_v3.pkl for GA seeding.
"""

import copy
import math
import pickle
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import LineString
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

BASE = r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IN_CHANNELS = 10   # must match GINClassifier and GA.py
SCALE       = 200.0

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (shared with GA.py — keep identical)
# ─────────────────────────────────────────────────────────────────────────────

def _angle_gaps(G, node):
    """Return sorted angular gaps between consecutive neighbour directions."""
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
    """Kawasaki violation at one interior node (0 for border nodes)."""
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    # Skip border nodes (all edges are fold_type=1)
    if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
        return 0.0
    gaps = _angle_gaps(G, node)
    if not gaps:
        return 0.0
    n = len(gaps)
    return (abs(sum(gaps[i] for i in range(0, n, 2)) - math.pi) +
            abs(sum(gaps[i] for i in range(1, n, 2)) - math.pi))


def _mae_at(G, node):
    """Maekawa violation at one interior node (0 for border nodes)."""
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return 0.0
    if all(G[node][nb].get('fold_type') == 1 for nb in nbs):
        return 0.0
    m = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 2)
    v = sum(1 for nb in nbs if G[node][nb].get('fold_type') == 3)
    return float(abs(abs(m - v) - 2))


def extract_node_features(G, scale=SCALE):
    """
    10-dimensional node feature vector per node:
      0  x_norm
      1  y_norm
      2  degree
      3  is_border       — 1.0 if ALL edges are fold_type=1
      4  angle_mean      — mean of angular gaps
      5  angle_std
      6  angle_min
      7  angle_max
      8  kaw_violation   — Kawasaki residual at this node
      9  mae_violation   — Maekawa residual at this node
    """
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


def nx_to_pyg(G, label, scale=SCALE):
    nodes = list(G.nodes())
    if len(nodes) < 2 or G.number_of_edges() < 2:
        return None

    node_to_idx   = {n: i for i, n in enumerate(nodes)}
    node_features = extract_node_features(G, scale=scale)

    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y          = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# ─────────────────────────────────────────────────────────────────────────────
# Hard-negative generator — corrupt real CPs
# ─────────────────────────────────────────────────────────────────────────────

def _is_interior_node(G, node):
    nbs = list(G.neighbors(node))
    return len(nbs) >= 2 and not all(
        G[node][nb].get('fold_type') == 1 for nb in nbs)


def _corrupt_fold_flip(G, flip_prob=0.15):
    """Flip only a small fraction of folds — subtle Maekawa violation."""
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') in (2, 3)]
    # Only flip 1-3 edges max, not a blanket percentage
    n_flip = random.randint(1, max(1, len(interior_edges) // 5))
    for u, v in random.sample(interior_edges, min(n_flip, len(interior_edges))):
        G[u][v]['fold_type'] = 3 if G[u][v]['fold_type'] == 2 else 2
    return G


def _corrupt_node_jitter(G, scale=SCALE, jitter_frac=0.25):
    """
    Perturb interior node positions (breaks Kawasaki).
    Jitter magnitude scales with the paper size.
    """
    mag = scale * jitter_frac
    interior = [n for n in G.nodes() if _is_interior_node(G, n)]
    for n in interior:
        if random.random() < 0.6:
            G.nodes[n]['x'] += random.uniform(-mag, mag)
            G.nodes[n]['y'] += random.uniform(-mag, mag)
    return G


def _corrupt_edge_delete(G, delete_frac=0.35):
    """Remove a fraction of interior edges (breaks origami structure)."""
    interior_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('fold_type') != 1]
    n_del = max(1, int(len(interior_edges) * delete_frac))
    for u, v in random.sample(interior_edges, min(n_del, len(interior_edges))):
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    # Remove isolates created by deletion
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def _corrupt_edge_add_random(G, scale=SCALE, n_add=5):
    """Add random crossing interior edges (ruins planarity/structure)."""
    nodes = list(G.nodes())
    interior = [n for n in nodes if _is_interior_node(G, n)]
    if len(interior) < 2:
        return G
    for _ in range(n_add * 4):
        if n_add <= 0:
            break
        u, v = random.sample(interior, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, fold_type=random.choice([2, 3]))
            n_add -= 1
    return G


CORRUPTION_MODES = [
    _corrupt_fold_flip,
    _corrupt_node_jitter,
    _corrupt_edge_delete,
    _corrupt_edge_add_random,
]

def _corrupt_single_vertex(G, scale=SCALE):
    """
    Hardest corruption: move ONE interior vertex far enough to break
    Kawasaki at just that node and its neighbours. The rest of the graph
    looks completely valid. This forces the GNN to do real local reasoning.
    """
    interior = [n for n in G.nodes() if _is_interior_node(G, n)]
    if not interior:
        return G
    node = random.choice(interior)
    # Move it by 40-80% of scale in a random direction
    mag = scale * random.uniform(0.4, 0.8)
    angle = random.uniform(0, 2 * math.pi)
    s = scale - 5
    G.nodes[node]['x'] = float(np.clip(
        G.nodes[node]['x'] + mag * math.cos(angle), -s, s))
    G.nodes[node]['y'] = float(np.clip(
        G.nodes[node]['y'] + mag * math.sin(angle), -s, s))
    return G


def make_corrupted_cp(real_graph):
    """
    Corrupt a real CP to produce a hard negative.
    Randomly pick ONE corruption mode per negative — don't stack them,
    as stacking makes it too obvious.
    """
    G = copy.deepcopy(real_graph)

    mode = random.choice([
        lambda g: _corrupt_fold_flip(g, flip_prob=0.15),
        lambda g: _corrupt_node_jitter(g, jitter_frac=0.15),  # gentler jitter
        lambda g: _corrupt_edge_delete(g, delete_frac=0.20),  # fewer deletions
        lambda g: _corrupt_single_vertex(g),
        lambda g: _corrupt_edge_add_random(g, n_add=2),       # fewer additions
    ])
    G = mode(G)
    G = nx.convert_node_labels_to_integers(G)
    return G
# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

print("Loading real CPs (positives)...")
with open(f"{BASE}\\graphs.pkl", 'rb') as f:
    real_graphs = pickle.load(f)
print(f"  {len(real_graphs)} real CPs loaded")

# ── Negatives: all corrupted from real CPs ───────────────────────────────────
print("Generating corrupted-CP negatives (hard negatives)...")

n_neg          = len(real_graphs)
raw_negatives  = []   # NetworkX graphs for GA seeding
neg_pyg        = []   # PyG Data objects for training

attempts = 0
while len(raw_negatives) < n_neg and attempts < n_neg * 6:
    attempts += 1
    src = random.choice(real_graphs)
    G   = make_corrupted_cp(src)
    if G.number_of_nodes() < 2 or G.number_of_edges() < 2:
        continue
    pyg = nx_to_pyg(G, label=0)
    if pyg is None:
        continue
    raw_negatives.append(G)
    neg_pyg.append(pyg)
    if len(raw_negatives) % 100 == 0:
        print(f"  {len(raw_negatives)}/{n_neg} negatives generated...")

print(f"  {len(raw_negatives)} corrupted-CP negatives ready")

# Save raw negatives for GA seeding
with open(f"{BASE}\\negatives_v3.pkl", 'wb') as f:
    pickle.dump(raw_negatives, f)
print(f"Saved {len(raw_negatives)} raw corrupted-CP negatives → negatives_v3.pkl")

# ── Positives ────────────────────────────────────────────────────────────────
print("Converting real CPs to PyG...")
positives = [p for p in (nx_to_pyg(G, label=1) for G in real_graphs) if p is not None]
print(f"  {len(positives)} positives ready")

# ── Split ────────────────────────────────────────────────────────────────────
all_data = positives + neg_pyg
random.shuffle(all_data)
n       = len(all_data)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train_set = all_data[:n_train]
val_set   = all_data[n_train:n_train + n_val]
test_set  = all_data[n_train + n_val:]

print(f"\nDataset: {len(train_set)} train | {len(val_set)} val | {len(test_set)} test")
torch.save(train_set, f"{BASE}\\train_v3.pt")
torch.save(val_set,   f"{BASE}\\val_v3.pt")
torch.save(test_set,  f"{BASE}\\test_v3.pt")
print("Saved train_v3.pt / val_v3.pt / test_v3.pt")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# Model  —  IN_CHANNELS=10
# ─────────────────────────────────────────────────────────────────────────────

class GINClassifier(torch.nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, hidden=64, num_classes=2):
        super().__init__()

        def mlp(in_c, out_c):
            return Sequential(
                Linear(in_c, out_c), BatchNorm1d(out_c), ReLU(),
                Linear(out_c, out_c), ReLU()
            )

        self.conv1 = GINConv(mlp(in_channels, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.conv3 = GINConv(mlp(hidden, hidden))
        self.classifier = Sequential(
            Linear(hidden * 2, hidden), ReLU(),
            torch.nn.Dropout(0.3), Linear(hidden, 2)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.classifier(x)


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
model     = GINClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(loader):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        correct    += out.argmax(dim=1).eq(batch.y.squeeze()).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            out     = model(batch.x, batch.edge_index, batch.batch)
            correct += out.argmax(dim=1).eq(batch.y.squeeze()).sum().item()
    return correct / len(loader.dataset)


best_val_acc = 0.0
epochs       = 120  # more epochs since the problem is harder now
print(f"\nTraining for {epochs} epochs...\n")

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(train_loader)
    val_acc               = evaluate(val_loader)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{BASE}\\best_model.pt")

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Best Val: {best_val_acc:.4f}")

model.load_state_dict(torch.load(f"{BASE}\\best_model.pt", weights_only=False))
test_acc = evaluate(test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Best Val Accuracy:   {best_val_acc:.4f}")
print("\nNOTE: best_model.pt now expects IN_CHANNELS=10.")
print("GA.py uses the same extract_node_features() — both files are in sync.")