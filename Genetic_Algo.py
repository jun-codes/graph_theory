import pickle
import torch
import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import LineString
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

BASE = r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project"

# ── GNN ───────────────────────────────────────────────────────────────────────

class GINClassifier(torch.nn.Module):
    def __init__(self, in_channels=5, hidden=64, num_classes=2):
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GINClassifier().to(device)
model.load_state_dict(torch.load(f"{BASE}\\best_model.pt", weights_only=False))
model.eval()
print(f"GNN loaded on {device}")

with open(f"{BASE}\\graphs.pkl", 'rb') as f:
    real_graphs = pickle.load(f)
print(f"Loaded {len(real_graphs)} real CPs for novelty reference")

# ── Planarity ─────────────────────────────────────────────────────────────────

def edge_crosses_any(G, u, v):
    p1       = (G.nodes[u]['x'], G.nodes[u]['y'])
    p2       = (G.nodes[v]['x'], G.nodes[v]['y'])
    new_line = LineString([p1, p2])
    for a, b in G.edges():
        if a in (u, v) or b in (u, v):
            continue
        existing = LineString([
            (G.nodes[a]['x'], G.nodes[a]['y']),
            (G.nodes[b]['x'], G.nodes[b]['y'])
        ])
        if new_line.crosses(existing):
            return True
    return False

# ── Interior check ────────────────────────────────────────────────────────────

def is_interior(G, node):
    nbs = list(G.neighbors(node))
    if len(nbs) < 2:
        return False
    return not all(G[node][nb].get('fold_type') == 1 for nb in nbs)

# ── Penalties ─────────────────────────────────────────────────────────────────

def kawasaki_penalty(G):
    penalty, count = 0.0, 0
    for node in G.nodes():
        if not is_interior(G, node):
            continue
        neighbors = list(G.neighbors(node))
        cx = G.nodes[node]['x']
        cy = G.nodes[node]['y']
        angles = sorted(
            np.arctan2(G.nodes[nb]['y'] - cy, G.nodes[nb]['x'] - cx)
            for nb in neighbors
        )
        n    = len(angles)
        gaps = [angles[(i + 1) % n] - angles[i] for i in range(n)]
        gaps = [g + 2 * np.pi if g < 0 else g for g in gaps]
        even_sum = sum(gaps[i] for i in range(0, n, 2))
        odd_sum  = sum(gaps[i] for i in range(1, n, 2))
        penalty += abs(even_sum - np.pi) + abs(odd_sum - np.pi)
        count   += 1
    return penalty / max(1, count)


def maekawa_penalty(G):
    penalty, count = 0.0, 0
    for node in G.nodes():
        if not is_interior(G, node):
            continue
        neighbors = list(G.neighbors(node))
        m    = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 2)
        v_   = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 3)
        diff = abs(m - v_)
        if diff != 2:
            penalty += abs(diff - 2)
        count += 1
    return penalty / max(1, count)


def compute_similarity(G1, G2):
    def fvec(G):
        degs     = sorted([d for _, d in G.degree()], reverse=True)
        n        = G.number_of_nodes()
        e        = G.number_of_edges()
        density  = e / max(1, n * (n - 1) / 2)
        m_count  = sum(1 for _, _, d in G.edges(data=True) if d.get('fold_type') == 2)
        v_count  = sum(1 for _, _, d in G.edges(data=True) if d.get('fold_type') == 3)
        b_count  = sum(1 for _, _, d in G.edges(data=True) if d.get('fold_type') == 1)
        mv_ratio = m_count / max(1, v_count)
        gaps = []
        for node in G.nodes():
            if not is_interior(G, node):
                continue
            nbs  = list(G.neighbors(node))
            cx   = G.nodes[node]['x']
            cy   = G.nodes[node]['y']
            angs = sorted(np.arctan2(G.nodes[nb]['y'] - cy,
                                      G.nodes[nb]['x'] - cx) for nb in nbs)
            for i in range(len(angs)):
                gap = angs[(i + 1) % len(angs)] - angs[i]
                if gap < 0:
                    gap += 2 * np.pi
                gaps.append(gap)
        ag_mean = float(np.mean(gaps)) if gaps else 0.0
        ag_std  = float(np.std(gaps))  if gaps else 0.0
        return np.array(
            [n, e, density, m_count, v_count, b_count, mv_ratio, ag_mean, ag_std]
            + (degs + [0] * 20)[:20],
            dtype=float
        )
    v1   = fvec(G1)
    v2   = fvec(G2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


def novelty_penalty(G, real_gs, sample_k=30):
    sample  = random.sample(real_gs, min(sample_k, len(real_gs)))
    max_sim = max(compute_similarity(G, r) for r in sample)
    return max(0.0, max_sim - 0.75)


def gnn_score(G):
    nodes = list(G.nodes())
    if len(nodes) < 2 or G.number_of_edges() < 2:
        return 0.0

    node_to_idx   = {n: i for i, n in enumerate(nodes)}
    node_features = []
    for node in nodes:
        angles = G.nodes[node].get('angles', [])
        node_features.append([
            G.nodes[node]['x'],
            G.nodes[node]['y'],
            G.nodes[node].get('degree', G.degree(node)),
            float(np.mean(angles)) if angles else 0.0,
            float(np.std(angles))  if angles else 0.0,
        ])

    x_tensor   = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index  = []
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])
    if not edge_index:
        return 0.0

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    batch      = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        out  = model(x_tensor, edge_index, batch)
        prob = torch.softmax(out, dim=1)
    return prob[0][1].item()


def fitness(G):
    gnn  = gnn_score(G)
    kaw  = kawasaki_penalty(G)
    mae  = maekawa_penalty(G)
    nov  = novelty_penalty(G, real_graphs)
    score = gnn - 0.35 * kaw - 0.25 * mae - 0.4 * nov
    return score, gnn, kaw, mae, nov

# ── Graph utilities ───────────────────────────────────────────────────────────

def recompute_features(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        cx = G.nodes[node]['x']
        cy = G.nodes[node]['y']
        angles = sorted(
            np.arctan2(G.nodes[nb]['y'] - cy, G.nodes[nb]['x'] - cx)
            for nb in neighbors
        )
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles
    return G


def make_random_planar_graph(scale=200.0):
    """
    Random planar graph inside a fixed square border.

    The 4 corners are always nodes 0-3, connected as border edges (fold_type=1).
    Random interior points are added, then Delaunay triangulation connects
    everything. Any edge that lies exactly on the square boundary is also
    tagged fold_type=1; all others get random mountain (2) or valley (3).
    """
    s = scale - 5  # inner edge of the square

    # Fixed square corners (always present)
    corners = np.array([
        [-s, -s],
        [ s, -s],
        [ s,  s],
        [-s,  s],
    ], dtype=float)

    # Random interior points (kept away from the border)
    n_interior = random.randint(8, 36)
    interior   = np.random.uniform(-s + 15, s - 15, (n_interior, 2))

    points = np.vstack([corners, interior])   # corners are indices 0-3
    tri    = Delaunay(points)

    # Border edges: the 4 sides of the square
    border_edges = {
        (0, 1), (1, 2), (2, 3), (0, 3)   # using sorted index pairs
    }

    G = nx.Graph()
    for i, (x, y) in enumerate(points):
        G.add_node(i, x=float(x), y=float(y))

    for simplex in tri.simplices:
        for i in range(3):
            u   = int(simplex[i])
            v   = int(simplex[(i + 1) % 3])
            key = (min(u, v), max(u, v))
            if not G.has_edge(u, v):
                ft = 1 if key in border_edges else random.choice([2, 3])
                G.add_edge(u, v, fold_type=ft)

    # Ensure the 4 border edges always exist (Delaunay may skip collinear points)
    for u, v in [(0,1),(1,2),(2,3),(0,3)]:
        if not G.has_edge(u, v):
            G.add_edge(u, v, fold_type=1)

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = recompute_features(G)
    return G

# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(G, scale=200.0):
    G        = copy.deepcopy(G)
    mutation = random.choice(['flip_fold', 'move_node', 'remove_edge', 'add_edge'])

    edges          = list(G.edges(data=True))
    nodes          = list(G.nodes())
    interior_edges = [(u, v, d) for u, v, d in edges if d.get('fold_type') != 1]
    interior_nodes = [n for n in nodes if is_interior(G, n)]

    if mutation == 'flip_fold' and interior_edges:
        n_flip = max(1, len(interior_edges) // 10)
        for u, v, d in random.sample(interior_edges, min(n_flip, len(interior_edges))):
            G[u][v]['fold_type'] = 3 if d['fold_type'] == 2 else 2

    elif mutation == 'move_node' and interior_nodes:
        node  = random.choice(interior_nodes)
        new_x = G.nodes[node]['x'] + random.uniform(-15, 15)
        new_y = G.nodes[node]['y'] + random.uniform(-15, 15)
        # Keep interior nodes away from the square border
        s = scale - 20
        G.nodes[node]['x'] = max(-s, min(s, new_x))
        G.nodes[node]['y'] = max(-s, min(s, new_y))

    elif mutation == 'remove_edge' and len(interior_edges) > 5:
        u, v, _ = random.choice(interior_edges)
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    elif mutation == 'add_edge' and len(interior_nodes) > 2:
        for _ in range(20):
            u, v = random.sample(interior_nodes, 2)
            if not G.has_edge(u, v) and not edge_crosses_any(G, u, v):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))
                break

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = recompute_features(G)
    return G

# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(G, title="Generated CP", ax=None):
    pos      = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}
    mountain = [(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 2]
    valley   = [(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 3]
    border   = [(u, v) for u, v, d in G.edges(data=True) if d.get('fold_type') == 1]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    nx.draw_networkx_edges(G, pos, edgelist=mountain, edge_color='red',   width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=valley,   edge_color='blue',  width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=border,   edge_color='black', width=2.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=8, node_color='black', ax=ax)
    ax.set_title(title, fontsize=9)
    ax.axis('equal')
    ax.axis('off')

# ── GA ────────────────────────────────────────────────────────────────────────

def run_ga(population_size=50, generations=80, elite_keep=10, mutations_per=3):
    print(f"\nStarting GA — pop={population_size}, generations={generations}")

    population = [make_random_planar_graph() for _ in range(population_size)]
    print(f"Initial population: {population_size} random planar graphs (square border) ready")

    best_scores     = []
    mean_scores     = []
    best_ever       = None
    best_ever_score = -999

    for gen in range(1, generations + 1):
        scored = []
        for G in population:
            f, gnn, kaw, mae, nov = fitness(G)
            scored.append((f, gnn, kaw, mae, nov, G))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score  = scored[0][0]
        top_gnn    = scored[0][1]
        top_kaw    = scored[0][2]
        top_mae    = scored[0][3]
        top_nov    = scored[0][4]
        mean_score = np.mean([s[0] for s in scored])

        best_scores.append(top_score)
        mean_scores.append(mean_score)

        if top_score > best_ever_score:
            best_ever_score = top_score
            best_ever       = copy.deepcopy(scored[0][5])

        if gen % 5 == 0 or gen == 1:
            print(f"Gen {gen:03d} | Best: {top_score:.4f} "
                  f"(GNN={top_gnn:.3f} Kaw={top_kaw:.3f} "
                  f"Mae={top_mae:.3f} Nov={top_nov:.3f}) "
                  f"| Mean: {mean_score:.4f}")

        new_population = [copy.deepcopy(s[5]) for s in scored[:elite_keep]]
        top_half       = [s[5] for s in scored[:population_size // 2]]
        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child)
            new_population.append(child)

        population = new_population

    print(f"\nGA complete — best fitness: {best_ever_score:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(best_scores, label='Best fitness', color='blue')
    plt.plot(mean_scores, label='Mean fitness', color='orange', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BASE}\\ga_convergence.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (f, gnn, kaw, mae, nov, G) in enumerate(scored[:6]):
        visualise(
            G,
            title=f"Rank {i+1} | fit={f:.3f} GNN={gnn:.3f} Nov={nov:.3f}",
            ax=axes.flatten()[i]
        )
    plt.suptitle("Top 6 Generated Crease Patterns", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{BASE}\\ga_top6.png", dpi=150)
    plt.show()

    with open(f"{BASE}\\best_generated.pkl", 'wb') as f_out:
        pickle.dump(best_ever, f_out)
    print("Best generated CP saved → best_generated.pkl")

    return best_ever, scored


best, results = run_ga(
    population_size=50,
    generations=80,
    elite_keep=10,
    mutations_per=3,
)