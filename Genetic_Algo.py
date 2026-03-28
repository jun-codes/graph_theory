import pickle
import torch
import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
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

# ── Load real CPs once at startup ─────────────────────────────────────────────

with open(f"{BASE}\\graphs.pkl", 'rb') as _f:
    real_graphs = pickle.load(_f)
print(f"Loaded {len(real_graphs)} real CPs for novelty reference")

# ── Planarity helpers ─────────────────────────────────────────────────────────

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


def any_incident_crosses(G, node):
    for nb in G.neighbors(node):
        if edge_crosses_any(G, node, nb):
            return True
    return False

# ── Kawasaki helpers ──────────────────────────────────────────────────────────

def kaw_at(G, node):
    """
    Kawasaki violation at a single interior vertex.
    Returns 0.0 for boundary vertices or vertices with <2 neighbours.
    """
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    if any(G[node][nb].get('fold_type') == 1 for nb in neighbors):
        return 0.0  # boundary vertex
    nx_c = G.nodes[node]['x']
    ny_c = G.nodes[node]['y']
    angles = sorted(
        np.arctan2(G.nodes[nb]['y'] - ny_c, G.nodes[nb]['x'] - nx_c)
        for nb in neighbors
    )
    n    = len(angles)
    gaps = [angles[(i + 1) % n] - angles[i] for i in range(n)]
    gaps = [g + 2 * np.pi if g < 0 else g for g in gaps]
    even_sum = sum(gaps[i] for i in range(0, n, 2))
    odd_sum  = sum(gaps[i] for i in range(1, n, 2))
    return abs(even_sum - np.pi) + abs(odd_sum - np.pi)


def kaw_ok(G, nodes_to_check, eps=0.05):
    """
    Return True if Kawasaki is satisfied (within eps) at every node in
    nodes_to_check. Used to validate mutations before committing them.
    """
    return all(kaw_at(G, n) <= eps for n in nodes_to_check)


def kawasaki_penalty(G):
    """Global mean Kawasaki violation — used in fitness logging only."""
    total, count = 0.0, 0
    for node in G.nodes():
        v = kaw_at(G, node)
        if v > 0:
            total += v
            count += 1
    return total / max(1, count)

# ── Kawasaki repair (run once on initial population) ─────────────────────────

def repair_kawasaki(G, max_iters=600, step=8.0, scale=200.0):
    """
    Iteratively move interior nodes to satisfy Kawasaki at every vertex.
    Each iteration targets the worst-violating node and hill-climbs its
    position with 30 random perturbations, keeping the best improvement.
    Also checks that the move does not introduce any crossings.
    Stops when all violations are below eps=0.05.
    """
    G = copy.deepcopy(G)

    def interior_nodes():
        return [
            n for n in G.nodes()
            if not any(G[n][nb].get('fold_type') == 1 for nb in G.neighbors(n))
            and len(list(G.neighbors(n))) >= 2
        ]

    for _ in range(max_iters):
        inodes = interior_nodes()
        if not inodes:
            break

        # Violation at node = kaw_at(node) + sum of kaw_at(neighbour) for all
        # neighbours, because moving a node changes angles at its neighbours too
        def total_local_viol(node):
            v = kaw_at(G, node)
            for nb in G.neighbors(node):
                v += kaw_at(G, nb)
            return v

        worst_node = max(inodes, key=total_local_viol)
        worst_viol = total_local_viol(worst_node)

        if worst_viol < 0.05:
            break  # all vertices satisfied

        old_x = G.nodes[worst_node]['x']
        old_y = G.nodes[worst_node]['y']
        best_viol = worst_viol
        best_x, best_y = old_x, old_y

        for _ in range(30):
            nx_ = np.clip(old_x + random.uniform(-step, step), -scale + 5, scale - 5)
            ny_ = np.clip(old_y + random.uniform(-step, step), -scale + 5, scale - 5)
            G.nodes[worst_node]['x'] = nx_
            G.nodes[worst_node]['y'] = ny_

            if any_incident_crosses(G, worst_node):
                continue  # planarity check

            v = total_local_viol(worst_node)
            if v < best_viol:
                best_viol = v
                best_x, best_y = nx_, ny_

        G.nodes[worst_node]['x'] = best_x
        G.nodes[worst_node]['y'] = best_y

    return recompute_features(G)


def repair_even_degree(G):
    """
    Repair odd-degree interior vertices by pairing them and adding a
    non-crossing edge. Called before kawasaki repair so the vertex
    degree structure is already correct when we solve for angles.
    """
    G = copy.deepcopy(G)

    def interior(n):
        return not any(G[n][nb].get('fold_type') == 1 for nb in G.neighbors(n))

    odd_nodes = [n for n in G.nodes() if interior(n) and G.degree(n) % 2 != 0]
    random.shuffle(odd_nodes)

    while len(odd_nodes) >= 2:
        u = odd_nodes.pop()
        for i, v in enumerate(odd_nodes):
            if not G.has_edge(u, v) and not edge_crosses_any(G, u, v):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))
                odd_nodes.pop(i)
                break

    return G

# ── Similarity ────────────────────────────────────────────────────────────────

def compute_similarity(G1, G2):
    def feature_vec(G):
        degrees    = sorted([d for _, d in G.degree()], reverse=True)
        n          = G.number_of_nodes()
        e          = G.number_of_edges()
        density    = e / max(1, n * (n - 1) / 2)
        deg_padded = (degrees + [0] * 20)[:20]
        return np.array([n, e, density] + deg_padded, dtype=float)

    v1   = feature_vec(G1)
    v2   = feature_vec(G2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)

# ── Penalties ─────────────────────────────────────────────────────────────────

def maekawa_penalty(G):
    penalty, count = 0.0, 0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        if any(G[node][nb].get('fold_type') == 1 for nb in neighbors):
            continue
        m    = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 2)
        v_   = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 3)
        diff = abs(m - v_)
        if diff != 2:
            penalty += abs(diff - 2)
        count += 1
    return penalty / max(1, count)


def novelty_penalty(G, real_gs, sample_k=50):
    sample  = random.sample(real_gs, min(sample_k, len(real_gs)))
    sims    = [compute_similarity(G, r) for r in sample]
    return max(0.0, np.mean(sims) - 0.75)


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

    x_tensor  = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = []
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
    gnn   = gnn_score(G)
    kaw   = kawasaki_penalty(G)   # for logging — should be ~0 always now
    mae   = maekawa_penalty(G)
    nov   = novelty_penalty(G, real_graphs)
    # Kawasaki weight low because it's enforced structurally, not via penalty.
    # Novelty weight high to push away from existing CPs.
    score = gnn - 0.05 * kaw - 0.2 * mae - 1.0 * nov
    return score, gnn, kaw, mae, nov

# ── Graph utilities ───────────────────────────────────────────────────────────

def recompute_features(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        nx_c = G.nodes[node]['x']
        ny_c = G.nodes[node]['y']
        angles = sorted(
            np.arctan2(G.nodes[nb]['y'] - ny_c, G.nodes[nb]['x'] - nx_c)
            for nb in neighbors
        )
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles
    return G

# ── Mutation (hard Kawasaki enforcement) ──────────────────────────────────────
#
# STRATEGY:
#   Each mutation is attempted, then validated. If any affected vertex now
#   violates Kawasaki (kaw_at > 0.05), the mutation is reverted entirely.
#   This means the feasible region is exactly {graphs satisfying Kawasaki}.
#
#   After repair_kawasaki() brings the initial population into the feasible
#   region, mutations can only produce feasible offspring — or they revert.
#
#   Affected vertices per mutation type:
#     move_node   → the moved node + all its neighbours (angles at both change)
#     add_edge    → the two new endpoints (their angle sets change)
#     remove_edge → the two former endpoints
#     flip_fold   → fold_type doesn't change geometry so kaw is unaffected;
#                   only Maekawa changes, which is a soft penalty

def mutate(G, scale=200.0):
    G        = copy.deepcopy(G)
    mutation = random.choice(['flip_fold', 'move_node', 'remove_edge', 'add_edge'])

    edges          = list(G.edges(data=True))
    nodes          = list(G.nodes())
    interior_edges = [(u, v, d) for u, v, d in edges if d.get('fold_type') != 1]
    interior_nodes = [
        n for n in nodes
        if not any(G[n][nb].get('fold_type') == 1 for nb in G.neighbors(n))
    ]

    if mutation == 'flip_fold' and interior_edges:
        # Kawasaki unaffected (geometry unchanged). Enforce Maekawa direction.
        for u, v, d in random.sample(interior_edges, min(5, len(interior_edges))):
            old_t = d['fold_type']
            new_t = 3 if old_t == 2 else 2

            def mv_delta(node, old_t, new_t):
                m  = sum(1 for nb in G.neighbors(node)
                         if G[node][nb].get('fold_type') == 2)
                vv = sum(1 for nb in G.neighbors(node)
                         if G[node][nb].get('fold_type') == 3)
                before = abs(m - vv)
                m2  = m  + (1 if new_t == 2 else 0) - (1 if old_t == 2 else 0)
                vv2 = vv + (1 if new_t == 3 else 0) - (1 if old_t == 3 else 0)
                return abs(m2 - vv2) - before

            if mv_delta(u, old_t, new_t) + mv_delta(v, old_t, new_t) <= 0:
                G[u][v]['fold_type'] = new_t
                break

    elif mutation == 'move_node' and interior_nodes:
        node  = random.choice(interior_nodes)
        old_x = G.nodes[node]['x']
        old_y = G.nodes[node]['y']
        new_x = np.clip(old_x + random.uniform(-15, 15), -scale + 5, scale - 5)
        new_y = np.clip(old_y + random.uniform(-15, 15), -scale + 5, scale - 5)

        G.nodes[node]['x'] = new_x
        G.nodes[node]['y'] = new_y

        # Affected: moved node + all its neighbours
        affected = [node] + list(G.neighbors(node))

        if any_incident_crosses(G, node) or not kaw_ok(G, affected):
            # Revert — this move breaks planarity or Kawasaki
            G.nodes[node]['x'] = old_x
            G.nodes[node]['y'] = old_y

    elif mutation == 'remove_edge' and len(interior_edges) > 5:
        # Prefer edges where both endpoints are odd-degree (preserves parity)
        odd_edges = [
            (u, v, d) for u, v, d in interior_edges
            if G.degree(u) % 2 != 0 and G.degree(v) % 2 != 0
        ]
        pool = odd_edges if odd_edges else interior_edges
        u, v, _ = random.choice(pool)

        if G.has_edge(u, v):
            G.remove_edge(u, v)
            # Affected: both former endpoints
            if not kaw_ok(G, [u, v]):
                G.add_edge(u, v, fold_type=random.choice([2, 3]))  # revert

    elif mutation == 'add_edge' and len(interior_nodes) > 2:
        # Prefer odd-degree pairs (adding edge restores even degree at both)
        odd_nodes = [n for n in interior_nodes if G.degree(n) % 2 != 0]
        pool      = odd_nodes if len(odd_nodes) >= 2 else interior_nodes

        for _ in range(20):
            u, v = random.sample(pool, 2)
            if G.has_edge(u, v) or edge_crosses_any(G, u, v):
                continue
            ft = random.choice([2, 3])
            G.add_edge(u, v, fold_type=ft)
            # Affected: both new endpoints
            if not kaw_ok(G, [u, v]):
                G.remove_edge(u, v)  # revert
            else:
                break  # valid edge found

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = recompute_features(G)
    return G

# ── Diversity helpers ─────────────────────────────────────────────────────────

def shared_fitness(population, raw_fitnesses, sigma=0.85):
    shared = []
    for i, (gi, fi) in enumerate(zip(population, raw_fitnesses)):
        sharing_sum = 0.0
        for j, gj in enumerate(population):
            if i == j:
                continue
            sim = compute_similarity(gi, gj)
            if sim > sigma:
                sharing_sum += (sim - sigma) / (1 - sigma)
        shared.append(fi / (1 + sharing_sum))
    return shared


def select_diverse_top(population, fitnesses, k=6, min_dissimilarity=0.15):
    ranked   = sorted(zip(fitnesses, population), reverse=True, key=lambda x: x[0])
    selected = []
    for fit, candidate in ranked:
        if len(selected) >= k:
            break
        if not any(compute_similarity(candidate, s) > (1 - min_dissimilarity)
                   for s in selected):
            selected.append(candidate)
    if len(selected) < k:
        for fit, candidate in ranked:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= k:
                break
    return selected

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

    with open(f"{BASE}\\negatives.pkl", 'rb') as f:
        negatives = pickle.load(f)

    # ── Build and repair initial population ───────────────────────────────────
    # Step 1: seed from corrupted CPs
    # Step 2: repair even-degree (structural prerequisite for Kawasaki)
    # Step 3: repair Kawasaki geometrically (move nodes until kaw≈0)
    # After this, all individuals are in the feasible region and mutations
    # enforce Kawasaki hard via revert-on-violation.
    print("Building initial population (repairing even-degree + Kawasaki)...")
    population = []
    for i in range(population_size):
        G = copy.deepcopy(random.choice(negatives))
        G = repair_even_degree(G)
        G = repair_kawasaki(G)
        population.append(G)
        if (i + 1) % 10 == 0:
            print(f"  Repaired {i + 1}/{population_size} individuals")

    # Sanity check — report mean kaw after repair
    mean_kaw_after = np.mean([kawasaki_penalty(G) for G in population])
    print(f"Initial population ready. Mean Kaw after repair: {mean_kaw_after:.4f}")

    best_scores     = []
    mean_scores     = []
    best_ever       = None
    best_ever_score = -999

    for gen in range(1, generations + 1):
        scored   = []
        raw_fits = []
        for G in population:
            f, gnn, kaw, mae, nov = fitness(G)
            scored.append((f, gnn, kaw, mae, nov, G))
            raw_fits.append(f)

        shared        = shared_fitness(population, raw_fits, sigma=0.85)
        shared_scored = sorted(zip(shared, scored), reverse=True, key=lambda x: x[0])
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score  = scored[0][0]
        top_gnn    = scored[0][1]
        top_kaw    = scored[0][2]
        top_mae    = scored[0][3]
        top_nov    = scored[0][4]
        mean_score = np.mean(raw_fits)

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

        new_population  = [copy.deepcopy(s[5]) for s in scored[:elite_keep]]
        top_half_shared = [s[1][5] for s in shared_scored[:population_size // 2]]

        while len(new_population) < population_size:
            parent = random.choice(top_half_shared)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child)
            new_population.append(child)

        population = new_population

    print(f"\nGA complete — best fitness: {best_ever_score:.4f}")

    # ── Convergence plot ──────────────────────────────────────────────────────
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

    # ── Diverse top 6 ─────────────────────────────────────────────────────────
    final_population = [s[5] for s in scored]
    final_fitnesses  = [s[0] for s in scored]
    diverse_top6     = select_diverse_top(
        final_population, final_fitnesses, k=6, min_dissimilarity=0.15
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, G in enumerate(diverse_top6):
        f, gnn, kaw, mae, nov = fitness(G)
        visualise(
            G,
            title=f"Rank {i+1} | fit={f:.3f} GNN={gnn:.3f} Nov={nov:.3f}",
            ax=axes.flatten()[i]
        )
    plt.suptitle("Top 6 Generated Crease Patterns (Diverse)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{BASE}\\ga_top6.png", dpi=150)
    plt.show()

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(f"{BASE}\\best_generated.pkl", 'wb') as f_out:
        pickle.dump(best_ever, f_out)
    with open(f"{BASE}\\diverse_top6.pkl", 'wb') as f_out:
        pickle.dump(diverse_top6, f_out)

    print("Best generated CP saved → best_generated.pkl")
    print("Diverse top 6 saved     → diverse_top6.pkl")

    return best_ever, scored, diverse_top6


best, results, diverse = run_ga(
    population_size=50,
    generations=80,
    elite_keep=10,
    mutations_per=3,
)