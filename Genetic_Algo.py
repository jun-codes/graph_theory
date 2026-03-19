import pickle
import torch
import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import Data
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

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
model.load_state_dict(torch.load(
    r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\best_model.pt",
    weights_only=False))
model.eval()
print(f"GNN loaded on {device}")

with open(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\graphs.pkl", 'rb') as f:
    real_cps = pickle.load(f)
print(f"Loaded {len(real_cps)} real CPs for initial population")

def kawasaki_penalty(G):
    """Sum of Kawasaki violations across all interior vertices"""
    penalty = 0.0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        if any(G[node][nb].get('fold_type') == 1 for nb in neighbors):
            continue
        nx_coord = G.nodes[node]['x']
        ny_coord = G.nodes[node]['y']
        angles = []
        for nb in neighbors:
            dx = G.nodes[nb]['x'] - nx_coord
            dy = G.nodes[nb]['y'] - ny_coord
            angles.append(np.arctan2(dy, dx))
        angles.sort()
        even_sum = sum(angles[i] - angles[i-1] for i in range(0, len(angles), 2))
        odd_sum  = sum(angles[i] - angles[i-1] for i in range(1, len(angles), 2))
        violation = abs(abs(even_sum) - np.pi) + abs(abs(odd_sum) - np.pi)
        penalty  += violation
    return penalty / max(1, G.number_of_nodes())

def maekawa_penalty(G):
    """Sum of Maekawa violations across all interior vertices"""
    penalty = 0.0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        if any(G[node][nb].get('fold_type') == 1 for nb in neighbors):
            continue
        mountain = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 2)
        valley   = sum(1 for nb in neighbors if G[node][nb].get('fold_type') == 3)
        diff     = abs(mountain - valley)
        if diff != 2:
            penalty += abs(diff - 2)
    return penalty / max(1, G.number_of_nodes())

def gnn_score(G):
    """Returns probability that G is a valid CP (0.0 to 1.0)"""
    nodes = list(G.nodes())
    if len(nodes) < 2 or G.number_of_edges() < 2:
        return 0.0

    node_features = []
    for node in nodes:
        x          = G.nodes[node]['x']
        y          = G.nodes[node]['y']
        degree     = G.nodes[node].get('degree', G.degree(node))
        angles     = G.nodes[node].get('angles', [])
        angle_mean = float(np.mean(angles)) if angles else 0.0
        angle_std  = float(np.std(angles))  if angles else 0.0
        node_features.append([x, y, degree, angle_mean, angle_std])

    x_tensor   = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = []
    for u, v in G.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    if not edge_index:
        return 0.0

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    batch      = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        out  = model(x_tensor, edge_index, batch)
        prob = torch.softmax(out, dim=1)
    return prob[0][1].item()  

def fitness(G):
    """
    fitness = GNN_score - kawasaki_penalty - maekawa_penalty
    Higher = better = more like a valid CP
    """
    gnn    = gnn_score(G)
    kaw    = kawasaki_penalty(G)
    mae    = maekawa_penalty(G)
    score  = gnn - 0.3 * kaw - 0.3 * mae
    return score, gnn, kaw, mae

def recompute_features(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        nx_coord  = G.nodes[node]['x']
        ny_coord  = G.nodes[node]['y']
        angles    = []
        for nb in neighbors:
            dx = G.nodes[nb]['x'] - nx_coord
            dy = G.nodes[nb]['y'] - ny_coord
            angles.append(np.arctan2(dy, dx))
        angles.sort()
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles
    return G

def mutate(G, scale=200.0):
    G = copy.deepcopy(G)
    mutation = random.choice(['flip_fold', 'move_node', 'remove_edge', 'add_edge'])

    edges         = list(G.edges(data=True))
    nodes         = list(G.nodes())
    interior_edges = [(u,v,d) for u,v,d in edges if d.get('fold_type') != 1]
    interior_nodes = [n for n in nodes
                      if not any(G[n][nb].get('fold_type')==1 for nb in G.neighbors(n))]

    if mutation == 'flip_fold' and interior_edges:
        n_flip = max(1, len(interior_edges) // 10)
        for u, v, d in random.sample(interior_edges, min(n_flip, len(interior_edges))):
            G[u][v]['fold_type'] = 3 if d['fold_type'] == 2 else 2

    elif mutation == 'move_node' and interior_nodes:
        node  = random.choice(interior_nodes)
        new_x = G.nodes[node]['x'] + random.uniform(-15, 15)
        new_y = G.nodes[node]['y'] + random.uniform(-15, 15)
        G.nodes[node]['x'] = max(-scale+5, min(scale-5, new_x))
        G.nodes[node]['y'] = max(-scale+5, min(scale-5, new_y))

    elif mutation == 'remove_edge' and len(interior_edges) > 5:
        u, v, _ = random.choice(interior_edges)
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    elif mutation == 'add_edge' and len(interior_nodes) > 2:
        u, v = random.sample(interior_nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, fold_type=random.choice([2, 3]))

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    G = recompute_features(G)
    return G

def visualise(G, title="Generated CP", ax=None):
    pos      = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}
    mountain = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==2]
    valley   = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==3]
    border   = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))

    nx.draw_networkx_edges(G, pos, edgelist=mountain, edge_color='red',   width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=valley,   edge_color='blue',  width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=border,   edge_color='black', width=2.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=8,        node_color='black', ax=ax)
    ax.set_title(title, fontsize=9)
    ax.axis('equal')
    ax.axis('off')

def run_ga(
    population_size = 50,
    generations     = 20,
    elite_keep      = 10,   
    mutations_per   = 3,    
    seed_from_real  = 0.5,  
):
    print(f"\nStarting GA — pop={population_size}, generations={generations}")

    population = []
    for i in range(population_size):
        G = copy.deepcopy(random.choice(real_cps))
        for _ in range(2):
            G = mutate(G)
        population.append(G)
    print(f"Initial population: {len(population)} individuals")

    best_scores  = []
    mean_scores  = []
    best_ever    = None
    best_ever_score = -999

    for gen in range(1, generations + 1):
        scored = []
        for G in population:
            f, gnn, kaw, mae = fitness(G)
            scored.append((f, gnn, kaw, mae, G))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score  = scored[0][0]
        top_gnn    = scored[0][1]
        top_kaw    = scored[0][2]
        top_mae    = scored[0][3]
        mean_score = np.mean([s[0] for s in scored])

        best_scores.append(top_score)
        mean_scores.append(mean_score)

        if top_score > best_ever_score:
            best_ever_score = top_score
            best_ever       = copy.deepcopy(scored[0][4])

        if gen % 5 == 0 or gen == 1:
            print(f"Gen {gen:03d} | Best: {top_score:.4f} "
                  f"(GNN={top_gnn:.3f} Kaw={top_kaw:.3f} Mae={top_mae:.3f}) "
                  f"| Mean: {mean_score:.4f}")

        new_population = [copy.deepcopy(s[4]) for s in scored[:elite_keep]]

        top_half = [s[4] for s in scored[:population_size // 2]]
        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child  = copy.deepcopy(parent)
            for _ in range(mutations_per):
                child = mutate(child)
            new_population.append(child)

        population = new_population

    print(f"\nGA complete — best fitness: {best_ever_score:.4f}")


    plt.figure(figsize=(8,4))
    plt.plot(best_scores, label='Best fitness',  color='blue')
    plt.plot(mean_scores, label='Mean fitness',  color='orange', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\ga_convergence.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (f, gnn, kaw, mae, G) in enumerate(scored[:6]):
        visualise(G, title=f"Rank {i+1} | fit={f:.3f} GNN={gnn:.3f}", ax=axes[i])
    plt.suptitle("Top 6 Generated Crease Patterns", fontsize=13)
    plt.tight_layout()
    plt.savefig(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\ga_top6.png", dpi=150)
    plt.show()

    with open(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\best_generated.pkl", 'wb') as f_out:
        pickle.dump(best_ever, f_out)
    print("Best generated CP saved to best_generated.pkl")

    return best_ever, scored

best, results = run_ga(
    population_size = 50,
    generations     = 20,
    elite_keep      = 10,
    mutations_per   = 3,
)