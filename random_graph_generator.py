

import pickle
from pathlib import Path

import torch
import numpy as np
import networkx as nx
import random
from scipy.spatial import Delaunay
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

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_model.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GINClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model.eval()
print("Model loaded")

def generate_random_graph(scale=200.0):
    num_points = random.randint(50, 150)
    points     = np.random.uniform(-scale+10, scale-10, (num_points, 2))
    tri        = Delaunay(points)

    G = nx.Graph()
    for i, (x, y) in enumerate(points):
        G.add_node(i, x=x, y=y)

    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i+1) % 3]
            if not G.has_edge(u, v) and random.random() > 0.3:
                G.add_edge(u, v, fold_type=random.choice([1,2,3]))

    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)

    for node in G.nodes():
        neighbors  = list(G.neighbors(node))
        nx_coord   = G.nodes[node]['x']
        ny_coord   = G.nodes[node]['y']
        angles     = []
        for nb in neighbors:
            dx = G.nodes[nb]['x'] - nx_coord
            dy = G.nodes[nb]['y'] - ny_coord
            angles.append(np.arctan2(dy, dx))
        angles.sort()
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles

    G.graph['label'] = 0
    return G

def nx_to_pyg(G):
    node_features = []
    for node in G.nodes():
        x          = G.nodes[node]['x']
        y          = G.nodes[node]['y']
        degree     = G.nodes[node]['degree']
        angles     = G.nodes[node]['angles']
        angle_mean = float(np.mean(angles)) if angles else 0.0
        angle_std  = float(np.std(angles))  if angles else 0.0
        node_features.append([x, y, degree, angle_mean, angle_std])

    x_tensor   = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    for u, v in G.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch      = torch.zeros(x_tensor.size(0), dtype=torch.long)
    return x_tensor, edge_index, batch

print("\nTesting on 200 fresh random planar graphs (never seen)...")
correct = 0
total   = 200

for i in range(total):
    G      = generate_random_graph()
    result = nx_to_pyg(G)
    if result is None:
        total -= 1
        continue

    x_tensor, edge_index, batch = result
    x_tensor   = x_tensor.to(device)
    edge_index  = edge_index.to(device)
    batch       = batch.to(device)

    with torch.no_grad():
        out  = model(x_tensor, edge_index, batch)
        pred = out.argmax(dim=1).item()

    if pred == 0:  
        correct += 1

print(f"\nResults on random planar graphs:")
print(f"  Correctly identified as INVALID: {correct}/{total} ({100*correct/total:.1f}%)")
print(f"  Incorrectly called VALID:        {total-correct}/{total} ({100*(total-correct)/total:.1f}%)")
