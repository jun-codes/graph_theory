import pickle
from pathlib import Path

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

with (DATA_DIR / "graphs.pkl").open('rb') as f:
    positives = pickle.load(f)

with (DATA_DIR / "negatives.pkl").open('rb') as f:
    negatives = pickle.load(f)

print(f"Positives: {len(positives)}, Negatives: {len(negatives)}")

def nx_to_pyg(G):
    node_features = []
    for node in G.nodes():
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        degree = G.nodes[node]['degree']
        angles = G.nodes[node]['angles']
        angle_mean = float(np.mean(angles)) if angles else 0.0
        angle_std  = float(np.std(angles))  if angles else 0.0
        node_features.append([x, y, degree, angle_mean, angle_std])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # edge index
    edge_index = []
    edge_attr  = []
    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # undirected — add both directions
        edge_attr.append([data.get('fold_type', 0)])
        edge_attr.append([data.get('fold_type', 0)])
    
    if len(edge_index) == 0:
        return None  # skip empty graphs
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    label      = torch.tensor([G.graph['label']], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)

all_graphs = positives + negatives
dataset = []

skipped = 0
for G in all_graphs:
    data = nx_to_pyg(G)
    if data is None:
        skipped += 1
        continue
    dataset.append(data)

print(f"Total dataset size: {len(dataset)} (skipped {skipped} empty graphs)")

import random
random.shuffle(dataset)

n = len(dataset)
train_end = int(0.8 * n)
val_end   = int(0.9 * n)

train_set = dataset[:train_end]
val_set   = dataset[train_end:val_end]
test_set  = dataset[val_end:]

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
torch.save(train_set, DATA_DIR / "train.pt")
torch.save(val_set,   DATA_DIR / "val.pt")
torch.save(test_set,  DATA_DIR / "test.pt")

print("Saved train/val/test splits")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

print("Dataloaders ready")
