import torch
import numpy as np
import networkx as nx
import cv2
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
print("Model loaded")

def test_cp_file(filepath):
    edges = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            fold_type = int(parts[0])
            x1, y1 = float(parts[1]), float(parts[2])
            x2, y2 = float(parts[3]), float(parts[4])
            edges.append((fold_type, x1, y1, x2, y2))
    return edges

def test_png_file(filepath):
    img  = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (800, 800))

    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_img = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges_img, rho=1, theta=np.pi/180,
                             threshold=80,       
                             minLineLength=40,   
                             maxLineGap=10)
    edges = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            edges.append((2, float(x1), float(y1), float(x2), float(y2)))
    print(f"  Detected {len(edges)} lines from image")
    return edges


def build_graph_from_edges(edges, tolerance=8.0):  
    G = nx.Graph()
    raw_points = []
    for _, x1, y1, x2, y2 in edges:
        raw_points.append((x1, y1))
        raw_points.append((x2, y2))

    unique_points = []
    point_map = {}
    for pt in raw_points:
        matched = False
        for i, upt in enumerate(unique_points):
            if abs(pt[0]-upt[0]) < tolerance and abs(pt[1]-upt[1]) < tolerance:
                point_map[pt] = i
                matched = True
                break
        if not matched:
            point_map[pt] = len(unique_points)
            unique_points.append(pt)

    for i, (x, y) in enumerate(unique_points):
        G.add_node(i, x=x, y=y)

    for fold_type, x1, y1, x2, y2 in edges:
        u = point_map[(x1, y1)]
        v = point_map[(x2, y2)]
        if u != v:
            G.add_edge(u, v, fold_type=fold_type)

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        nx_coord  = G.nodes[node]['x']
        ny_coord  = G.nodes[node]['y']
        angles = []
        for nb in neighbors:
            dx = G.nodes[nb]['x'] - nx_coord
            dy = G.nodes[nb]['y'] - ny_coord
            angles.append(np.arctan2(dy, dx))
        angles.sort()
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles

    return G

def predict(G):
    node_features = []
    for node in G.nodes():
        x       = G.nodes[node]['x']
        y       = G.nodes[node]['y']
        degree  = G.nodes[node]['degree']
        angles  = G.nodes[node]['angles']
        angle_mean = float(np.mean(angles)) if angles else 0.0
        angle_std  = float(np.std(angles))  if angles else 0.0
        node_features.append([x, y, degree, angle_mean, angle_std])

    x_tensor   = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    for u, v in G.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    if not edge_index:
        print("  Graph has no edges — cannot predict")
        return

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch      = torch.zeros(x_tensor.size(0), dtype=torch.long)

    x_tensor   = x_tensor.to(device)
    edge_index = edge_index.to(device)
    batch      = batch.to(device)

    with torch.no_grad():
        out  = model(x_tensor, edge_index, batch)
        prob = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1).item()

    label = " VALID crease pattern" if pred == 1 else " NOT a crease pattern"
    print(f"\n  Prediction : {label}")
    print(f"  Confidence : {prob[0][pred].item()*100:.1f}%")
    print(f"  (valid={prob[0][1].item()*100:.1f}%, invalid={prob[0][0].item()*100:.1f}%)")


test_file = r"C:\Users\Arjun\Desktop\test.cp"   

print(f"\nTesting: {test_file}")
if test_file.endswith('.cp'):
    edges = test_cp_file(test_file)
else:
    edges = test_png_file(test_file)

print(f"  Edges found: {len(edges)}")
G = build_graph_from_edges(edges)
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
predict(G)