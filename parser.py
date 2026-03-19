import os
import networkx as nx
import numpy as np
from collections import defaultdict

def parse_cp_file(filepath):
    edges = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                fold_type = int(parts[0])
                x1, y1 = float(parts[1]), float(parts[2])
                x2, y2 = float(parts[3]), float(parts[4])
                edges.append((fold_type, x1, y1, x2, y2))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return edges

def build_graph(edges, tolerance=1e-6):
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
            if abs(pt[0] - upt[0]) < tolerance and abs(pt[1] - upt[1]) < tolerance:
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
        nx_coord = G.nodes[node]['x']
        ny_coord = G.nodes[node]['y']
        angles = []
        for nb in neighbors:
            dx = G.nodes[nb]['x'] - nx_coord
            dy = G.nodes[nb]['y'] - ny_coord
            angles.append(np.arctan2(dy, dx))
        angles.sort()
        G.nodes[node]['degree'] = len(neighbors)
        G.nodes[node]['angles'] = angles
    
    return G

def parse_all_cp_files(cp_folder):
    graphs = []
    files = [f for f in os.listdir(cp_folder) if f.endswith('.cp')]
    print(f"Found {len(files)} CP files")
    
    for filename in files:
        filepath = os.path.join(cp_folder, filename)
        edges = parse_cp_file(filepath)
        if not edges:
            print(f"Skipped (empty): {filename}")
            continue
        G = build_graph(edges)
        G.graph['filename'] = filename
        G.graph['num_edges'] = len(edges)
        graphs.append(G)
    
    print(f"Successfully parsed: {len(graphs)} graphs")
    return graphs

cp_folder = r"C:\Users\Arjun\Downloads\dataset\cp_files"  
graphs = parse_all_cp_files(cp_folder)

G = graphs[0]
print(f"\nSample graph: {G.graph['filename']}")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Sample node features: {dict(G.nodes[0])}")

import pickle

for G in graphs:
    G.graph['label'] = 1

output_path = r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\graphs.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Saved {len(graphs)} labeled graphs")