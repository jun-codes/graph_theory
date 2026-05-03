import pickle
from pathlib import Path

import numpy as np
import networkx as nx

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs" / "z3_symmetry_kawasaki"

with (DATA_DIR / "graphs.pkl").open('rb') as f:
    real_cps = pickle.load(f)

with (OUTPUT_DIR / "z3_symmetry_kawasaki_best_generated.pkl").open('rb') as f:
    generated = pickle.load(f)

def graph_signature(G):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    degrees = sorted([d for _, d in G.degree()])
    mountain = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==2)
    valley   = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==3)
    border   = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==1)
    avg_degree = np.mean(degrees) if degrees else 0
    degree_std = np.std(degrees)  if degrees else 0
    return np.array([nodes, edges, mountain, valley, border, avg_degree, degree_std])

def similarity_score(sig1, sig2):
    denom = np.maximum(np.abs(sig1) + np.abs(sig2), 1e-6)
    return 1.0 - np.mean(np.abs(sig1 - sig2) / denom)

gen_sig = graph_signature(generated)
print(f"Generated CP stats:")
print(f"  Nodes: {generated.number_of_nodes()}")
print(f"  Edges: {generated.number_of_edges()}")
print(f"  Filename: {generated.graph.get('filename', 'unknown')}")

similarities = []
for G in real_cps:
    sig  = graph_signature(G)
    sim  = similarity_score(gen_sig, sig)
    similarities.append((sim, G.graph.get('filename', '?')))

similarities.sort(reverse=True)

print(f"\nTop 10 most similar CPs in dataset:")
for i, (sim, fname) in enumerate(similarities[:10]):
    print(f"  {i+1}. {fname:<60} similarity: {sim:.4f}")

print(f"\nMost similar: {similarities[0][1]} at {similarities[0][0]:.4f}")
if similarities[0][0] > 0.98:
    print("  ⚠️  Very high similarity — might be same or near-identical CP")
elif similarities[0][0] > 0.90:
    print("  ⚠️  High similarity — structurally close but likely different")
else:
    print("  ✅ Novel — sufficiently different from all dataset CPs")
