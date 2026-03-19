import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\graphs.pkl", 'rb') as f:
    positives = pickle.load(f)

with open(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\negatives.pkl", 'rb') as f:
    negatives = pickle.load(f)

def get_stats(graphs):
    stats = {
        'nodes':        [],
        'edges':        [],
        'avg_degree':   [],
        'n_mountain':   [],
        'n_valley':     [],
        'n_border':     [],
        'mv_ratio':     [],  # mountain/valley ratio — should be ~1 for valid CPs
        'avg_angles':   [],  # avg number of angles per node
    }
    for G in graphs:
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        
        mountain = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==2)
        valley   = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==3)
        border   = sum(1 for u,v,d in G.edges(data=True) if d.get('fold_type')==1)
        
        stats['nodes'].append(nodes)
        stats['edges'].append(edges)
        stats['avg_degree'].append(np.mean(degrees) if degrees else 0)
        stats['n_mountain'].append(mountain)
        stats['n_valley'].append(valley)
        stats['n_border'].append(border)
        stats['mv_ratio'].append(mountain / valley if valley > 0 else 0)
        
        all_angles = [len(G.nodes[n]['angles']) for n in G.nodes()]
        stats['avg_angles'].append(np.mean(all_angles) if all_angles else 0)
    
    return stats

pos_stats = get_stats(positives)
neg_stats = get_stats(negatives)

# --- print summary table ---
metrics = ['nodes', 'edges', 'avg_degree', 'n_mountain', 'n_valley', 'n_border', 'mv_ratio', 'avg_angles']
print(f"{'Metric':<20} {'Positives Mean':>16} {'Positives Std':>14} {'Negatives Mean':>16} {'Negatives Std':>14}")
print("-" * 82)
for m in metrics:
    pm = np.mean(pos_stats[m])
    ps = np.std(pos_stats[m])
    nm = np.mean(neg_stats[m])
    ns = np.std(neg_stats[m])
    print(f"{m:<20} {pm:>16.2f} {ps:>14.2f} {nm:>16.2f} {ns:>14.2f}")

# --- plot distributions ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, m in enumerate(metrics):
    ax = axes[i]
    ax.hist(pos_stats[m], bins=30, alpha=0.6, color='blue', label='Valid CPs')
    ax.hist(neg_stats[m], bins=30, alpha=0.6, color='red',  label='Corrupted')
    ax.set_title(m)
    ax.legend(fontsize=7)

plt.suptitle("Valid CPs vs Corrupted Negatives — Feature Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\data_comparison.png", dpi=150)
plt.show()
print("Plot saved")