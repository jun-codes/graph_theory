import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random

with open(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\negatives.pkl", 'rb') as f:
    graphs = pickle.load(f)

G = random.choice(graphs)

print(f"Showing: {G.graph['filename']}")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}

mountain = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==2]
valley   = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==3]
border   = [(u,v) for u,v,d in G.edges(data=True) if d.get('fold_type')==1]

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, edgelist=mountain, edge_color='red',   width=1.5)
nx.draw_networkx_edges(G, pos, edgelist=valley,   edge_color='blue',  width=1.0)
nx.draw_networkx_edges(G, pos, edgelist=border,   edge_color='black', width=2.0)
nx.draw_networkx_nodes(G, pos, node_size=10,      node_color='black')

legend = [
    mpatches.Patch(color='red',   label='Mountain fold'),
    mpatches.Patch(color='blue',  label='Valley fold'),
    mpatches.Patch(color='black', label='Border'),
]
plt.legend(handles=legend, loc='upper right')
plt.title(G.graph['filename'].replace('.cp', ''))
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()