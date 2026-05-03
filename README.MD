# Origami Crease Pattern Generation

This repository contains a graph-theoretic origami crease-pattern generation project. It combines graph neural networks, a crease-level line-graph classifier, genetic algorithms, topology repair, Z3 Maekawa repair, Kawasaki projection repair, gradient geometry repair, and symmetry-constrained generation.

The accompanying project report is stored at `reports/Graph_Theory.pdf`.

## Dashboard

Run the Streamlit dashboard from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

The dashboard shows the current generated artifacts from `outputs/z3_symmetry_kawasaki/`, including:

- top-six generated crease-pattern image
- convergence plot
- local constraint violation plot
- per-rank graph diagnostics
- individual graph previews
- downloadable `.cp` crease-pattern files

The public deployment link used for the project is:

```text
https://origamigenerator.streamlit.app/
```

## Repository Layout

```text
.
├── data/                         # Graph datasets and train/val/test tensors
├── examples/                     # Small example .cp files
├── models/                       # Trained model checkpoints
├── outputs/                      # Generated graphs, images, and .cp exports
│   ├── baseline/
│   ├── line_filtered/
│   ├── line_repaired/
│   ├── line_z3/
│   ├── line_topology_z3/
│   ├── line_kawasaki_z3/
│   ├── sym_topology/
│   ├── z3_symmetry_kawasaki/
│   └── z3_symmetry_kawasaki_1/
├── reports/                      # Final report PDF
├── runs/                         # Longer historical experiment runs
├── streamlit_app.py              # Dashboard entrypoint
├── Genetic_Algo_Z3_Topology_Symmetry_Kawasaki.py
├── GNN_Classifier.py
├── LineGraph_Classifier.py
└── requirements.txt
```

Python scripts are kept at the root so the existing import style and Streamlit Cloud entrypoint remain simple. Large data, model, report, and output artifacts are organized into folders.

## Core Files

- `streamlit_app.py`: Streamlit dashboard for displaying artifacts and running the final generator.
- `Genetic_Algo_Z3_Topology_Symmetry_Kawasaki.py`: final generation pipeline.
- `GNN_Classifier.py`: vertex-based GIN classifier training.
- `LineGraph_Classifier.py`: crease-as-node line-graph classifier training.
- `line_graph_scoring.py`: reusable line-GNN inference helper.
- `origami_constraints.py`: local Kawasaki, Maekawa, crossing, and visualization utilities.
- `topology_even_repair.py`: even-degree topology repair for Maekawa compatibility.
- `maekawa_z3_repair.py`: Z3 mountain-valley assignment repair.
- `kawasaki_projection_repair.py`: direct Kawasaki sector-angle projection repair.
- `gradient_geometry_repair.py`: gradient coordinate repair for Kawasaki and symmetry.
- `cp_io.py`: `.cp` export helpers.

## Data and Artifact Format

Crease patterns are represented as NetworkX graphs. Nodes store embedded coordinates and local angular features. Edges store fold labels:

```text
1 = border
2 = mountain
3 = valley
```

Editable `.cp` files use one crease segment per row:

```text
fold_type x1 y1 x2 y2
```

The current dashboard CP dataset is in:

```text
outputs/z3_symmetry_kawasaki/z3_symmetry_kawasaki_diverse_top6_cp/
```

The best current generated candidate is:

```text
outputs/z3_symmetry_kawasaki/z3_symmetry_kawasaki_best_generated.cp
```

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Main dependencies include Streamlit, PyTorch, PyTorch Geometric, NetworkX, Shapely, Z3, NumPy, Pandas, and Matplotlib.

## Common Commands

Run the dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

Run the final generator directly:

```powershell
python Genetic_Algo_Z3_Topology_Symmetry_Kawasaki.py
```

Evaluate line-graph classifier outputs:

```powershell
python LineGraph_Tester.py
```

Export a pickle graph or graph list to `.cp` files:

```powershell
python export_generated_cp.py --input outputs/z3_symmetry_kawasaki/z3_symmetry_kawasaki_diverse_top6.pkl
```

Test a single `.cp` file with the vertex-GNN:

```powershell
python tester.py outputs/z3_symmetry_kawasaki/z3_symmetry_kawasaki_best_generated.cp
```

## Final Pipeline Summary

The final generator starts from random planar graph seeds inside a square paper boundary. It evolves candidates using mutation, elitism, diversity selection, and learned validity scoring. Each candidate is repaired and filtered through:

1. topology repair for even non-border interior degree,
2. Z3 Maekawa mountain-valley assignment,
3. direct Kawasaki projection,
4. gradient geometry repair,
5. crossing checks and hard-gated fitness evaluation.

The resulting candidates are exported as graph pickles, rendered PNGs, and editable `.cp` files.
