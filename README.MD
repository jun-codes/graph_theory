Graph Theory Project: Crease Pattern Analysis and Generation

This project uses machine learning and genetic algorithms to classify, analyze, and generate crease patterns (CPs) for origami. It combines graph neural networks with GA's to explore the space of valid fold patterns.

Overview

The project has three main components:

1. Data Processing: Load and parse crease pattern files, extract graph features, and split datasets for training.
2. Classification: Train a Graph Isomorphism Network (GIN) to distinguish valid crease patterns from corrupted or invalid ones.
3. Generation and Optimization: Use genetic algorithms to generate novel crease patterns and evaluate them against real-world designs.

Project Structure

Core Training and Evaluation:
- GNN_Classifier.py: Trains a Graph Isomorphism Network on labeled graph data. Uses 3 GIN layers with MLPs, global pooling, and a classifier head. Saves the best model to best_model.pt during training.
- data_split.py: Loads positive examples (valid CPs) and negative examples (corrupted patterns), converts them to PyTorch Geometric format, and creates train/validation/test splits (80/10/10). Outputs to train.pt, val.pt, test.pt.
- tester.py: Evaluates the trained model on new crease pattern files (both .cp text format and .png images). Extracts edges, builds node-edge graphs, and predicts validity.
- LineGraph_Classifier.py: Trains a separate crease-line graph classifier. Each crease becomes a node in a line graph, and line-graph edges describe creases that meet at the same CP vertex.
- line_graph_features.py: Converts NetworkX CP graphs into line-graph features with crease-level node features and crease-adjacency edge features.
- line_graph_scoring.py: Loads line_best_model.pt and provides reusable line-GNN scoring utilities.
- LineGraph_Tester.py: Evaluates .pkl graph artifacts or .cp files with the line-GNN, plus Kawasaki and Maekawa summaries.
- line_graph_ga_filter.py: Optional helper for penalizing GA candidates that the line-GNN classifies as invalid. It is not wired into Genetic_Algo.py by default.

Generation:
- Genetic_Algo.py: Evolves crease patterns using the trained GNN as a fitness function. Computes Kawasaki and Maekawa penalties (folding constraints), mutates graphs, and selects high-scoring candidates.
- Genetic_Algo_Line_Filtered.py: Separate experimental copy of the GA that adds a conservative line-GNN validity penalty. It writes line_filtered_best_generated.pkl, line_filtered_diverse_top6.pkl, line_ga_convergence.png, and line_ga_top6.png without overwriting the original GA outputs.
- random_graph_generator.py: Generates synthetic random graphs using Delaunay triangulation. Assigns fold types (mountain, valley, border) and node features (angle sequences, degree).

Visualization:
- similarity.py: Compares generated patterns to real dataset examples using graph signatures (nodes, edges, fold types, degree statistics).
- comparison.py: Computes and plots statistical distributions (nodes, edges, degree, fold counts, fold ratios) comparing valid vs. invalid patterns.
- visualiser.py: Renders graphs with color-coded edge types and node positions extracted from crease pattern data.
- file_extractor.py: Parses .cp files and .png images to extract fold information and build NetworkX graphs.


Features

Graph Representation: Each crease pattern is stored as a NetworkX graph with:
  - Node attributes: x, y coordinates, degree, angle sequence
  - Edge attributes: fold_type (1=border, 2=mountain, 3=valley)
  - Graph label: 1 for valid CPs, 0 for corrupted/invalid

Neural Network Architecture: A 3-layer GIN with 64 hidden units, batch normalization, MLPs on edges, and global mean/max pooling. Output is a 2-class classifier (valid or invalid).

Line-Graph Classifier: A separate GINE classifier treats creases as graph nodes. It uses 25 crease-node features and 10 crease-adjacency edge features, including fold type, line geometry, shared-vertex angle, cyclic adjacency, endpoint degree parity, and local Kawasaki/Maekawa residuals. The current checkpoint is line_best_model.pt.

Genetic Algorithm: Maintains a population of crease patterns, evaluates fitness using the GNN, applies mutations, and selects candidates with lowest Kawasaki/Maekawa penalties.

Input/Output Formats:
  - Input: .cp files (fold_type x1 y1 x2 y2 per line) and .png images (edge detection via Canny)
  - Output: pickle files with NetworkX graphs, PyTorch model checkpoints (.pt)


Workflow:

1. Prepare data:
   python data_split.py

2. Train the GNN classifier:
   python GNN_Classifier.py
   (Runs for 60 epochs, saves best model)

3. Generate new patterns using genetics:
   python Genetic_Algo.py
   (Loads best_model.pt, evolves population)

4. Test a specific file:
   python tester.py <path_to_cp_or_png_file>
   (Predicts if the pattern is a valid crease pattern)

5. Train the line-graph classifier:
   python LineGraph_Classifier.py
   (Saves line_best_model.pt)

6. Evaluate line-graph classifier artifacts:
   python LineGraph_Tester.py
   (Evaluates graphs.pkl, negatives.pkl, negatives_v3.pkl, best_generated.pkl, and diverse_top6.pkl)

7. Compare patterns:
   python similarity.py
   (Computes similarity between generated and real CPs)

8. Visualize and analyze:
   python comparison.py
   (Histogram distributions of graph statistics)
   python visualiser.py
   (Plot an example from the dataset)

Recent Line-Graph Results:

- Full line-GNN test accuracy after 10 epochs: 0.9762
- Held-out confusion matrix, rows=true [invalid, valid], cols=pred [invalid, valid]: [[66, 0], [3, 57]]
- Full artifact evaluation:
  - graphs.pkl: 607/626 predicted valid
  - negatives.pkl: 625/626 predicted invalid
  - negatives_v3.pkl: 612/626 predicted invalid
  - best_generated.pkl: predicted invalid, valid probability about 0.0001
  - diverse_top6.pkl: all six predicted invalid

Current Direction:

The line-GNN is useful as a conservative filter for generated crease patterns. A separate line-filtered GA smoke run was added without changing the original GA.

Line-Filtered GA Smoke Run:

- Script: Genetic_Algo_Line_Filtered.py
- Run settings: population_size=30, generations=20, elite_keep=6, mutations_per=3
- Best generated output: line_filtered_best_generated.pkl
  - line-GNN valid probability: 0.0081
  - predicted invalid
  - mean Kawasaki residual: 1.7315
  - mean Maekawa residual: 1.3333
- Diverse top 6 output: line_filtered_diverse_top6.pkl
  - all six predicted invalid
  - mean Kawasaki residual: 0.7487
  - mean Maekawa residual: 1.5297

The filter improved the line-GNN score slightly for the single best candidate and reduced Kawasaki residuals in the diverse set, but it did not produce valid line-GNN candidates in 20 generations. This suggests the next bottleneck is the GA repair/mutation operators, especially local Kawasaki/Maekawa repair, not just classifier scoring.

Gradient Geometry Repair Experiment:

- Script: Genetic_Algo_Line_Filtered_Repaired.py
- Repair utility: gradient_geometry_repair.py
- Method: PyTorch gradient descent on mutable node coordinates, with border nodes fixed, targeting Kawasaki residual, mirror symmetry, and short-edge degeneracy. Repaired candidates are rejected if they introduce crease crossings.
- Run settings: population_size=30, generations=20, elite_keep=6, mutations_per=3
- Initial seed repairs accepted: 22/30
- Best repaired output: line_repaired_best_generated.pkl
  - line-GNN valid probability: 0.0005
  - predicted invalid
  - mean Kawasaki residual: 0.9922
  - mean Maekawa residual: 2.0000
- Repaired diverse top 6: line_repaired_diverse_top6.pkl
  - all six predicted invalid
  - mean Kawasaki residual: 0.9327
  - mean Maekawa residual: 1.8972

The gradient repair successfully lowers Kawasaki for selected candidates and can improve symmetry, but it cannot fix Maekawa because Maekawa depends on discrete mountain/valley counts, not coordinates. The next repair step should be a discrete fold-assignment repair for Maekawa, paired with the gradient Kawasaki/symmetry repair.

Z3 Maekawa Repair Experiment:

- Repair utility: maekawa_z3_repair.py
- Script: Genetic_Algo_Line_Filtered_Repaired_Z3.py
- Method: one Z3 Boolean per non-border crease, with True=mountain and False=valley. Each interior vertex enforces Maekawa by requiring mountain count d/2-1 or d/2+1. The optimizer minimizes changed fold labels.
- Existing repaired-output smoke check:
  - line_repaired_best_generated.pkl: SAT, changed 6 creases, Maekawa 2.0000 -> 0.0000
  - line_repaired_diverse_top6.pkl: 3/6 SAT; topology-unsat cases had odd non-border vertex degrees
- GA smoke settings: population_size=30, generations=20, elite_keep=6, mutations_per=3
- GA result:
  - Initial Z3 repairs: 30/30 topology_unsat
  - Every generation stayed topology_unsat, so Z3 could not perform exact Maekawa repair by labels alone
  - line_z3_best_generated.pkl: predicted invalid, valid probability 0.0000, mean Kawasaki 0.9821, mean Maekawa 1.1429
  - line_z3_diverse_top6.pkl: all six predicted invalid, mean Kawasaki 0.7881, mean Maekawa 1.2549

Conclusion: Z3 works when topology is Maekawa-satisfiable, but the current generator/mutations produce too many odd non-border vertex degrees. The next required change is topology repair or topology-aware mutation: enforce even non-border degree at interior vertices before Z3 label solving.


