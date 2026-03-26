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

Generation:
- Genetic_Algo.py: Evolves crease patterns using the trained GNN as a fitness function. Computes Kawasaki and Maekawa penalties (folding constraints), mutates graphs, and selects high-scoring candidates.
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

5. Compare patterns:
   python similarity.py
   (Computes similarity between generated and real CPs)

6. Visualize and analyze:
   python comparison.py
   (Histogram distributions of graph statistics)
   python visualiser.py
   (Plot an example from the dataset)



