"""
Evaluate the line-graph crease-pattern classifier on saved graphs or .cp files.

Default usage evaluates the current repo artifacts:
  python LineGraph_Tester.py

Specific files:
  python LineGraph_Tester.py best_generated.pkl diverse_top6.pkl
  python LineGraph_Tester.py path/to/pattern.cp
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Optional

import networkx as nx
import numpy as np

from line_graph_scoring import (
    DEFAULT_CHECKPOINT,
    LineGraphScorer,
    summarize_scores,
)


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUTPUTS_DIR = BASE / "outputs"


DEFAULT_TARGETS = [
    (DATA_DIR / "graphs.pkl", 1),
    (DATA_DIR / "negatives.pkl", 0),
    (DATA_DIR / "negatives_v3.pkl", 0),
    (OUTPUTS_DIR / "baseline" / "best_generated.pkl", None),
    (OUTPUTS_DIR / "baseline" / "diverse_top6.pkl", None),
    (OUTPUTS_DIR / "z3_symmetry_kawasaki" / "z3_symmetry_kawasaki_best_generated.pkl", None),
    (OUTPUTS_DIR / "z3_symmetry_kawasaki" / "z3_symmetry_kawasaki_diverse_top6.pkl", None),
]


def parse_cp_file(path: Path):
    edges = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            fold_type = int(parts[0])
            x1, y1 = float(parts[1]), float(parts[2])
            x2, y2 = float(parts[3]), float(parts[4])
            edges.append((fold_type, x1, y1, x2, y2))
    return edges


def build_graph_from_edges(edges, tolerance: float = 1e-6):
    G = nx.Graph()
    unique_points = []
    point_map = {}

    for _, x1, y1, x2, y2 in edges:
        for point in [(x1, y1), (x2, y2)]:
            matched = None
            for i, existing in enumerate(unique_points):
                if (
                    abs(point[0] - existing[0]) < tolerance
                    and abs(point[1] - existing[1]) < tolerance
                ):
                    matched = i
                    break
            if matched is None:
                matched = len(unique_points)
                unique_points.append(point)
            point_map[point] = matched

    for i, (x, y) in enumerate(unique_points):
        G.add_node(i, x=float(x), y=float(y))

    for fold_type, x1, y1, x2, y2 in edges:
        u = point_map[(x1, y1)]
        v = point_map[(x2, y2)]
        if u != v:
            G.add_edge(u, v, fold_type=fold_type)

    return G


def load_graphs(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".cp":
        return [build_graph_from_edges(parse_cp_file(path))]
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, nx.Graph):
            return [obj]
        if isinstance(obj, list) and all(isinstance(item, nx.Graph) for item in obj):
            return obj
        raise ValueError(f"{path} does not contain a NetworkX graph or graph list")
    raise ValueError(f"Unsupported input type: {path.suffix}")


def infer_expected_label(path: Path, explicit: Optional[int]):
    if explicit is not None:
        return explicit
    name = path.name.lower()
    if name == "graphs.pkl":
        return 1
    if name in {"negatives.pkl", "negatives_v3.pkl"}:
        return 0
    return None


def format_label(label: Optional[int]):
    if label == 1:
        return "valid"
    if label == 0:
        return "invalid"
    return "unknown"


def print_summary(path: Path, summary: dict, expected: Optional[int]):
    acc = "n/a" if summary["accuracy"] is None else f"{summary['accuracy']:.4f}"
    print(f"\n{path.name}  expected={format_label(expected)}")
    print(
        f"  count={summary['count']} acc={acc} "
        f"pred_valid={summary['pred_valid']} pred_invalid={summary['pred_invalid']}"
    )
    print(
        "  valid_prob "
        f"mean={summary['valid_prob_mean']:.4f} "
        f"median={summary['valid_prob_median']:.4f} "
        f"range=[{summary['valid_prob_min']:.4f}, {summary['valid_prob_max']:.4f}]"
    )
    print(
        f"  constraints mean_kaw={summary['kaw_mean']:.4f} "
        f"mean_mae={summary['mae_mean']:.4f}"
    )


def print_details(scores, top_n: int):
    if not scores or top_n <= 0:
        return
    ordered = sorted(scores, key=lambda s: s.valid_prob, reverse=True)
    print(f"  top {min(top_n, len(ordered))} by valid probability:")
    for score in ordered[:top_n]:
        label = "valid" if score.prediction == 1 else "invalid"
        print(
            f"    #{score.index:<3} pred={label:<7} valid={score.valid_prob:.4f} "
            f"nodes={score.nodes:<4} creases={score.creases:<4} "
            f"kaw={score.kaw_mean:.4f} mae={score.mae_mean:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the line-graph classifier.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected", type=int, choices=[0, 1], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--details", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.paths:
        targets = [(path, args.expected) for path in args.paths]
    else:
        targets = [
            (BASE / filename, expected)
            for filename, expected in DEFAULT_TARGETS
            if (BASE / filename).exists()
        ]

    scorer = LineGraphScorer(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    for path, default_expected in targets:
        path = path if path.is_absolute() else BASE / path
        graphs = load_graphs(path)
        if args.limit is not None:
            graphs = graphs[: args.limit]

        expected = infer_expected_label(path, args.expected)
        if args.expected is None:
            expected = default_expected if default_expected is not None else expected

        scores = scorer.score_graphs(graphs, source=path.name, batch_size=args.batch_size)
        summary = summarize_scores(scores, expected_label=expected)
        print_summary(path, summary, expected)

        if expected is None or len(scores) <= args.details:
            print_details(scores, args.details)


if __name__ == "__main__":
    main()
