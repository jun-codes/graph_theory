"""
Reusable inference helpers for the line-graph crease-pattern classifier.

This module is separate from the GA and the original vertex-GNN pipeline.  It
loads line_best_model.pt and scores NetworkX crease-pattern graphs after
converting them to the crease-as-node line-graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx
import numpy as np
import torch

from LineGraph_Classifier import LineGraphClassifier
from torch_geometric.loader import DataLoader
from line_graph_features import (
    LINE_EDGE_FEATURE_NAMES,
    LINE_NODE_FEATURE_NAMES,
    kawasaki_violation,
    line_graph_to_pyg,
    maekawa_violation,
)


BASE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE / "models" / "line_best_model.pt"


@dataclass
class LineGraphScore:
    index: int
    valid_prob: float
    invalid_prob: float
    prediction: int
    nodes: int
    creases: int
    line_edges: int
    kaw_mean: float
    kaw_max: float
    mae_mean: float
    mae_max: float
    source: str = ""


def _nonzero_mean(values: Iterable[float], eps: float = 1e-9) -> float:
    nonzero = [float(v) for v in values if abs(float(v)) > eps]
    return float(np.mean(nonzero)) if nonzero else 0.0


def constraint_stats(G: nx.Graph) -> dict[str, float]:
    kaw_values = [kawasaki_violation(G, n) for n in G.nodes()]
    mae_values = [maekawa_violation(G, n) for n in G.nodes()]
    return {
        "kaw_mean": _nonzero_mean(kaw_values),
        "kaw_max": float(max(kaw_values)) if kaw_values else 0.0,
        "mae_mean": _nonzero_mean(mae_values),
        "mae_max": float(max(mae_values)) if mae_values else 0.0,
    }


class LineGraphScorer:
    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: Optional[str | torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model = LineGraphClassifier(
            in_channels=checkpoint.get("in_channels", len(LINE_NODE_FEATURE_NAMES)),
            edge_dim=checkpoint.get("edge_dim", len(LINE_EDGE_FEATURE_NAMES)),
            hidden=checkpoint.get("hidden", 96),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def score_graph(self, G: nx.Graph, index: int = 0, source: str = "") -> LineGraphScore:
        data = line_graph_to_pyg(G, label=0)
        if data is None:
            raise ValueError("Cannot score a graph with no creases")

        data = data.to(self.device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_attr, batch)
            prob = torch.softmax(out, dim=1)[0].detach().cpu()

        stats = constraint_stats(G)
        return LineGraphScore(
            index=index,
            valid_prob=float(prob[1].item()),
            invalid_prob=float(prob[0].item()),
            prediction=int(torch.argmax(prob).item()),
            nodes=G.number_of_nodes(),
            creases=G.number_of_edges(),
            line_edges=int(data.edge_index.size(1)),
            kaw_mean=stats["kaw_mean"],
            kaw_max=stats["kaw_max"],
            mae_mean=stats["mae_mean"],
            mae_max=stats["mae_max"],
            source=source,
        )

    def score_graphs(
        self,
        graphs: Iterable[nx.Graph],
        source: str = "",
        batch_size: int = 16,
    ) -> List[LineGraphScore]:
        data_items = []
        metadata = []

        for i, G in enumerate(graphs):
            data = line_graph_to_pyg(G, label=0)
            if data is None:
                continue
            stats = constraint_stats(G)
            data_items.append(data)
            metadata.append(
                {
                    "index": i,
                    "nodes": G.number_of_nodes(),
                    "creases": G.number_of_edges(),
                    "line_edges": int(data.edge_index.size(1)),
                    "kaw_mean": stats["kaw_mean"],
                    "kaw_max": stats["kaw_max"],
                    "mae_mean": stats["mae_mean"],
                    "mae_max": stats["mae_max"],
                    "source": source,
                }
            )

        if not data_items:
            return []

        probs = []
        loader = DataLoader(data_items, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs.extend(torch.softmax(out, dim=1).detach().cpu())

        scores = []
        for prob, meta in zip(probs, metadata):
            scores.append(
                LineGraphScore(
                    index=meta["index"],
                    valid_prob=float(prob[1].item()),
                    invalid_prob=float(prob[0].item()),
                    prediction=int(torch.argmax(prob).item()),
                    nodes=meta["nodes"],
                    creases=meta["creases"],
                    line_edges=meta["line_edges"],
                    kaw_mean=meta["kaw_mean"],
                    kaw_max=meta["kaw_max"],
                    mae_mean=meta["mae_mean"],
                    mae_max=meta["mae_max"],
                    source=meta["source"],
                )
            )
        return scores


def summarize_scores(scores: List[LineGraphScore], expected_label: Optional[int] = None):
    if not scores:
        return {
            "count": 0,
            "accuracy": None,
            "valid_prob_mean": None,
            "valid_prob_median": None,
            "valid_prob_min": None,
            "valid_prob_max": None,
            "pred_valid": 0,
            "pred_invalid": 0,
        }

    valid_probs = np.array([s.valid_prob for s in scores], dtype=float)
    predictions = np.array([s.prediction for s in scores], dtype=int)
    summary = {
        "count": len(scores),
        "accuracy": None,
        "valid_prob_mean": float(np.mean(valid_probs)),
        "valid_prob_median": float(np.median(valid_probs)),
        "valid_prob_min": float(np.min(valid_probs)),
        "valid_prob_max": float(np.max(valid_probs)),
        "pred_valid": int(np.sum(predictions == 1)),
        "pred_invalid": int(np.sum(predictions == 0)),
        "kaw_mean": float(np.mean([s.kaw_mean for s in scores])),
        "mae_mean": float(np.mean([s.mae_mean for s in scores])),
    }
    if expected_label is not None:
        summary["accuracy"] = float(np.mean(predictions == expected_label))
    return summary
