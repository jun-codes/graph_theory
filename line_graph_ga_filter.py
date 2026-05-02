"""
Optional line-GNN filter for GA fitness.

This file does not modify Genetic_Algo.py.  It exposes a small helper that can
be imported by the GA when we want to penalize candidates the line-graph
classifier considers invalid.

Example integration inside a GA fitness function:

    from line_graph_ga_filter import LineGraphGAFilter

    line_filter = LineGraphGAFilter(min_valid_prob=0.50, penalty_weight=1.0)

    score = existing_score
    score -= line_filter.penalty(G)

The conservative default is filter-style behavior: only penalize candidates
below a valid-probability threshold, instead of directly optimizing the line
classifier probability.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from line_graph_scoring import DEFAULT_CHECKPOINT, LineGraphScorer


class LineGraphGAFilter:
    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        min_valid_prob: float = 0.50,
        penalty_weight: float = 1.0,
    ):
        self.min_valid_prob = float(min_valid_prob)
        self.penalty_weight = float(penalty_weight)
        self.scorer = LineGraphScorer(checkpoint_path)

    def valid_probability(self, G: nx.Graph) -> float:
        return self.scorer.score_graph(G).valid_prob

    def penalty(self, G: nx.Graph) -> float:
        valid_prob = self.valid_probability(G)
        shortfall = max(0.0, self.min_valid_prob - valid_prob)
        return self.penalty_weight * shortfall

    def adjusted_fitness(self, base_fitness: float, G: nx.Graph) -> float:
        return float(base_fitness) - self.penalty(G)
