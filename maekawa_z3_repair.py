"""
Exact Maekawa fold-label repair using Z3.

The repair only changes non-border fold labels:
  1 = border (fixed)
  2 = mountain
  3 = valley

For every interior vertex with non-border degree d, Maekawa requires the
mountain count to be d/2 - 1 or d/2 + 1.  Vertices with odd non-border degree,
or non-border degree below 2, cannot be repaired by label assignment alone and
are reported as topology failures.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
from z3 import Bool, If, Optimize, Or, Sum, is_true, sat


EdgeKey = Tuple[int, int]


@dataclass
class MaekawaRepairStats:
    status: str
    changed_edges: int
    before_penalty: float
    after_penalty: float
    constrained_vertices: int
    odd_degree_vertices: int
    unsat_reason: str


def _edge_key(u: int, v: int) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def _is_interior_vertex(G: nx.Graph, node: int) -> bool:
    neighbors = list(G.neighbors(node))
    return len(neighbors) >= 2 and not all(
        G[node][nb].get("fold_type") == 1 for nb in neighbors
    )


def _mae_at(G: nx.Graph, node: int) -> float:
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    if all(G[node][nb].get("fold_type") == 1 for nb in neighbors):
        return 0.0

    mountain = sum(1 for nb in neighbors if G[node][nb].get("fold_type") == 2)
    valley = sum(1 for nb in neighbors if G[node][nb].get("fold_type") == 3)
    return float(abs(abs(mountain - valley) - 2))


def maekawa_penalty(G: nx.Graph) -> float:
    values = [_mae_at(G, node) for node in G.nodes()]
    nonzero = [value for value in values if value > 0]
    return float(sum(nonzero) / len(nonzero)) if nonzero else 0.0


def _non_border_edges(G: nx.Graph) -> List[EdgeKey]:
    return sorted(
        _edge_key(u, v)
        for u, v, data in G.edges(data=True)
        if data.get("fold_type") != 1
    )


def _incident_non_border_edges(G: nx.Graph, node: int) -> List[EdgeKey]:
    return sorted(
        _edge_key(node, nb)
        for nb in G.neighbors(node)
        if G[node][nb].get("fold_type") != 1
    )


def _current_is_mountain(G: nx.Graph, edge: EdgeKey) -> bool:
    u, v = edge
    return G[u][v].get("fold_type") == 2


def repair_maekawa_z3(
    G: nx.Graph,
    timeout_ms: int = 1000,
) -> tuple[nx.Graph, MaekawaRepairStats]:
    before = maekawa_penalty(G)
    edges = _non_border_edges(G)
    if not edges:
        stats = MaekawaRepairStats(
            status="skipped",
            changed_edges=0,
            before_penalty=before,
            after_penalty=before,
            constrained_vertices=0,
            odd_degree_vertices=0,
            unsat_reason="no non-border creases",
        )
        return G, stats

    constrained = []
    topology_bad = []
    for node in G.nodes():
        if not _is_interior_vertex(G, node):
            continue
        incident = _incident_non_border_edges(G, node)
        degree = len(incident)
        if degree < 2 or degree % 2 == 1:
            topology_bad.append((node, degree))
            continue
        constrained.append((node, incident))

    if topology_bad:
        reason = ", ".join(f"{node}:d{degree}" for node, degree in topology_bad[:12])
        if len(topology_bad) > 12:
            reason += f", ... +{len(topology_bad) - 12} more"
        stats = MaekawaRepairStats(
            status="topology_unsat",
            changed_edges=0,
            before_penalty=before,
            after_penalty=before,
            constrained_vertices=len(constrained),
            odd_degree_vertices=len(topology_bad),
            unsat_reason=reason,
        )
        return G, stats

    if not constrained:
        stats = MaekawaRepairStats(
            status="skipped",
            changed_edges=0,
            before_penalty=before,
            after_penalty=before,
            constrained_vertices=0,
            odd_degree_vertices=0,
            unsat_reason="no constrained interior vertices",
        )
        return G, stats

    variables: Dict[EdgeKey, object] = {
        edge: Bool(f"mv_{edge[0]}_{edge[1]}") for edge in edges
    }
    opt = Optimize()
    opt.set(timeout=timeout_ms)

    for _, incident in constrained:
        degree = len(incident)
        mountain_count = Sum([If(variables[edge], 1, 0) for edge in incident])
        opt.add(Or(
            mountain_count == degree // 2 - 1,
            mountain_count == degree // 2 + 1,
        ))

    change_terms = []
    for edge in edges:
        current = _current_is_mountain(G, edge)
        change_terms.append(If(variables[edge] == current, 0, 1))
    opt.minimize(Sum(change_terms))

    result = opt.check()
    if result != sat:
        stats = MaekawaRepairStats(
            status="unsat" if str(result) == "unsat" else "unknown",
            changed_edges=0,
            before_penalty=before,
            after_penalty=before,
            constrained_vertices=len(constrained),
            odd_degree_vertices=0,
            unsat_reason=str(result),
        )
        return G, stats

    model = opt.model()
    repaired = copy.deepcopy(G)
    changed = 0
    for edge in edges:
        u, v = edge
        was_mountain = _current_is_mountain(G, edge)
        is_mountain = is_true(model.evaluate(variables[edge], model_completion=True))
        new_fold = 2 if is_mountain else 3
        if is_mountain != was_mountain:
            changed += 1
        repaired[u][v]["fold_type"] = new_fold

    after = maekawa_penalty(repaired)
    stats = MaekawaRepairStats(
        status="sat",
        changed_edges=changed,
        before_penalty=before,
        after_penalty=after,
        constrained_vertices=len(constrained),
        odd_degree_vertices=0,
        unsat_reason="",
    )
    return repaired, stats
