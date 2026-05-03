from __future__ import annotations

import argparse
import copy
import csv
import math
import pickle
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gradient_geometry_repair import gradient_repair_kawasaki_symmetry
from kawasaki_projection_repair import repair_kawasaki_projection
from maekawa_z3_repair import repair_maekawa_z3
import origami_constraints as oc


SCALE = 200.0


def profile_settings(name: str) -> dict[str, float | int]:
    profiles = {
        "quick": {
            "projection_passes": 3,
            "projection_top_k": 6,
            "projection_step": 0.62,
            "projection_max_move": 14.0,
            "min_gap_deg": 5.0,
            "local_worst_k": 4,
            "local_tries": 24,
            "local_radius": 7.0,
            "grad_steps": 12,
            "grad_lr": 0.28,
            "grad_kaw_weight": 1.25,
            "grad_symmetry_weight": 1.15,
            "grad_move_weight": 0.012,
            "grad_max_move": 12.0,
            "pair_tol": 165.0,
            "sym_sweeps": 3,
            "projection_2_passes": 2,
            "projection_2_top_k": 6,
            "projection_2_step": 0.45,
            "projection_2_max_move": 10.0,
            "objective_slack": 0.02,
        },
        "strenuous": {
            "projection_passes": 7,
            "projection_top_k": 10,
            "projection_step": 0.68,
            "projection_max_move": 20.0,
            "min_gap_deg": 4.0,
            "local_worst_k": 7,
            "local_tries": 130,
            "local_radius": 9.5,
            "grad_steps": 55,
            "grad_lr": 0.30,
            "grad_kaw_weight": 1.55,
            "grad_symmetry_weight": 1.70,
            "grad_move_weight": 0.010,
            "grad_max_move": 15.0,
            "pair_tol": 185.0,
            "sym_sweeps": 6,
            "projection_2_passes": 5,
            "projection_2_top_k": 10,
            "projection_2_step": 0.50,
            "projection_2_max_move": 13.0,
            "objective_slack": 0.035,
        },
    }
    return profiles[name]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, item) -> None:
    with path.open("wb") as handle:
        pickle.dump(item, handle)


def symmetry_error(G: nx.Graph) -> float:
    graph_bbox = oc.bbox(G)
    mutable = [n for n in G.nodes() if not oc.is_boundary_node(G, n, graph_bbox)]
    if len(mutable) < 2:
        return 0.0
    coords = np.array([oc.coords(G, n) for n in mutable], dtype=float)
    errors = []
    for x, y in coords:
        dists = np.sqrt((coords[:, 0] + x) ** 2 + (coords[:, 1] - y) ** 2)
        errors.append(float(np.min(dists) / SCALE))
    return float(np.mean(errors))


def objective(G: nx.Graph, *, symmetry_weight: float = 0.85) -> float:
    crossing_penalty = 100.0 if oc.has_crossings(G) else 0.0
    return (
        oc.kawasaki_penalty(G)
        + 0.20 * oc.maekawa_penalty(G)
        + symmetry_weight * symmetry_error(G)
        + crossing_penalty
    )


def mirror_pairs(G: nx.Graph, *, max_dist: float = 120.0) -> list[tuple[int, int]]:
    graph_bbox = oc.bbox(G)
    mutable = [n for n in G.nodes() if not oc.is_boundary_node(G, n, graph_bbox)]
    left = [n for n in mutable if oc.coords(G, n)[0] < 0]
    right = [n for n in mutable if oc.coords(G, n)[0] >= 0]
    used_left = set()
    pairs = []

    for r in sorted(right, key=lambda n: abs(oc.coords(G, n)[0]), reverse=True):
        rx, ry = oc.coords(G, r)
        candidates = [n for n in left if n not in used_left]
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda n: (oc.coords(G, n)[0] + rx) ** 2 + (oc.coords(G, n)[1] - ry) ** 2,
        )
        bx, by = oc.coords(G, best)
        if math.hypot(bx + rx, by - ry) <= max_dist:
            used_left.add(best)
            pairs.append((r, best))
    return pairs


def set_node_xy(G: nx.Graph, node: int, x: float, y: float) -> None:
    min_x, max_x, min_y, max_y = oc.bbox(G)
    inset = 20.0
    G.nodes[node]["x"] = float(np.clip(x, min_x + inset, max_x - inset))
    G.nodes[node]["y"] = float(np.clip(y, min_y + inset, max_y - inset))


def symmetrize_coordinates(G: nx.Graph, *, sweeps: int = 2) -> nx.Graph:
    best = copy.deepcopy(G)
    best_score = objective(best)

    for _ in range(sweeps):
        changed = False
        for right, left in mirror_pairs(best):
            trial = copy.deepcopy(best)
            rx, ry = oc.coords(trial, right)
            lx, ly = oc.coords(trial, left)
            x_abs = 0.5 * (abs(rx) + abs(lx))
            y_mid = 0.5 * (ry + ly)
            set_node_xy(trial, right, x_abs, y_mid)
            set_node_xy(trial, left, -x_abs, y_mid)
            trial = oc.recompute_features(trial)

            trial_score = objective(trial)
            if not oc.has_crossings(trial) and trial_score <= best_score + 0.015:
                best = trial
                best_score = trial_score
                changed = True

        if not changed:
            break

    return best


def local_heatmap_descent(
    G: nx.Graph,
    *,
    worst_k: int = 5,
    tries_per_vertex: int = 80,
    radius: float = 9.0,
) -> nx.Graph:
    best = copy.deepcopy(G)
    graph_bbox = oc.bbox(best)
    global_score = objective(best)

    worst = [
        (node, value)
        for node, value in oc.worst_vertices(best, k=worst_k, metric="kawasaki")
        if value > 1e-4
    ]

    for hotspot, _ in worst:
        affected = [hotspot] + oc.non_border_neighbors(best, hotspot)
        movable = [
            node for node in affected
            if node in best and not oc.is_boundary_node(best, node, graph_bbox)
        ]
        if not movable:
            continue

        local_nodes = set(affected)
        for node in affected:
            if node in best:
                local_nodes.update(best.neighbors(node))

        def local_score(graph: nx.Graph) -> float:
            return float(sum(oc.kawasaki_at(graph, n) for n in local_nodes if n in graph))

        best_local = local_score(best)
        for _ in range(tries_per_vertex):
            node = random.choice(movable)
            ox, oy = oc.coords(best, node)
            angle = random.random() * 2.0 * math.pi
            step = radius * (0.25 + 0.75 * random.random())
            trial = copy.deepcopy(best)
            set_node_xy(trial, node, ox + step * math.cos(angle), oy + step * math.sin(angle))
            trial = oc.recompute_features(trial)
            if oc.has_crossings(trial):
                continue

            trial_local = local_score(trial)
            trial_global = objective(trial)
            if trial_local + 1e-5 < best_local and trial_global <= global_score + 0.025:
                best = trial
                best_local = trial_local
                global_score = min(global_score, trial_global)

    return best


def repair_one(
    G: nx.Graph,
    *,
    rounds: int,
    profile: dict[str, float | int],
) -> tuple[nx.Graph, list[dict[str, object]]]:
    current = oc.recompute_features(copy.deepcopy(G))
    history = []

    for round_idx in range(1, rounds + 1):
        print(f"  repair round {round_idx}/{rounds}", flush=True)
        before = oc.constraint_summary(current)
        before_sym = symmetry_error(current)
        before_obj = objective(current)
        worst_before = oc.worst_vertices(current, k=4, metric="kawasaki")

        candidate, projection_stats = repair_kawasaki_projection(
            current,
            passes=int(profile["projection_passes"]),
            top_k=int(profile["projection_top_k"]),
            step=float(profile["projection_step"]),
            max_move=float(profile["projection_max_move"]),
            min_gap_deg=float(profile["min_gap_deg"]),
        )
        candidate = local_heatmap_descent(
            candidate,
            worst_k=int(profile["local_worst_k"]),
            tries_per_vertex=int(profile["local_tries"]),
            radius=float(profile["local_radius"]),
        )
        candidate, grad_stats = gradient_repair_kawasaki_symmetry(
            candidate,
            steps=int(profile["grad_steps"]),
            lr=float(profile["grad_lr"]),
            kaw_weight=float(profile["grad_kaw_weight"]),
            symmetry_weight=float(profile["grad_symmetry_weight"]),
            move_weight=float(profile["grad_move_weight"]),
            max_move=float(profile["grad_max_move"]),
            pair_tol=float(profile["pair_tol"]),
        )
        candidate = symmetrize_coordinates(candidate, sweeps=int(profile["sym_sweeps"]))
        candidate, projection_stats_2 = repair_kawasaki_projection(
            candidate,
            passes=int(profile["projection_2_passes"]),
            top_k=int(profile["projection_2_top_k"]),
            step=float(profile["projection_2_step"]),
            max_move=float(profile["projection_2_max_move"]),
            min_gap_deg=float(profile["min_gap_deg"]),
        )
        candidate, z3_stats = repair_maekawa_z3(candidate, timeout_ms=1000)
        candidate = oc.recompute_features(candidate)

        after = oc.constraint_summary(candidate)
        after_sym = symmetry_error(candidate)
        after_obj = objective(candidate)
        accepted = (
            not oc.has_crossings(candidate)
            and after.maekawa <= max(before.maekawa, 1e-6)
            and after_obj <= before_obj + float(profile["objective_slack"])
            and (
                after.kawasaki < before.kawasaki - 1e-4
                or after_sym < before_sym - 1e-4
            )
        )

        if accepted:
            current = candidate

        history.append({
            "round": round_idx,
            "accepted": accepted,
            "before_kaw": before.kawasaki,
            "after_kaw": after.kawasaki if accepted else before.kawasaki,
            "before_max_kaw": before.max_kawasaki,
            "after_max_kaw": after.max_kawasaki if accepted else before.max_kawasaki,
            "before_sym": before_sym,
            "after_sym": after_sym if accepted else before_sym,
            "before_mae": before.maekawa,
            "after_mae": after.maekawa if accepted else before.maekawa,
            "crossing_free": after.crossing_free if accepted else before.crossing_free,
            "projection": projection_stats.reason,
            "projection_2": projection_stats_2.reason,
            "gradient": grad_stats.reason,
            "z3": z3_stats.status,
            "worst_before": "; ".join(f"{node}:{value:.4f}" for node, value in worst_before),
        })

        if not accepted:
            break

    return current, history


def draw_top6(graphs: list[nx.Graph], path: Path, *, violation: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, graph in enumerate(graphs[:6]):
        summary = oc.constraint_summary(graph)
        title = (
            f"Rank {i + 1} Kaw={summary.kawasaki:.3f} "
            f"MaxK={summary.max_kawasaki:.3f} Sym={symmetry_error(graph):.3f}"
        )
        ax = axes.flatten()[i]
        if violation:
            oc.visualise_with_violations(graph, title=title, ax=ax)
        else:
            draw_plain_graph(graph, title=title, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def draw_plain_graph(G: nx.Graph, title: str, ax) -> None:
    pos = {node: oc.coords(G, node) for node in G.nodes()}
    for fold_type, color, width in [(2, "red", 1.2), (3, "blue", 0.8), (1, "black", 2.0)]:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (u, v) for u, v, data in G.edges(data=True)
                if data.get("fold_type") == fold_type
            ],
            edge_color=color,
            width=width,
            ax=ax,
        )
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color="black", ax=ax)
    ax.set_title(title, fontsize=8)
    ax.axis("equal")
    ax.axis("off")


def write_report(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heatmap-guided Kawasaki cleanup and symmetry pass for runs/kaw_v4."
    )
    parser.add_argument("--run-dir", default="runs/kaw_v4")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--profile", choices=("quick", "strenuous"), default="quick")
    parser.add_argument("--prefix")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    settings = profile_settings(args.profile)
    prefix = args.prefix
    if prefix is None:
        prefix = (
            "manual_strenuous_kawasaki_symmetric"
            if args.profile == "strenuous"
            else "manual_kawasaki_symmetric"
        )

    best_path = run_dir / "line_kawasaki_z3_best_generated.pkl"
    diverse_path = run_dir / "line_kawasaki_z3_diverse_top6.pkl"
    best_graph = load_pickle(best_path)
    diverse_graphs = load_pickle(diverse_path)

    print(f"Repairing best graph with {args.profile} profile", flush=True)
    repaired_best, best_history = repair_one(
        best_graph,
        rounds=args.rounds,
        profile=settings,
    )
    repaired_diverse = []
    report_rows = []

    for graph_index, graph in enumerate(diverse_graphs):
        print(f"\nRepairing diverse graph {graph_index}", flush=True)
        repaired, history = repair_one(
            graph,
            rounds=args.rounds,
            profile=settings,
        )
        repaired_diverse.append(repaired)
        for row in history:
            report_rows.append({"graph": graph_index, **row})

    for row in best_history:
        report_rows.append({"graph": "best", **row})

    if repaired_diverse and objective(repaired_diverse[0]) < objective(repaired_best):
        repaired_best = copy.deepcopy(repaired_diverse[0])
        report_rows.append({
            "graph": "best",
            "round": "use_repaired_rank_1",
            "accepted": True,
            "before_kaw": oc.kawasaki_penalty(best_graph),
            "after_kaw": oc.kawasaki_penalty(repaired_best),
            "before_max_kaw": oc.constraint_summary(best_graph).max_kawasaki,
            "after_max_kaw": oc.constraint_summary(repaired_best).max_kawasaki,
            "before_sym": symmetry_error(best_graph),
            "after_sym": symmetry_error(repaired_best),
            "before_mae": oc.maekawa_penalty(best_graph),
            "after_mae": oc.maekawa_penalty(repaired_best),
            "crossing_free": not oc.has_crossings(repaired_best),
            "projection": "rank_1",
            "projection_2": "rank_1",
            "gradient": "rank_1",
            "z3": "rank_1",
            "worst_before": "best matched diverse rank 1",
        })

    best_out = run_dir / f"{prefix}_best_generated.pkl"
    diverse_out = run_dir / f"{prefix}_diverse_top6.pkl"
    report_out = run_dir / f"{prefix}_report.csv"
    top6_out = run_dir / f"{prefix}_top6.png"
    violation_out = run_dir / f"{prefix}_violation_top6.png"

    save_pickle(best_out, repaired_best)
    save_pickle(diverse_out, repaired_diverse)
    write_report(report_out, report_rows)
    draw_top6(
        repaired_diverse,
        top6_out,
        violation=False,
    )
    draw_top6(
        repaired_diverse,
        violation_out,
        violation=True,
    )

    print("Saved manual Kawasaki + symmetry repair outputs:")
    print(f"  {best_out}")
    print(f"  {diverse_out}")
    print(f"  {report_out}")
    print(f"  {top6_out}")
    print(f"  {violation_out}")

    print("\nBefore:")
    for i, graph in enumerate(diverse_graphs):
        s = oc.constraint_summary(graph)
        print(f"  [{i}] kaw={s.kawasaki:.4f} max_kaw={s.max_kawasaki:.4f} sym={symmetry_error(graph):.4f}")

    print("\nAfter:")
    for i, graph in enumerate(repaired_diverse):
        s = oc.constraint_summary(graph)
        print(f"  [{i}] kaw={s.kawasaki:.4f} max_kaw={s.max_kawasaki:.4f} sym={symmetry_error(graph):.4f}")


if __name__ == "__main__":
    main()
