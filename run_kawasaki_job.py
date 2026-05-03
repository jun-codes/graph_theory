from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kawasaki GA into one output directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--elite-keep", type=int, required=True)
    parser.add_argument("--mutations-per", type=int, required=True)
    parser.add_argument("--kawasaki-gate-end", type=float)
    parser.add_argument("--symmetry-weight-start", type=float)
    parser.add_argument("--symmetry-weight-end", type=float)
    parser.add_argument("--symmetry-repair-interval", type=int)
    parser.add_argument("--mirror-move-prob", type=float)
    parser.add_argument("--mirror-node-tol", type=float)
    parser.add_argument("--mirror-edge-tol", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.stderr = sys.stdout

    import Genetic_Algo_Line_Filtered_Repaired_Z3_Topology_Kawasaki as ga

    ga.BASE = str(out_dir)
    if args.kawasaki_gate_end is not None:
        ga.KAWASAKI_GATE_END = args.kawasaki_gate_end
        print(f"KAWASAKI_GATE_END set to {ga.KAWASAKI_GATE_END}")
    if args.symmetry_weight_start is not None:
        ga.SYMMETRY_WEIGHT_START = args.symmetry_weight_start
        print(f"SYMMETRY_WEIGHT_START set to {ga.SYMMETRY_WEIGHT_START}")
    if args.symmetry_weight_end is not None:
        ga.SYMMETRY_WEIGHT_END = args.symmetry_weight_end
        print(f"SYMMETRY_WEIGHT_END set to {ga.SYMMETRY_WEIGHT_END}")
    if args.symmetry_repair_interval is not None:
        ga.SYMMETRY_REPAIR_INTERVAL = args.symmetry_repair_interval
        print(f"SYMMETRY_REPAIR_INTERVAL set to {ga.SYMMETRY_REPAIR_INTERVAL}")
    if args.mirror_move_prob is not None:
        ga.MIRROR_MOVE_PROB = args.mirror_move_prob
        print(f"MIRROR_MOVE_PROB set to {ga.MIRROR_MOVE_PROB}")
    if args.mirror_node_tol is not None:
        ga.MIRROR_NODE_TOL = args.mirror_node_tol
        print(f"MIRROR_NODE_TOL set to {ga.MIRROR_NODE_TOL}")
    if args.mirror_edge_tol is not None:
        ga.MIRROR_EDGE_TOL = args.mirror_edge_tol
        print(f"MIRROR_EDGE_TOL set to {ga.MIRROR_EDGE_TOL}")

    print(
        "Retry config: "
        f"out_dir={out_dir} "
        f"population_size={args.population_size} "
        f"generations={args.generations} "
        f"elite_keep={args.elite_keep} "
        f"mutations_per={args.mutations_per}",
        flush=True,
    )
    ga.run_ga(
        population_size=args.population_size,
        generations=args.generations,
        elite_keep=args.elite_keep,
        mutations_per=args.mutations_per,
    )


if __name__ == "__main__":
    main()
