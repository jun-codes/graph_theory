"""
Train a crease-line graph classifier.

This is intentionally separate from GNN_Classifier.py.  The original CP graph
has vertices as nodes and creases as edges; this script converts each CP into a
line graph where creases are nodes, then trains a GINE classifier with edge
features describing how creases meet around CP vertices.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import pickle
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential

from line_graph_features import (
    LINE_EDGE_FEATURE_NAMES,
    LINE_NODE_FEATURE_NAMES,
    line_graph_to_pyg,
)


try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

    HAS_PYG = True
    PYG_IMPORT_ERROR = None
except ImportError as exc:
    DataLoader = None
    GINEConv = None
    global_max_pool = None
    global_mean_pool = None
    HAS_PYG = False
    PYG_IMPORT_ERROR = exc


BASE = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LineGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden: int = 96):
        if not HAS_PYG:
            raise RuntimeError(
                "LineGraphClassifier requires torch_geometric; "
                f"import failed with: {PYG_IMPORT_ERROR}"
            )
        super().__init__()

        def mlp(in_c: int, out_c: int):
            return Sequential(
                Linear(in_c, out_c),
                BatchNorm1d(out_c),
                ReLU(),
                Linear(out_c, out_c),
                ReLU(),
            )

        self.conv1 = GINEConv(mlp(in_channels, hidden), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
        self.conv3 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
        self.classifier = Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden, 2),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.classifier(pooled)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def build_dataset(
    positives_path: Path,
    negatives_path: Path,
    max_graphs: int | None = None,
    seed: int = 7,
) -> Tuple[List, int, int]:
    positives = load_pickle(positives_path)
    negatives = load_pickle(negatives_path)

    if max_graphs is not None:
        per_class = max(1, max_graphs // 2)
        rng = random.Random(seed)
        positives = rng.sample(positives, min(per_class, len(positives)))
        negatives = rng.sample(negatives, min(per_class, len(negatives)))

    dataset = []
    skipped_pos = 0
    skipped_neg = 0

    for i, graph in enumerate(positives):
        data = line_graph_to_pyg(graph, label=1)
        if data is None:
            skipped_pos += 1
            continue
        data.source_index = i
        dataset.append(data)

    for i, graph in enumerate(negatives):
        data = line_graph_to_pyg(graph, label=0)
        if data is None:
            skipped_neg += 1
            continue
        data.source_index = i
        dataset.append(data)

    return dataset, skipped_pos, skipped_neg


def split_dataset(dataset: Sequence, seed: int):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = [dataset[i] for i in indices[:n_train]]
    val = [dataset[i] for i in indices[n_train : n_train + n_val]]
    test = [dataset[i] for i in indices[n_train + n_val :]]
    return train, val, test


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.view(-1)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        correct += out.argmax(dim=1).eq(target).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.view(-1)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item() * batch.num_graphs
            correct += out.argmax(dim=1).eq(target).sum().item()
            total += batch.num_graphs

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the line-graph CP classifier.")
    parser.add_argument("--positives", type=Path, default=BASE / "graphs.pkl")
    parser.add_argument("--negatives", type=Path, default=BASE / "negatives_v3.pkl")
    parser.add_argument("--output", type=Path, default=BASE / "line_best_model.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional balanced subset size for quick smoke runs.",
    )
    return parser.parse_args()


def main():
    if not HAS_PYG:
        raise SystemExit(
            "LineGraph_Classifier.py requires torch_geometric, but it is not "
            f"installed/importable here: {PYG_IMPORT_ERROR}"
        )

    args = parse_args()
    set_seed(args.seed)

    print("Loading and converting CPs to line graphs...")
    dataset, skipped_pos, skipped_neg = build_dataset(
        args.positives,
        args.negatives,
        max_graphs=args.max_graphs,
        seed=args.seed,
    )
    if len(dataset) < 10:
        raise SystemExit(f"Dataset too small after conversion: {len(dataset)} graphs")

    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    print(
        f"Dataset: {len(train_set)} train | {len(val_set)} val | "
        f"{len(test_set)} test"
    )
    print(f"Skipped: {skipped_pos} positives | {skipped_neg} negatives")
    print(
        f"Features: {len(LINE_NODE_FEATURE_NAMES)} line-node | "
        f"{len(LINE_EDGE_FEATURE_NAMES)} line-edge"
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LineGraphClassifier(
        in_channels=len(LINE_NODE_FEATURE_NAMES),
        edge_dim=len(LINE_EDGE_FEATURE_NAMES),
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_state = None
    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"best_val={best_val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f} | test loss: {test_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": len(LINE_NODE_FEATURE_NAMES),
            "edge_dim": len(LINE_EDGE_FEATURE_NAMES),
            "hidden": args.hidden,
            "node_feature_names": LINE_NODE_FEATURE_NAMES,
            "edge_feature_names": LINE_EDGE_FEATURE_NAMES,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "history": history,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
