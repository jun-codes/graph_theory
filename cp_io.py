import os


def format_cp_number(value, decimals=6):
    value = float(value)
    if abs(value) < 10 ** (-(decimals + 1)):
        value = 0.0
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def _canonical_edge_record(G, u, v, fold_type, clip_box=None):
    x1 = float(G.nodes[u]["x"])
    y1 = float(G.nodes[u]["y"])
    x2 = float(G.nodes[v]["x"])
    y2 = float(G.nodes[v]["y"])

    if clip_box is not None:
        lo, hi = -float(clip_box), float(clip_box)
        x1 = max(lo, min(hi, x1))
        y1 = max(lo, min(hi, y1))
        x2 = max(lo, min(hi, x2))
        y2 = max(lo, min(hi, y2))

    # Keep a stable endpoint order in the text file.
    if (x2, y2) < (x1, y1):
        x1, y1, x2, y2 = x2, y2, x1, y1

    if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
        return None

    return int(fold_type), x1, y1, x2, y2


def cp_edge_records(G, clip_box=None):
    records = []
    for u, v, data in G.edges(data=True):
        fold_type = data.get("fold_type", 2)
        record = _canonical_edge_record(G, u, v, fold_type, clip_box=clip_box)
        if record is not None:
            records.append(record)
    records.sort()
    return records


def write_cp_file(G, path, decimals=6, clip_box=None):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for fold_type, x1, y1, x2, y2 in cp_edge_records(G, clip_box=clip_box):
            f.write(
                f"{fold_type} "
                f"{format_cp_number(x1, decimals)} "
                f"{format_cp_number(y1, decimals)} "
                f"{format_cp_number(x2, decimals)} "
                f"{format_cp_number(y2, decimals)}\n"
            )


def write_cp_collection(graphs, directory, prefix="pattern", decimals=6, clip_box=None):
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, G in enumerate(graphs, start=1):
        path = os.path.join(directory, f"{prefix}_{i}.cp")
        write_cp_file(G, path, decimals=decimals, clip_box=clip_box)
        paths.append(path)
    return paths
