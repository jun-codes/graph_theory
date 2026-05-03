from __future__ import annotations

import contextlib
import importlib
import io
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import origami_constraints as oc


os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
PREFIX = "z3_symmetry_kawasaki"
OUTPUTS_DIR = ROOT / "outputs"
FINAL_OUTPUT_DIR = OUTPUTS_DIR / PREFIX
FINAL_RUN_DIRS = [
    OUTPUTS_DIR / PREFIX,
    OUTPUTS_DIR / f"{PREFIX}_1",
    ROOT,
]
RUN_LOG = FINAL_OUTPUT_DIR / f"{PREFIX}_streamlit_run.log"


def newest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime_ns)
    return None


def newest_matching(patterns: list[str], directories: list[Path] = FINAL_RUN_DIRS) -> Path | None:
    matches: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches.extend(directory.glob(pattern))
    return newest_existing(matches)


def default_artifact_path(filename: str) -> Path:
    return FINAL_OUTPUT_DIR / filename


def refresh_artifact_paths() -> None:
    global TOP6_IMAGE, CONVERGENCE_IMAGE, VIOLATION_IMAGE, BEST_PKL, DIVERSE_PKL, BEST_CP, CP_DIR

    TOP6_IMAGE = newest_matching([f"{PREFIX}_ga_top6*.png"]) or default_artifact_path(f"{PREFIX}_ga_top6.png")
    CONVERGENCE_IMAGE = newest_matching([f"{PREFIX}_ga_convergence*.png"]) or default_artifact_path(f"{PREFIX}_ga_convergence.png")
    VIOLATION_IMAGE = newest_matching([f"{PREFIX}_violation_top6*.png"]) or default_artifact_path(f"{PREFIX}_violation_top6.png")
    BEST_PKL = newest_matching([f"{PREFIX}_best_generated*.pkl"]) or default_artifact_path(f"{PREFIX}_best_generated.pkl")
    DIVERSE_PKL = newest_matching([f"{PREFIX}_diverse_top6*.pkl"]) or default_artifact_path(f"{PREFIX}_diverse_top6.pkl")
    BEST_CP = newest_matching([f"{PREFIX}_best_generated*.cp"]) or default_artifact_path(f"{PREFIX}_best_generated.cp")
    CP_DIR = newest_matching([f"{PREFIX}_diverse_top6*_cp"]) or default_artifact_path(f"{PREFIX}_diverse_top6_cp")


refresh_artifact_paths()


def load_pickle(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def cp_files() -> list[Path]:
    if not CP_DIR.exists():
        return []
    return sorted(CP_DIR.glob("rank_*.cp"))


def zip_cp_files(paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
        if BEST_CP.exists():
            archive.write(BEST_CP, arcname=BEST_CP.name)
    buffer.seek(0)
    return buffer.getvalue()


def cp_dataset_rows() -> list[dict[str, object]]:
    rows = []
    for path in cp_files():
        line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        rows.append(
            {
                "file": path.name,
                "creases": line_count,
                "format": "fold_type x1 y1 x2 y2",
                "source": "z3 symmetry Kawasaki top-6 export",
            }
        )
    if BEST_CP.exists():
        line_count = len(BEST_CP.read_text(encoding="utf-8", errors="replace").splitlines())
        rows.insert(
            0,
            {
                "file": BEST_CP.name,
                "creases": line_count,
                "format": "fold_type x1 y1 x2 y2",
                "source": "best generated export",
            },
        )
    return rows


def graph_rows() -> list[dict[str, object]]:
    graphs = load_pickle(DIVERSE_PKL)
    if not isinstance(graphs, list):
        return []

    rows = []
    for idx, graph in enumerate(graphs, start=1):
        summary = oc.constraint_summary(graph)
        cp_path = CP_DIR / f"rank_{idx}.cp"
        rows.append(
            {
                "rank": idx,
                "nodes": graph.number_of_nodes(),
                "creases": graph.number_of_edges(),
                "kawasaki": round(summary.kawasaki, 4),
                "max_kawasaki": round(summary.max_kawasaki, 4),
                "maekawa": round(summary.maekawa, 4),
                "crossing_free": summary.crossing_free,
                "cp_file": cp_path.name if cp_path.exists() else "",
            }
        )
    return rows


def render_graph(graph, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    oc.visualise_with_violations(graph, title=title, ax=ax)
    fig.tight_layout(pad=0.2)
    return fig


def run_algorithm(
    *,
    population_size: int,
    generations: int,
    elite_keep: int,
    mutations_per: int,
    symmetry_mode: str,
) -> str:
    output = io.StringIO()
    start = time.perf_counter()

    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        module = importlib.import_module("Genetic_Algo_Z3_Topology_Symmetry_Kawasaki")
        module.BASE = str(ROOT)
        module.OUTPUT_DIR = FINAL_OUTPUT_DIR
        module.USE_SYMMETRY = True
        module.SYMMETRY_MODE = symmetry_mode
        module.run_ga(
            population_size=population_size,
            generations=generations,
            elite_keep=elite_keep,
            mutations_per=mutations_per,
        )

    elapsed = time.perf_counter() - start
    text = output.getvalue()
    text += f"\nStreamlit run completed in {elapsed:.1f}s\n"
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(text, encoding="utf-8")
    return text


def render_artifacts() -> None:
    refresh_artifact_paths()
    st.subheader("Generated Candidates")

    if TOP6_IMAGE.exists():
        st.image(TOP6_IMAGE.read_bytes(), caption=TOP6_IMAGE.name, width="stretch")
    else:
        st.info("No top-6 image exists yet. Run the algorithm to generate one.")

    artifact_cols = st.columns(2)
    if CONVERGENCE_IMAGE.exists():
        with artifact_cols[0]:
            st.image(
                CONVERGENCE_IMAGE.read_bytes(),
                caption=CONVERGENCE_IMAGE.name,
                width="stretch",
            )
    if VIOLATION_IMAGE.exists():
        with artifact_cols[1]:
            st.image(
                VIOLATION_IMAGE.read_bytes(),
                caption=VIOLATION_IMAGE.name,
                width="stretch",
            )

    rows = graph_rows()
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        graphs = load_pickle(DIVERSE_PKL)
        if isinstance(graphs, list) and graphs:
            st.subheader("Individual Graph Views")
            cols = st.columns(3)
            for idx, graph in enumerate(graphs[:6], start=1):
                with cols[(idx - 1) % 3]:
                    fig = render_graph(graph, f"rank {idx}")
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

    cp_paths = cp_files()
    if cp_paths or BEST_CP.exists():
        st.subheader("CP File Dataset")
        dataset_rows = cp_dataset_rows()
        if dataset_rows:
            st.dataframe(pd.DataFrame(dataset_rows), width="stretch", hide_index=True)
            st.caption(
                "Each .cp row stores one crease as: fold_type x1 y1 x2 y2, "
                "where 1=border, 2=mountain, and 3=valley."
            )

        cols = st.columns(3)

        if BEST_CP.exists():
            with cols[0]:
                st.download_button(
                    "Download best CP",
                    data=BEST_CP.read_bytes(),
                    file_name=BEST_CP.name,
                    mime="text/plain",
                    width="stretch",
                )

        with cols[1]:
            st.download_button(
                "Download all CP files",
                data=zip_cp_files(cp_paths),
                file_name=f"{PREFIX}_cp_files.zip",
                mime="application/zip",
                width="stretch",
                disabled=not cp_paths and not BEST_CP.exists(),
            )

        st.divider()
        for row_start in range(0, len(cp_paths), 3):
            row_cols = st.columns(3)
            for offset, col in enumerate(row_cols):
                idx = row_start + offset
                if idx >= len(cp_paths):
                    continue
                path = cp_paths[idx]
                with col:
                    st.download_button(
                        f"Download {path.stem}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/plain",
                        width="stretch",
                        key=f"cp_download_{path.name}_{path.stat().st_mtime_ns}",
                    )


def main() -> None:
    st.set_page_config(
        page_title="Z3 Symmetry Kawasaki Generator",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        div[data-testid="stDownloadButton"] button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Origami Generator")
    st.caption("Generate theorem-repaired crease-pattern graphs and download editable .cp files.")

    with st.sidebar:
        st.header("Run Settings")
        population_size = st.slider("Population", min_value=8, max_value=50, value=30, step=2)
        generations = st.slider("Generations", min_value=3, max_value=80, value=30, step=1)
        elite_keep = st.slider("Elite keep", min_value=2, max_value=12, value=6, step=1)
        mutations_per = st.slider("Mutations per child", min_value=1, max_value=6, value=3, step=1)
        symmetry_mode = st.radio("Symmetry", ["vertical", "diagonal"], horizontal=True)

        run_clicked = st.button(
            "Generate Crease Patterns",
            type="primary",
            width="stretch",
        )

    if run_clicked:
        if elite_keep >= population_size:
            st.error("Elite keep must be smaller than the population.")
        else:
            status = st.empty()
            status.info("Running the genetic algorithm. This can take several minutes.")
            with st.spinner("Generating crease patterns..."):
                try:
                    log_text = run_algorithm(
                        population_size=population_size,
                        generations=generations,
                        elite_keep=elite_keep,
                        mutations_per=mutations_per,
                        symmetry_mode=symmetry_mode,
                    )
                except Exception as exc:
                    status.error(f"Generation failed: {exc}")
                    st.exception(exc)
                else:
                    status.success("Generation complete. Artifacts refreshed.")
                    with st.expander("Run log", expanded=False):
                        st.code(log_text[-12000:])

    render_artifacts()

    if RUN_LOG.exists():
        with st.expander("Latest saved run log", expanded=False):
            st.code(RUN_LOG.read_text(encoding="utf-8", errors="replace")[-12000:])


if __name__ == "__main__":
    main()
