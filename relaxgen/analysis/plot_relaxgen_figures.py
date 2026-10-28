#!/usr/bin/env python3
"""Figures for the Bi2Te3 RelaxGen pilot.

Each subcommand rescans its data, so rerunning picks up new results:

* ``scf-time``: wall time of one production QE SCF (PBEsol, 60/480 Ry, 6x6x6
  k-points, 39-40 atoms) by backend.  "From scratch" is an SCF that starts from
  atomic wavefunctions: every single point, and the first SCF of each relaxation
  session.  "Relaxation step" is every later SCF of a session, which starts from
  the previous geometry's wavefunctions.

The Stage A figures (uMLIP ensemble, 2124 starts x EqV3 / MACE-ft / UMA, ML fmax
1e-3) read the endpoint RMSD matrices written by
``umlip_ensemble/deep*/stagea_deep.py`` (one matrix per defect configuration):

* ``stagea-distance``: per start, how far apart two models' endpoints are,
  with translation-only matching and with lattice-symmetry matching;
* ``stagea-basins``: distinct endpoints found by each model alone and by all
  three pooled;
* ``stagea-energy-order``: for two distinct endpoints, do two models agree on
  which is lower in energy?
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PILOT = Path("/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot")
RELAXGEN = Path("/scratch/gqs7zr/relaxgen")
DEFAULT_OUT = RELAXGEN / "figures"
STAGE_A_SYMMETRY = RELAXGEN / "umlip_ensemble/deep_sym"
STAGE_A_TRANSLATION = RELAXGEN / "umlip_ensemble/deep"
MODELS = ("eqv3", "mace_ft", "uma")
MODEL_LABELS = {"eqv3": "EqV3 (fine-tuned)", "mace_ft": "MACE (fine-tuned)", "uma": "UMA (PBE, not fine-tuned)"}
SHORT = {"eqv3": "EqV3", "mace_ft": "MACE-ft", "uma": "UMA"}
PAIRS = (("eqv3", "mace_ft"), ("eqv3", "uma"), ("mace_ft", "uma"))

# Single-point campaigns at the production protocol.
SCF_STATUS_GLOBS = (
    PILOT / "09_umlip_dft_handoff_threshold/scf_tasks/*/status.json",
    PILOT / "10_qe_backend_check/scf_tasks/*/status.json",
    PILOT / "10_qe_backend_check_fullnode/scf_tasks/*/status.json",
    PILOT / "17_qe_backend_check_singlenode/scf_tasks/*/status.json",
    RELAXGEN / "threshold_rerun/*/scf_tasks/*/status.json",
)
# Relaxation campaigns at the production protocol (Gamma-only and 2x2x2 tests excluded).
RELAX_STATUS_GLOBS = (
    PILOT / "15_vbi001_umlip_dft_diversity/dft/tasks/*/status.json",
    RELAXGEN / "trajectories/*/tasks/*/status.json",
    RELAXGEN / "umlip_ensemble/stage_b/tasks/*/status.json",
)

# Reference palette (dataviz skill), light mode.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def node_count(nodelist: str) -> int:
    """Nodes in a Slurm nodelist such as ``a1,b2`` or ``udc-ba03-17c[0-1]``."""

    count = 0
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", nodelist):
        bracket = re.search(r"\[([^\]]*)\]", part)
        if not bracket:
            count += 1
            continue
        for piece in bracket.group(1).split(","):
            lo, _, hi = piece.partition("-")
            count += int(hi) - int(lo) + 1 if hi else 1
    return count


def backend_label(device: str) -> str | None:
    if device.startswith("cpu:"):
        _, nodelist, ranks, *_ = device.split(":")
        nodes = node_count(nodelist)
        ranks = int(ranks.removesuffix("ranks"))
        return f"CPU, {nodes} node{'s' if nodes > 1 else ''} x {ranks // nodes} ranks"
    if "B200" in device:
        return "B200 (1 GPU)"
    match = re.search(r"A100-\w+-(\d+)GB", device)
    if match:
        return f"A100 {match.group(1)} GB (1 GPU)"
    return None


def scf_times() -> list[tuple[str, str, float]]:
    """(backend, 'From scratch' | 'Relaxation step', minutes) for every measured SCF."""

    rows = []
    for pattern in SCF_STATUS_GLOBS:
        for path in glob.glob(str(pattern)):
            status = json.loads(Path(path).read_text())
            runtime = status.get("runtime", {})
            seconds = runtime.get("qe_wall_time_seconds")
            label = backend_label(runtime.get("allocated_gpu_name") or "")
            if status.get("execution_status") == "completed" and seconds and label:
                rows.append((label, "From scratch", seconds / 60))
    for pattern in RELAX_STATUS_GLOBS:
        for path in glob.glob(str(pattern)):
            status = json.loads(Path(path).read_text())
            for segment in status.get("segments", []):
                device = segment.get("allocated_device", "")
                label = backend_label(device.removeprefix("1x NVIDIA ") if not device.startswith("cpu") else device)
                steps = segment.get("step_wall_seconds", [])
                if not label or not steps:
                    continue
                rows.append((label, "From scratch", steps[0] / 60))
                rows.extend((label, "Relaxation step", s / 60) for s in steps[1:])
    return rows


def plot_scf_time(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    style(plt)
    rows = scf_times()
    by_backend: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for backend, kind, minutes in rows:
        by_backend[backend][kind].append(minutes)
    kinds = ("From scratch", "Relaxation step")
    # Fastest backend on top.
    order = sorted(by_backend, key=lambda b: np.median(by_backend[b]["From scratch"] or by_backend[b][kinds[1]]))

    rng = np.random.default_rng(0)
    figure, axis = plt.subplots(figsize=(9.0, 0.75 * len(order) + 1.6))
    offsets = {kinds[0]: 0.17, kinds[1]: -0.17}
    for row, backend in enumerate(order):
        y0 = len(order) - 1 - row
        for kind, color in zip(kinds, SERIES):
            values = np.asarray(by_backend[backend][kind])
            if values.size == 0:
                continue
            y = y0 + offsets[kind] + rng.uniform(-0.07, 0.07, values.size)
            axis.scatter(values, y, s=14, color=color, alpha=0.45, linewidths=0, zorder=2)
            median = float(np.median(values))
            axis.plot([median, median], [y0 + offsets[kind] - 0.13, y0 + offsets[kind] + 0.13],
                      color=color, linewidth=2.5, solid_capstyle="round", zorder=3)
            # Label past the slowest point so it never covers the data.
            axis.annotate(f"median {median:.0f} min  (n={values.size})", (values.max(), y0 + offsets[kind]),
                          xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(list(reversed(order)), color=TEXT)
    axis.set_ylim(-0.6, len(order) - 0.4)
    axis.set_xscale("log")
    ticks = [10, 15, 30, 60, 120, 240, 480]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([f"{t} min" if t < 60 else f"{t // 60} h" for t in ticks])
    lo = min(m for _, _, m in rows)
    hi = max(m for _, _, m in rows)
    axis.set_xlim(lo / 1.3, hi * 3.5)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("Wall time of one QE SCF (log scale)")
    axis.set_title("One production SCF by backend")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=6, label=k)
               for k, c in zip(kinds, SERIES)]
    axis.legend(handles=handles, loc="upper right", fontsize=9)
    figure.text(0.01, 0.01, "Bi2Te3 defect cells (39-40 atoms), PBEsol, 60/480 Ry, 6x6x6 k-points, conv_thr 1e-9. "
                "Bars: medians. Relaxation steps reuse the previous step's wavefunctions.",
                fontsize=8, color=TEXT_2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    out = args.output_dir / "scf_time_by_backend.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(out)
    return out


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------


def load_stage_a(directory: Path) -> tuple[dict[str, list], Any]:
    """Per-configuration endpoint rows [(model, candidate, energy)] and RMSD matrices."""

    meta = json.loads((directory / "stagea_meta.json").read_text())
    return meta, np.load(directory / "stagea_matrices.npz")


def per_start_pair_rmsd(meta, matrices) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for config, rows in meta.items():
        matrix = matrices[config]
        index = {(model, candidate): i for i, (model, candidate, _) in enumerate(rows)}
        for candidate in {candidate for _, candidate, _ in rows}:
            for a, b in PAIRS:
                if (a, candidate) in index and (b, candidate) in index:
                    values[a, b].append(float(matrix[index[a, candidate], index[b, candidate]]))
    return values


def save(figure, args: argparse.Namespace, name: str) -> Path:
    out = args.output_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170)
    print(out)
    return out


def plot_stage_a_distance(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    symmetric = per_start_pair_rmsd(*load_stage_a(args.symmetry_dir))
    translation = per_start_pair_rmsd(*load_stage_a(args.translation_dir))
    floor = 1e-3  # identical endpoints are drawn at the floor
    bins = np.logspace(-3, np.log10(3.0), 61)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for axis, pair in zip(axes, PAIRS):
        sym = np.clip(symmetric[pair], floor, None)
        tra = np.clip(translation[pair], floor, None)
        axis.hist(sym, bins=bins, color=SERIES[0], alpha=0.85, label="Symmetry-aware matching")
        axis.hist(tra, bins=bins, histtype="step", color=TEXT_2, linewidth=1.2, label="Translation-only matching")
        axis.set_xscale("log")
        axis.axvline(0.2, color=TEXT_2, linestyle=(0, (3, 3)), linewidth=1)
        far_sym, far_tra = np.mean(sym > 0.2), np.mean(tra > 0.2)
        axis.text(0.98, 0.95, f"> 0.2 A apart:\n{far_sym:.0%} (was {far_tra:.0%})", transform=axis.transAxes,
                  ha="right", va="top", fontsize=9, color=TEXT)
        axis.set_title(f"{SHORT[pair[0]]} vs {SHORT[pair[1]]}  (n={len(sym)})", fontsize=11)
        axis.set_xlabel("Endpoint RMSD from the same start (A, log)")
    axes[0].set_ylabel("Starts")
    axes[0].legend(loc="upper left", fontsize=8.5)
    figure.suptitle("Same start, different model: where do the endpoints land?", x=0.01, ha="left",
                    fontweight="bold", fontsize=12)
    figure.text(0.01, 0.01, "Each start relaxed to ML fmax 1e-3 eV/A by each model. "
                f"RMSD below {floor:g} A drawn at {floor:g} A. Dashed line: 0.2 A.", fontsize=8, color=TEXT_2)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = save(figure, args, "stagea_endpoint_distance.png")
    plt.close(figure)
    return out


def basin_counts(meta, matrices, threshold: float) -> dict[str, int]:
    """Distinct endpoints (complete linkage at ``threshold``) summed over configurations."""

    from relaxgen.analysis.analyze_umlip_endpoint_diversity import complete_linkage_labels

    totals: dict[str, int] = defaultdict(int)
    subsets = {**{m: (m,) for m in MODELS}, "all": MODELS}
    for config, rows in meta.items():
        matrix = matrices[config]
        models = [model for model, _, _ in rows]
        for name, subset in subsets.items():
            idx = [i for i, m in enumerate(models) if m in subset]
            if idx:
                totals[name] += len(set(complete_linkage_labels(matrix[np.ix_(idx, idx)], threshold).tolist()))
    return totals


def plot_stage_a_basins(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    meta, matrices = load_stage_a(args.symmetry_dir)
    thresholds = (0.1, 0.2, 0.3)
    counts = {t: basin_counts(meta, matrices, t) for t in thresholds}
    names = (*MODELS, "all")
    colors = (*SERIES, TEXT_2)
    labels = {**SHORT, "all": "All three pooled"}
    width = 0.19
    figure, axis = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(thresholds))
    for k, (name, color) in enumerate(zip(names, colors)):
        heights = [counts[t][name] for t in thresholds]
        bars = axis.bar(x + (k - 1.5) * (width + 0.015), heights, width, color=color, label=labels[name],
                        edgecolor=SURFACE, linewidth=1.5, zorder=2)
        for bar, height, t in zip(bars, heights, thresholds):
            text = f"{height}"
            if name == "all":
                text += f"\n+{counts[t]['all'] / counts[t]['eqv3'] - 1:.0%} vs EqV3"
            axis.annotate(text, (bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8.5,
                          color=TEXT if name == "all" else TEXT_2)
    axis.set_xticks(x)
    axis.set_xticklabels([f"endpoints merged within {t} A" for t in thresholds], color=TEXT)
    axis.set_ylabel("Distinct endpoints, summed over 188 configurations")
    axis.set_ylim(0, max(max(c.values()) for c in counts.values()) * 1.25)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right", ncols=4, fontsize=9, bbox_to_anchor=(1, 1.02))
    axis.set_title("Does pooling uMLIPs find more distinct structures?", pad=22)
    figure.text(0.01, 0.01, "2124 starts per model, ML fmax 1e-3 eV/A. Complete linkage on atom-matched RMSD, "
                "minimised over lattice symmetry operations.", fontsize=8, color=TEXT_2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    out = save(figure, args, "stagea_distinct_basins.png")
    plt.close(figure)
    return out


def energy_order_pairs(meta, matrices, distinct: float) -> dict[tuple[str, str], np.ndarray]:
    """For each model pair: (dE model A, dE model B) in meV over pairs of starts whose
    endpoints are more than ``distinct`` apart under both models."""

    from itertools import combinations

    out: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for config, rows in meta.items():
        matrix = matrices[config]
        pos = {(model, candidate): i for i, (model, candidate, _) in enumerate(rows)}
        energy = [e for _, _, e in rows]
        for a, b in PAIRS:
            common = sorted({c for m, c in pos if m == a} & {c for m, c in pos if m == b})
            for p, q in combinations(common, 2):
                ia, ja, ib, jb = pos[a, p], pos[a, q], pos[b, p], pos[b, q]
                if matrix[ia, ja] > distinct and matrix[ib, jb] > distinct:
                    out[a, b].append((1e3 * (energy[ia] - energy[ja]), 1e3 * (energy[ib] - energy[jb])))
    return {pair: np.asarray(v) for pair, v in out.items()}


def plot_stage_a_energy_order(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    meta, matrices = load_stage_a(args.symmetry_dir)
    pairs = energy_order_pairs(meta, matrices, args.distinct)
    band = args.band_mev
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for axis, pair in zip(axes, PAIRS):
        values = pairs.get(pair, np.zeros((0, 2)))
        lim = float(np.percentile(np.abs(values), 99)) * 1.1 if len(values) else 1.0
        # Disagreeing quadrants shaded.
        axis.fill_between([0, lim], -lim, 0, color=SERIES[1], alpha=0.07, linewidth=0, zorder=0)
        axis.fill_between([-lim, 0], 0, lim, color=SERIES[1], alpha=0.07, linewidth=0, zorder=0)
        axis.axhline(0, color=TEXT_2, linewidth=0.8)
        axis.axvline(0, color=TEXT_2, linewidth=0.8)
        axis.plot([-lim, lim], [-lim, lim], color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
        axis.scatter(values[:, 0], values[:, 1], s=9, color=SERIES[0], alpha=0.35, linewidths=0, zorder=2)
        clear = (np.abs(values[:, 0]) > band) & (np.abs(values[:, 1]) > band)
        agree = np.sign(values[clear, 0]) == np.sign(values[clear, 1])
        axis.text(0.03, 0.97, f"same order: {agree.mean():.0%}\n({clear.sum()} pairs with |dE| > {band:g} meV)",
                  transform=axis.transAxes, va="top", fontsize=9, color=TEXT, zorder=4,
                  bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2})
        axis.text(0.97, 0.03, "models disagree", transform=axis.transAxes, ha="right", fontsize=8.5,
                  color=SERIES[1])
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
        # Linear inside the +/- band, logarithmic beyond: gaps of a few meV and of
        # hundreds of meV are both visible.
        axis.set_xscale("symlog", linthresh=band)
        axis.set_yscale("symlog", linthresh=band)
        axis.set_aspect("equal")
        axis.set_xlabel(f"{SHORT[pair[0]]}: E(start 1) - E(start 2), meV")
        axis.set_ylabel(f"{SHORT[pair[1]]}: same difference, meV")
    figure.suptitle("Do the models agree which of two distinct structures is lower in energy?", x=0.01,
                    ha="left", fontweight="bold", fontsize=12)
    figure.text(0.01, 0.01, f"Pairs of starts in one configuration whose endpoints are > {args.distinct} A apart "
                f"(symmetry-aware) under both models. Axes linear within +/-{band:g} meV, logarithmic beyond; "
                "clipped at the 99th percentile of |dE|. Shaded: opposite order.", fontsize=8, color=TEXT_2)
    figure.tight_layout(rect=(0, 0.075, 1, 0.95))
    out = save(figure, args, "stagea_energy_order.png")
    plt.close(figure)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scf-time", "stagea-distance", "stagea-basins", "stagea-energy-order"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
        if name.startswith("stagea"):
            command.add_argument("--symmetry-dir", type=Path, default=STAGE_A_SYMMETRY)
            command.add_argument("--translation-dir", type=Path, default=STAGE_A_TRANSLATION)
        if name == "stagea-energy-order":
            command.add_argument("--distinct", type=float, default=0.2, help="A; endpoint pairs farther apart")
            command.add_argument("--band-mev", type=float, default=10.0, help="ordering counted outside this")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    {"scf-time": plot_scf_time, "stagea-distance": plot_stage_a_distance, "stagea-basins": plot_stage_a_basins,
     "stagea-energy-order": plot_stage_a_energy_order}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
