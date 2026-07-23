"""
Builds ONE aggregate figure covering a whole grid run's results, from the
per-grid-point JSON summaries already saved by scripts/exp1_synchronization.py
(also used by Experiment 3) or scripts/exp2_chaos_test.py -- DESIGN.md.

Replaces the old default of one detailed PNG per grid point (20-40 per
experiment), which stopped being useful at that count: the actual findings
from this session's overnight run only became visible after manually
pulling every JSON's numbers into a table. This script does that
aggregation directly and saves it as a figure, reusable for any grid
(overnight run, exp3_grid, or a future sweep) without re-running anything --
it only reads already-saved JSON.

Auto-detects which experiment's schema a directory holds from the first
*_summary.json field names found (no --experiment flag needed):
  Experiment 1/3 (exp1_synchronization.py): has "rate_hz" and "gj_label" --
    4-panel figure (Golgi ISI CV, Purkinje rate, dominant Golgi frequency,
    mean Golgi pairwise correlation), each vs. rate_hz, GJ off/on as two
    series, mean +/- std across seeds.
  Experiment 2 (exp2_chaos_test.py): has "g_gap_nS" and "perturbation_mv" --
    2-panel figure (log-D growth rate for golgi/granule/purkinje, Golgi
    spectrum peak/median), each vs. g_gap_nS, mean +/- std across seeds.

Usage:
    python scripts/summarize_grid_results.py outputs/overnight/exp1
    python scripts/summarize_grid_results.py outputs/overnight/exp2
    python scripts/summarize_grid_results.py outputs/exp3_grid
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_summaries(output_dir: Path) -> list[dict]:
    paths = sorted(glob.glob(str(output_dir / "*_summary.json")))
    if not paths:
        raise FileNotFoundError(f"No *_summary.json files found in {output_dir}")
    summaries = []
    for p in paths:
        with open(p) as f:
            summaries.append(json.load(f))
    return summaries


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def summarize_exp1_or_exp3(summaries: list[dict], output_dir: Path) -> Path:
    by_rate_gj = defaultdict(list)
    for s in summaries:
        by_rate_gj[(s["rate_hz"], s["gj_label"])].append(s)
    rates = sorted(set(s["rate_hz"] for s in summaries))

    metrics = [
        ("golgi_isi_cv", "Golgi ISI CV"),
        ("purkinje_rate_hz", "Purkinje rate (Hz)"),
        ("dominant_golgi_freq_hz", "Dominant Golgi frequency (Hz)"),
        ("golgi_mean_pairwise_corr", "Mean Golgi pairwise correlation"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f"Experiment 1/3 grid summary -- {output_dir} ({len(summaries)} runs, "
                 f"{len(rates)} rates x {len(set(s['seed'] for s in summaries))} seeds)")

    for ax, (field, label) in zip(axes, metrics):
        for gj_label, color in (("off", "tab:blue"), ("on", "tab:orange")):
            means, stds = [], []
            for rate in rates:
                rows = by_rate_gj.get((rate, gj_label), [])
                m, s = _mean_std([r[field] for r in rows])
                means.append(m)
                stds.append(s)
            ax.errorbar(rates, means, yerr=stds, marker="o", label=f"GJ {gj_label}", color=color)
        ax.set_xlabel("mossy-fiber rate (Hz)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = output_dir / "aggregate_summary.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def summarize_exp2(summaries: list[dict], output_dir: Path) -> Path:
    by_ggap = defaultdict(list)
    for s in summaries:
        by_ggap[s["g_gap_nS"]].append(s)
    ggaps = sorted(by_ggap.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Experiment 2 grid summary -- {output_dir} ({len(summaries)} runs, "
                 f"{len(ggaps)} g_gap values x {len(set(s['seed'] for s in summaries))} seeds)")

    for field, label, color in (
        ("log_D_slope_per_ms_golgi", "golgi", "tab:blue"),
        ("log_D_slope_per_ms_granule", "granule", "tab:green"),
        ("log_D_slope_per_ms_purkinje", "purkinje", "tab:red"),
    ):
        means, stds = [], []
        for g in ggaps:
            m, s = _mean_std([r[field] for r in by_ggap[g]])
            means.append(m)
            stds.append(s)
        axes[0].errorbar(ggaps, means, yerr=stds, marker="o", label=label, color=color)
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("g_gap_nS")
    axes[0].set_ylabel("log-D growth rate (1/ms)")
    axes[0].set_title("Divergence growth rate")
    axes[0].legend(fontsize=8)

    means, stds = [], []
    for g in ggaps:
        m, s = _mean_std([r["golgi_spectrum_peak_to_median"] for r in by_ggap[g]])
        means.append(m)
        stds.append(s)
    axes[1].errorbar(ggaps, means, yerr=stds, marker="o", color="tab:purple")
    axes[1].set_xlabel("g_gap_nS")
    axes[1].set_ylabel("Golgi spectrum peak/median")
    axes[1].set_title("Spectrum peak/median (lower = more broadband)")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig_path = output_dir / "aggregate_summary.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=str, help="Directory containing *_summary.json files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summaries = _load_summaries(output_dir)
    print(f"Loaded {len(summaries)} summary files from {output_dir}")

    first = summaries[0]
    if "rate_hz" in first and "gj_label" in first:
        fig_path = summarize_exp1_or_exp3(summaries, output_dir)
    elif "g_gap_nS" in first and "perturbation_mv" in first:
        fig_path = summarize_exp2(summaries, output_dir)
    else:
        raise ValueError(
            f"Could not recognize the schema of {output_dir}'s summary JSONs -- "
            f"expected Experiment 1/3 fields (rate_hz, gj_label) or Experiment 2 "
            f"fields (g_gap_nS, perturbation_mv). Found keys: {sorted(first.keys())}"
        )

    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
