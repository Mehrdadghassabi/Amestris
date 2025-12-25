#!/usr/bin/env python3
"""
Select low-quality MT outputs using COMET scores (bottom 20% by default).

This script creates a subset of *low-scoring* machine translation (MT) examples from
a CSV file that contains a COMET score column. The selected subset is intended to
represent *inadequate / rejected* translations for preference-based training
(preferred–rejected reinforcement learning style pipelines).

What this script does (exact behavior)
--------------------------------------
1) Load the input CSV.
2) Convert the specified score column (default: "comet_score") to numeric
   (non-numeric values become NaN).
3) Compute two thresholds on the numeric score distribution:
   - A knee/elbow score (distance-to-line on the sorted scores) as a *diagnostic
     reference*.
   - A quantile threshold q_thr based on the desired bottom fraction to keep:
         q_thr = quantile(scores, keep_fraction)
     For keep_fraction=0.20, q_thr is the 20th percentile.
4) Select rows using the quantile threshold (this is the actual selection rule):
   - Keep rows with numeric score <= q_thr (the bottom keep_fraction).
   - Optionally keep or drop NaN-score rows:
       * default: keep NaNs
       * --drop-nans: drop NaNs
5) Optionally create a timestamped backup of the original CSV.
6) Write a new CSV containing only the selected rows.
7) Optionally display a histogram and ECDF annotated with both the knee reference
   and the quantile threshold used for selection.

Important note about “knee” vs “bottom 20%”
-------------------------------------------
The knee threshold is computed and reported to help interpret the distribution and to
support visual inspection (histogram/ECDF). However, the script’s selection criterion
is strictly the *bottom keep_fraction quantile*.

In our experiments, the bottom 20% quantile threshold happened to be close to the knee
reference (and not substantially higher), which aligned with the observed score
distribution. This is a dataset-dependent observation and is not enforced as a rule;
the script always selects by quantile.

Outputs
-------
Given an input file: /path/data.csv

The script writes:
- Backup (optional, timestamped):
    data_BACKUP_YYYYMMDD_HHMMSS.csv
- Selected bottom subset:
    data_bottom_<keep_fraction>_thr_<q_thr>.csv

Usage examples
--------------
Keep bottom 20% (default), keep NaNs (default), show plots (default):
  python select_bottom_comet.py --input /path/data.csv --column comet_score

Keep bottom 10%:
  python select_bottom_comet.py --input /path/data.csv --keep-fraction 0.10

Drop rows with non-numeric/NaN scores:
  python select_bottom_comet.py --input /path/data.csv --drop-nans

Disable plots (useful on servers):
  python select_bottom_comet.py --input /path/data.csv --no-plots
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# Threshold estimation
# ======================================================================

def compute_knee_threshold(scores: np.ndarray) -> Tuple[float, int]:
    """
    Compute a knee/elbow score threshold (diagnostic reference) using a simple
    distance-to-line method on the sorted score distribution.

    Steps:
      1) Sort scores ascending.
      2) Normalize scores to [0, 1] for numerical stability.
      3) Draw a straight line between the first and last point.
      4) Compute each point’s absolute distance to that line.
      5) The knee is the point with maximum distance.

    Parameters
    ----------
    scores:
        1D NumPy array of numeric scores (finite, non-NaN).

    Returns
    -------
    knee_thr:
        Score value at the knee (in original score space).
    knee_idx:
        Index of the knee in the sorted array (0-based).
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")

    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("scores array is empty.")
    if n == 1:
        return float(sorted_scores[0]), 0

    # Normalized index axis in [0, 1]
    x = np.linspace(0.0, 1.0, n)

    # Normalize y-values in [0, 1]
    s_min = float(sorted_scores.min())
    s_max = float(sorted_scores.max())
    denom = (s_max - s_min) + 1e-12  # avoids division by zero if all scores identical
    y = (sorted_scores - s_min) / denom

    # Line connecting the endpoints
    y0, y1 = float(y[0]), float(y[-1])
    line = y0 + (y1 - y0) * x

    # Knee: maximum distance to the line
    dist = np.abs(y - line)
    knee_idx = int(dist.argmax())
    knee_thr = float(sorted_scores[knee_idx])
    return knee_thr, knee_idx


# ======================================================================
# Plotting helpers
# ======================================================================

def compute_fd_bins(values: np.ndarray, min_bins: int = 10, default_bins: int = 50) -> int:
    """
    Compute histogram bin count using the Freedman–Diaconis rule.

    This rule adapts bin width to the scale/spread of the data. If the rule cannot
    be applied robustly (e.g., zero IQR), a default bin count is used.

    Parameters
    ----------
    values:
        1D NumPy array of numeric values.
    min_bins:
        Minimum number of bins.
    default_bins:
        Fallback number of bins if the FD rule is not usable.

    Returns
    -------
    bins:
        Number of bins for a histogram.
    """
    if values.size == 0:
        return default_bins

    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return default_bins

    bin_width = 2 * iqr * (values.size ** (-1.0 / 3.0))
    if bin_width <= 0:
        return default_bins

    bins = int(np.ceil((values.max() - values.min()) / bin_width))
    return max(min_bins, bins)


def plot_histogram(values: np.ndarray, knee_thr: float, q_thr: float, column: str) -> None:
    """
    Plot a histogram of scores with:
      - knee threshold (reference)
      - quantile threshold (used for selection)
    """
    bins = compute_fd_bins(values)
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.axvline(knee_thr, linewidth=2, label=f"knee (ref) = {knee_thr:.4f}")
    plt.axvline(q_thr, linewidth=2, label=f"quantile (used) = {q_thr:.4f}")
    plt.title(f"Distribution of {column} (Histogram)")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ecdf(values: np.ndarray, knee_thr: float, q_thr: float, column: str) -> None:
    """
    Plot an empirical CDF of scores with:
      - knee threshold (reference)
      - quantile threshold (used for selection)
    """
    sorted_scores = np.sort(values)
    n = sorted_scores.size
    ecdf = np.arange(1, n + 1) / n

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_scores, ecdf)
    plt.axvline(knee_thr, linewidth=2, label=f"knee (ref) = {knee_thr:.4f}")
    plt.axvline(q_thr, linewidth=2, label=f"quantile (used) = {q_thr:.4f}")
    plt.title(f"ECDF of {column}")
    plt.xlabel(column)
    plt.ylabel("Fraction ≤ score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================
# File utilities
# ======================================================================

def create_timestamped_backup(csv_path: Path) -> Path:
    """
    Create a timestamped backup of the input CSV in the same directory.

    Backup name format:
      <stem>_BACKUP_<YYYYMMDD_HHMMSS><suffix>
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{csv_path.stem}_BACKUP_{timestamp}{csv_path.suffix}"
    backup_path = csv_path.with_name(backup_name)
    shutil.copy2(csv_path, backup_path)
    return backup_path


# ======================================================================
# Main processing
# ======================================================================

def process_csv(
    csv_path: Path,
    column: str,
    keep_fraction: float = 0.20,
    keep_nans: bool = True,
    create_backup: bool = True,
    show_plots: bool = True,
) -> None:
    """
    Select the bottom keep_fraction of rows based on a COMET score column.

    Exact selection rule
    --------------------
    Let q_thr be the keep_fraction quantile of numeric scores.
      q_thr = quantile(valid_numeric_scores, keep_fraction)

    Then:
      - Keep rows with numeric score <= q_thr.
      - If keep_nans=True (default), also keep rows whose score is NaN after coercion.
      - If keep_nans=False, NaN-score rows are not included.

    The knee threshold is computed and printed as a diagnostic reference only.

    Parameters
    ----------
    csv_path:
        Path to the input CSV file.
    column:
        Name of the COMET score column (or similar scalar quality metric).
    keep_fraction:
        Fraction in (0, 1] specifying how much of the *lowest* scoring data to keep.
        Example: 0.20 keeps the bottom 20%.
    keep_nans:
        If True, retain rows with non-numeric/NaN scores (after coercion).
    create_backup:
        If True, create a timestamped backup of the original CSV.
    show_plots:
        If True, display histogram and ECDF with knee and quantile thresholds.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("--keep-fraction must be in (0, 1].")

    df = pd.read_csv(csv_path, low_memory=False)

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert to numeric; invalid entries become NaN.
    scores = pd.to_numeric(df[column], errors="coerce")
    valid_scores = scores.dropna().astype(float)

    if valid_scores.empty:
        raise ValueError(f"No numeric values found in '{column}' after coercion.")

    # --------------------------------------------------------------
    # 1) Knee (diagnostic reference)
    # --------------------------------------------------------------
    knee_thr, knee_idx = compute_knee_threshold(valid_scores.values)

    # --------------------------------------------------------------
    # 2) Quantile threshold (USED for selection)
    # --------------------------------------------------------------
    q_thr = float(np.quantile(valid_scores.values, keep_fraction))

    print("\n=== Thresholds ===")
    print(f"knee_thr (reference)        = {knee_thr:.6f} (sorted index: {knee_idx})")
    print(f"keep_fraction (bottom kept) = {keep_fraction:.2%}")
    print(f"quantile_thr (USED)         = {q_thr:.6f}")

    # --------------------------------------------------------------
    # 3) Backup original file (optional)
    # --------------------------------------------------------------
    backup_path = None
    if create_backup:
        backup_path = create_timestamped_backup(csv_path)
        print("\n=== Backup created ===")
        print(backup_path)

    # --------------------------------------------------------------
    # 4) Build selection mask (bottom keep_fraction)
    # --------------------------------------------------------------
    if keep_nans:
        mask_keep = scores.isna() | (scores <= q_thr)
    else:
        mask_keep = scores.notna() & (scores <= q_thr)

    kept = int(mask_keep.sum())
    removed = int((~mask_keep).sum())
    df_selected = df.loc[mask_keep].copy()

    print("\n=== Selection result (BOTTOM SUBSET) ===")
    print(f"Kept rows:    {kept} ({kept / len(df):.2%})")
    print(f"Removed rows: {removed} ({removed / len(df):.2%})")
    print(f"NaN rows kept: {'yes' if keep_nans else 'no'}")

    # --------------------------------------------------------------
    # 5) Save selected subset
    # --------------------------------------------------------------
    out_name = f"{csv_path.stem}_bottom_{keep_fraction:.2f}_thr_{q_thr:.4f}{csv_path.suffix}"
    out_path = csv_path.with_name(out_name)
    df_selected.to_csv(out_path, index=False)

    print("\n=== Saved selected CSV (bottom subset) ===")
    print(out_path)

    # --------------------------------------------------------------
    # 6) Optional plots
    # --------------------------------------------------------------
    if show_plots:
        plot_values = valid_scores.values
        plot_histogram(plot_values, knee_thr, q_thr, column)
        plot_ecdf(plot_values, knee_thr, q_thr, column)

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"Input CSV:                  {csv_path}")
    if backup_path is not None:
        print(f"Backup CSV:                 {backup_path}")
    print(f"Selected (bottom) CSV:      {out_path}")
    print(f"Score column:               {column}")
    print(f"Knee threshold (reference): {knee_thr:.6f}")
    print(f"Quantile threshold (used):  {q_thr:.6f}")
    print(f"Bottom fraction kept:       {keep_fraction:.2%}")
    print(f"Rows kept:                  {kept}")
    print(f"Rows removed:               {removed}")


# ======================================================================
# Command-line interface
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for selecting the bottom portion of COMET scores.
    """
    parser = argparse.ArgumentParser(
        prog="select_bottom_comet.py",
        description=(
            "Select the lowest-scoring fraction of a COMET score distribution from a CSV. "
            "Computes a knee threshold as a diagnostic reference and uses a quantile "
            "threshold to keep the bottom portion (default: 20%)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--column",
        type=str,
        default="comet_score",
        help="Name of the column containing COMET (or similar) scores.",
    )

    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.20,
        help="Fraction in (0, 1] specifying how much of the *lowest* scoring data to keep.",
    )

    parser.add_argument(
        "--drop-nans",
        action="store_true",
        help="If set, exclude rows where the score is non-numeric/NaN after coercion.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup of the original CSV.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable histogram and ECDF plots.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path = Path(args.input).expanduser().resolve()

    process_csv(
        csv_path=csv_path,
        column=args.column,
        keep_fraction=args.keep_fraction,
        keep_nans=not args.drop_nans,
        create_backup=not args.no_backup,
        show_plots=not args.no_plots,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
