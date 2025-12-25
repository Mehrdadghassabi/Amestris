#!/usr/bin/env python3
"""
mt_score_csv.py

Compute reference-based COMET and BLEU scores for machine translation (MT) outputs
stored in a CSV file.

The script performs the following steps:

  1) Load an input CSV.
  2) Compute segment-level COMET and sentence-level BLEU for each row.
  3) Append a final summary row containing system-level scores.
  4) Write the updated CSV to disk (optionally overwriting the input file).

------------------------------------------------------------------------------
Input format
------------------------------------------------------------------------------
The CSV must contain (at minimum) three text columns:

  1) Source text column      (e.g., "en" or "du")
  2) Reference text column   (human translation; e.g., "du" or "en")
  3) MT output column        (system output; default name: "mt")

An optional identifier column (default name: "id") may also be present.

Examples (column naming is user-defined; set via command-line arguments):

  • English → German:
      columns: id, en, du, mt
      mapping: --src_col en --ref_col du --mt_col mt

  • German → English:
      columns: id, du, en, mt
      mapping: --src_col du --ref_col en --mt_col mt

A single implementation supports both directions: select the mapping at runtime
by specifying --src_col, --ref_col, and --mt_col.

------------------------------------------------------------------------------
Metrics and mapping
------------------------------------------------------------------------------
COMET (reference-based) requires, for each example:
    src = source sentence
    mt  = system output (hypothesis)
    ref = human reference translation

BLEU is computed using the same hypothesis/reference pairing:
    hypothesis = mt
    reference  = ref

The script adds per-segment columns:
    - comet_score
    - bleu_score

It also appends a final summary row containing system-level scores:
    - COMET system score returned by the COMET model
    - BLEU corpus score computed by SacreBLEU

If an identifier column exists, the summary row is labeled:
    id="__SYSTEM_AVG__"

If a previous run already appended that row, it is removed before scoring to
avoid double-counting.

------------------------------------------------------------------------------
Installation
------------------------------------------------------------------------------
Suggested dependencies (CPU-only is sufficient; GPU accelerates COMET if available):

    pip install --upgrade pip
    pip install pandas torch sacrebleu "unbabel-comet>=2.0.0"

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
English → German:
    python mt_score_csv.py --csv data.csv --src_col en --ref_col du --mt_col mt

German → English:
    python mt_score_csv.py --csv data.csv --src_col du --ref_col en --mt_col mt

By default, the input CSV is overwritten. To write to a new file:
    python mt_score_csv.py --csv data.csv --output_csv data_scored.csv

------------------------------------------------------------------------------
BLEU configuration
------------------------------------------------------------------------------
SacreBLEU is used with an explicit BLEU configuration for reproducibility:
  - tokenizer: "13a"
  - smoothing: "exp"
  - effective_order: True

Sentence-level BLEU uses smoothing; corpus BLEU is reported as SacreBLEU’s
corpus score.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint


DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"
SUMMARY_ID_VALUE = "__SYSTEM_AVG__"


# ------------------------------------------------------------------------------
# I/O and validation
# ------------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    """
    Ensure that all required columns are present in the DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    required:
        List of column names that must exist.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def _normalize_text_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Normalize text columns in-place by:
      - replacing NaN with an empty string
      - casting values to string

    This ensures consistent behavior across metric implementations and avoids
    failures caused by missing values.
    """
    for col in cols:
        df[col] = df[col].fillna("").astype(str)


def _drop_previous_summary_row(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Remove a previously appended summary row (if present).

    The summary row is identified by a sentinel identifier value
    (SUMMARY_ID_VALUE) in the specified id column.
    """
    if id_col in df.columns:
        before = len(df)
        df = df[df[id_col] != SUMMARY_ID_VALUE].reset_index(drop=True)
        after = len(df)
        if after != before:
            print(
                f"Removed previous summary row ({id_col}={SUMMARY_ID_VALUE!r}). "
                f"Rows: {before} → {after}"
            )
    return df


def _atomic_write_csv(df: pd.DataFrame, out_path: str, encoding: str = "utf-8") -> None:
    """
    Write a CSV atomically:

      1) Write to a temporary file in the destination directory.
      2) Replace the destination path with the temporary file.

    Atomic replacement reduces the risk of producing a partially written output
    if an error occurs during serialization.

    Notes
    -----
    On Windows, replacement may raise PermissionError when the destination file
    is open in another program (e.g., spreadsheet software).
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="._tmp_scored_", suffix=".csv", dir=out_dir)
    os.close(fd)

    try:
        df.to_csv(tmp_path, index=False, encoding=encoding)
        os.replace(tmp_path, out_path)
    except PermissionError as e:
        raise PermissionError(
            f"Could not write/replace output CSV at: {out_path}\n"
            "If the file is open in another program, close it and re-run."
        ) from e
    finally:
        # Best-effort cleanup if replacement did not occur.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ------------------------------------------------------------------------------
# COMET
# ------------------------------------------------------------------------------

def load_comet_model(model_name: str = DEFAULT_COMET_MODEL) -> Any:
    """
    Download and load a COMET model checkpoint.

    The default model is reference-based and requires (src, mt, ref).
    """
    print(f"Downloading/loading COMET model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    model.eval()
    return model


def build_comet_samples(
    df: pd.DataFrame,
    src_col: str,
    mt_col: str,
    ref_col: str,
) -> List[Dict[str, str]]:
    """
    Build COMET input samples from the DataFrame.

    Each row is converted into:
        {"src": <source>, "mt": <system output>, "ref": <reference>}
    """
    _require_columns(df, [src_col, mt_col, ref_col])
    _normalize_text_columns(df, [src_col, mt_col, ref_col])

    samples: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        samples.append(
            {
                "src": row[src_col],
                "mt": row[mt_col],
                "ref": row[ref_col],
            }
        )
    return samples


def run_comet(
    model: Any,
    samples: List[Dict[str, str]],
    batch_size: int,
    gpus: int,
) -> Tuple[List[float], float]:
    """
    Compute segment-level and system-level COMET scores.

    Returns
    -------
    segment_scores:
        List of per-segment COMET scores aligned with the input rows.
    system_score:
        System-level COMET score produced by the model.

    Notes
    -----
    COMET >= 2.0 typically returns a Prediction object with attributes:
      - scores
      - system_score
    """
    print("Computing COMET scores...")
    output = model.predict(samples, batch_size=batch_size, gpus=gpus)

    if hasattr(output, "scores") and hasattr(output, "system_score"):
        segment_scores = [float(s) for s in output.scores]
        system_score = float(output.system_score)
        return segment_scores, system_score

    if isinstance(output, dict):
        if "scores" in output and "system_score" in output:
            return [float(s) for s in output["scores"]], float(output["system_score"])

    raise RuntimeError(
        "Unexpected COMET prediction output. "
        f"type={type(output)}; available={dir(output)}"
    )


# ------------------------------------------------------------------------------
# BLEU (SacreBLEU)
# ------------------------------------------------------------------------------

def compute_bleu_scores(
    df: pd.DataFrame,
    mt_col: str,
    ref_col: str,
) -> Tuple[List[float], float]:
    """
    Compute sentence-level and corpus-level BLEU using SacreBLEU.

    Parameters
    ----------
    df:
        Input DataFrame.
    mt_col:
        Column name containing system outputs (hypotheses).
    ref_col:
        Column name containing human references.

    Returns
    -------
    segment_bleu_scores:
        Sentence-level BLEU scores for each row.
    corpus_bleu_score:
        Corpus-level BLEU score computed over the full dataset.

    Configuration
    -------------
    - tokenize="13a"
    - smooth_method="exp"
    - effective_order=True
    """
    print("Computing BLEU scores (sentence-level + corpus-level)...")
    _require_columns(df, [mt_col, ref_col])
    _normalize_text_columns(df, [mt_col, ref_col])

    bleu = BLEU(tokenize="13a", smooth_method="exp", effective_order=True)

    mt_list: List[str] = df[mt_col].tolist()
    ref_list: List[str] = df[ref_col].tolist()

    seg_scores: List[float] = []
    for hyp, ref in zip(mt_list, ref_list):
        seg_scores.append(float(bleu.sentence_score(hyp, [ref]).score))

    corpus_score = float(bleu.corpus_score(mt_list, [ref_list]).score)
    return seg_scores, corpus_score


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def _auto_gpu_setting(requested_gpus: int) -> int:
    """
    Resolve the GPU setting for COMET.

    Parameters
    ----------
    requested_gpus:
        -1  : auto-detect (use 1 GPU if CUDA is available; otherwise CPU)
         0  : force CPU
        >=1 : request that many GPUs (capped by device availability)

    Returns
    -------
    int
        Number of GPUs to pass to COMET.
    """
    cuda = torch.cuda.is_available()

    if requested_gpus >= 0:
        if not cuda and requested_gpus > 0:
            print("WARNING: CUDA not available; forcing gpus=0 for COMET.")
            return 0
        if cuda:
            return min(requested_gpus, torch.cuda.device_count())
        return 0

    # Auto mode
    return 1 if cuda else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute COMET and BLEU scores for an MT CSV and append a system summary row."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path for the scored CSV. If omitted, the input file is overwritten.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Output CSV encoding (default: utf-8).",
    )

    # Column mapping
    parser.add_argument("--src_col", default="en", help="Source text column name.")
    parser.add_argument("--ref_col", default="du", help="Reference text column name.")
    parser.add_argument("--mt_col", default="mt", help="MT output column name.")
    parser.add_argument(
        "--id_col",
        default="id",
        help=(
            "Identifier column name (optional). If absent, the summary row is appended "
            "without an identifier label."
        ),
    )

    # COMET settings
    parser.add_argument(
        "--comet_model",
        default=DEFAULT_COMET_MODEL,
        help=f"COMET model name or path (default: {DEFAULT_COMET_MODEL}).",
    )
    parser.add_argument(
        "--comet_batch_size",
        type=int,
        default=16,
        help="Batch size for COMET prediction.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="GPUs for COMET: -1=auto, 0=CPU, 1..N=request that many GPUs (capped by availability).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = os.path.abspath(args.csv)
    out_path = os.path.abspath(args.output_csv) if args.output_csv else in_path

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    print(f"Input CSV : {in_path}")
    print(f"Output CSV: {out_path}")
    print(f"Mapping   : src={args.src_col!r}, ref={args.ref_col!r}, mt={args.mt_col!r}")
    print("-" * 80)

    df = pd.read_csv(in_path)
    print("Loaded columns:", list(df.columns))

    # Remove any previously appended summary row
    df = _drop_previous_summary_row(df, id_col=args.id_col)

    # Resolve GPU usage for COMET
    gpus = _auto_gpu_setting(args.gpus)
    if gpus > 0:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: using gpus={gpus}. Device 0: {device_name}")
    else:
        print("CUDA not used: COMET will run on CPU (gpus=0).")
    print("-" * 80)

    # Build COMET samples (normalizes src/ref/mt columns)
    samples = build_comet_samples(
        df,
        src_col=args.src_col,
        mt_col=args.mt_col,
        ref_col=args.ref_col,
    )

    # Run COMET
    model = load_comet_model(args.comet_model)
    comet_seg, comet_sys = run_comet(
        model=model,
        samples=samples,
        batch_size=args.comet_batch_size,
        gpus=gpus,
    )
    if len(comet_seg) != len(df):
        raise RuntimeError(
            f"COMET produced {len(comet_seg)} segment scores for {len(df)} rows."
        )

    # Run BLEU (hypothesis=mt, reference=ref)
    bleu_seg, bleu_sys = compute_bleu_scores(
        df,
        mt_col=args.mt_col,
        ref_col=args.ref_col,
    )
    if len(bleu_seg) != len(df):
        raise RuntimeError(
            f"BLEU produced {len(bleu_seg)} segment scores for {len(df)} rows."
        )

    # Attach segment-level scores
    df["comet_score"] = comet_seg
    df["bleu_score"] = bleu_seg

    # Append summary row
    summary_row = {col: None for col in df.columns}
    if args.id_col in df.columns:
        summary_row[args.id_col] = SUMMARY_ID_VALUE
    summary_row["comet_score"] = comet_sys
    summary_row["bleu_score"] = bleu_sys
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # Write output
    _atomic_write_csv(df, out_path, encoding=args.encoding)

    print("-" * 80)
    print("Completed scoring. Added columns:")
    print("  - comet_score")
    print("  - bleu_score")
    if args.id_col in df.columns:
        print(f"Appended summary row with {args.id_col}={SUMMARY_ID_VALUE!r}.")
    else:
        print("Appended summary row (identifier column not present).")
    print("\nSystem-level scores:")
    print(f"  COMET : {comet_sys:.4f}")
    print(f"  BLEU  : {bleu_sys:.4f}")


if __name__ == "__main__":
    main()
