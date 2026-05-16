"""
Calculate COMET, BLEU and BLEURT scores for a CSV with columns: id, fr, en, answer.

- 'answer' is treated as the system translation (MT output).
- The src / mt / ref mapping used for COMET is:
      src = df["fr"]
      mt  = df["answer"]
      ref = df["en"]

  BLEU and BLEURT are computed using the same (mt, ref) pairing as COMET:
      hypothesis (mt) = df["answer"]
      reference (ref) = df["en"]

The script:
  * Adds per-segment scores:
        - 'comet_score'
        - 'bleu_score'
        - 'bleurt_score'
  * Appends a final summary row with the system-level average of each metric.
    This row is identified by id="__SYSTEM_AVG__" (if an 'id' column exists).

NOTE ABOUT BLEURT:
  - True BLEURT requires the 'bleurt' package plus TensorFlow, which currently
    has no wheels for Python 3.12.
  - This script tries to load BLEURT via `evaluate.load("bleurt")`.
  - If that fails (e.g., on Python 3.12), it will:
        * print a warning,
        * fill 'bleurt_score' with NaN,
        * still compute COMET and BLEU normally.
  - To get REAL BLEURT numbers, run this script in a Python 3.8–3.10 environment
    with `pip install bleurt`.

Before running (inside your virtualenv), install dependencies, for example:

    pip install --upgrade pip
    pip install "unbabel-comet>=2.0.0" pandas torch sacrebleu evaluate

(And, in a compatible Python version, also `pip install bleurt`.)
"""

import os
from typing import Any, List, Dict, Tuple

import pandas as pd
import torch
import sacrebleu
import evaluate
from comet import download_model, load_from_checkpoint


# ---------- Configuration ----------

# Path to your CSV file (this file will be OVERWRITTEN)
CSV_PATH = r"C:\Users\HAMAHANG\Desktop\1B\Train\en_to_de\wmt14_en_de_mt_FULL_cleaned.csv"

# COMET model to use (reference-based default)
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"

# Batch size for COMET predict (tune if you get OOM errors)
BATCH_SIZE = 24

# Batch size for BLEURT metric (batches are handled in this script)
BLEURT_BATCH_SIZE = 24


def load_comet_model(model_name: str = COMET_MODEL_NAME) -> Any:
    """
    Download and load the COMET model from Hugging Face Hub.
    """
    print(f"Downloading/loading COMET model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    model.eval()
    return model


def prepare_comet_data(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Build the list of dicts expected by COMET from the DataFrame.
    Ensures values are strings and NaNs are handled.

    NOTE: This function also normalizes the 'fr', 'en', and 'answer' columns
    (fillna + astype(str)), which are then reused by BLEU and BLEURT.

    Current src/mt/ref mapping used for COMET (and mirrored by BLEU/BLEURT):
        src = fr
        mt  = answer
        ref = en
    """
    print("Preparing data for COMET...")
    data: List[Dict[str, str]] = []

    # Ensure required columns exist
    required_cols = {"fr", "en", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Coerce to string and replace NaN with empty string
    for col in ["fr", "en", "answer"]:
        df[col] = df[col].fillna("").astype(str)

    for _, row in df.iterrows():
        sample = {
            "src": row["en"],
            "mt": row["answer"],
            "ref": row["fr"],
        }
        data.append(sample)

    print(f"Number of examples: {len(data)}")
    return data


def run_comet(
    model: Any,
    data: List[Dict[str, str]],
    batch_size: int,
    gpus: int,
) -> Tuple[List[float], float]:
    """
    Run COMET on the provided data and return (segment_scores, system_score).
    Handles COMET>=2.0 Prediction output and provides clear errors otherwise.
    """
    print("Computing COMET scores (this may take a while)...")
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    # COMET >= 2.0 returns a Prediction object with `.scores` and `.system_score`.
    if hasattr(output, "scores") and hasattr(output, "system_score"):
        segment_scores = [float(s) for s in output.scores]
        system_score = float(output.system_score)
        return segment_scores, system_score

    # Fallback for dict-like outputs
    if isinstance(output, dict):
        try:
            segment_scores = [float(s) for s in output["scores"]]
            system_score = float(output["system_score"])
            return segment_scores, system_score
        except Exception as e:
            raise RuntimeError(
                f"Unexpected COMET dict output format: keys={list(output.keys())}"
            ) from e

    raise RuntimeError(
        f"Unexpected COMET output type: {type(output)}; "
        f"available attributes: {dir(output)}"
    )


def compute_bleu_scores(df: pd.DataFrame) -> Tuple[List[float], float]:
    """
    Compute sentence-level BLEU and corpus-level BLEU using sacrebleu.

    BLEU uses the same (mt, ref) pairing as COMET:
        hypothesis (mt) = df["answer"]
        reference (ref) = df["en"]
    """
    print("Computing BLEU scores (sentence-level + corpus-level)...")

    # These columns were normalized to strings in prepare_comet_data
    mt_list: List[str] = df["answer"].tolist()
    ref_list: List[str] = df["en"].tolist()

    # Sentence-level BLEU
    bleu_segment_scores: List[float] = []
    for hyp, ref in zip(mt_list, ref_list):
        # sacrebleu.sentence_bleu returns an object with a .score attribute
        score_obj = sacrebleu.sentence_bleu(hyp, [ref])
        bleu_segment_scores.append(float(score_obj.score))

    # Corpus-level BLEU
    corpus_obj = sacrebleu.corpus_bleu(mt_list, [ref_list])
    bleu_system_score = float(corpus_obj.score)

    return bleu_segment_scores, bleu_system_score


def compute_bleurt_scores(
    df: pd.DataFrame,
    batch_size: int = BLEURT_BATCH_SIZE,
) -> Tuple[List[float], float]:
    """
    Compute sentence-level BLEURT scores and their mean using Hugging Face 'evaluate'.

    BLEURT uses the same (mt, ref) pairing as COMET:
        prediction (mt) = df["answer"]
        reference  (ref) = df["en"]

    If BLEURT cannot be loaded (e.g. due to Python 3.12 / missing 'bleurt' pkg),
    this function will return NaN scores and print a clear warning – COMET and
    BLEU will still be computed normally.
    """
    print("Computing BLEURT scores (this may take a while, especially on CPU)...")

    # Try to load BLEURT metric; this may fail in unsupported environments
    try:
        bleurt_metric = evaluate.load("bleurt")
    except Exception as e:
        print("\nWARNING: Could not load BLEURT metric via `evaluate.load('bleurt')`.")
        print(f"  Reason: {e}")
        print("  BLEURT scores will be set to NaN.")
        print("  To obtain REAL BLEURT scores, run this script in Python 3.8–3.10")
        print("  and install `bleurt` (plus TensorFlow) in that environment.\n")

        nan_scores = [float("nan")] * len(df)
        return nan_scores, float("nan")

    predictions: List[str] = df["answer"].tolist()
    references: List[str] = df["en"].tolist()

    bleurt_scores: List[float] = []

    # Manual batching to keep memory usage under control
    for start in range(0, len(predictions), batch_size):
        end = start + batch_size
        batch_preds = predictions[start:end]
        batch_refs = references[start:end]

        result = bleurt_metric.compute(
            predictions=batch_preds,
            references=batch_refs,
        )
        batch_scores = [float(s) for s in result["scores"]]
        bleurt_scores.extend(batch_scores)

    if len(bleurt_scores) != len(df):
        raise RuntimeError(
            f"Mismatch between number of rows ({len(df)}) and "
            f"number of BLEURT scores ({len(bleurt_scores)})."
        )

    # System-level BLEURT = mean over all sentence-level scores
    bleurt_system_score = (
        float(sum(bleurt_scores) / len(bleurt_scores)) if bleurt_scores else float("nan")
    )

    return bleurt_scores, bleurt_system_score


def main() -> None:
    # ---------- Resolve and show the exact path we will use ----------
    abs_path = os.path.abspath(CSV_PATH)
    print(f"Configured CSV_PATH: {CSV_PATH}")
    print(f"Absolute path used: {abs_path}")
    print("-" * 80)

    # ---------- Load CSV ----------
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"CSV file not found at: {abs_path}")

    print(f"Loading CSV from: {abs_path}")
    df = pd.read_csv(abs_path)
    print("Columns on load:", list(df.columns))
    print("First 3 rows on load:")
    print(df.head(3))
    print("-" * 80)

    # ---------- QUICK WRITE TEST (before metric computation) ----------
    # This checks that we can actually overwrite this file.
    print("Running write test: adding temporary '__write_test__' column...")
    df["__write_test__"] = "test"
    try:
        df.to_csv(abs_path, index=False, encoding="utf-8")
    except PermissionError as e:
        print("\nERROR: Could not write to the CSV file during write test.")
        print("Most likely the file is OPEN in Excel or another program.")
        print("Close the file and run this script again.")
        raise e

    # Reload and verify the temp column is really there
    df_reload = pd.read_csv(abs_path)
    print("Columns after write test reload:", list(df_reload.columns))
    if "__write_test__" not in df_reload.columns:
        raise RuntimeError(
            "Write test failed: '__write_test__' column not found after saving.\n"
            "This means the script is NOT actually writing to the file you think, "
            "or something is reverting the file."
        )

    # Drop the temporary column and continue
    df = df_reload.drop(columns=["__write_test__"])

    # If there is already a previous summary row from an earlier run,
    # drop it so we don't double-count it in the metrics.
    if "id" in df.columns:
        original_len = len(df)
        df = df[df["id"] != "__SYSTEM_AVG__"].reset_index(drop=True)
        if len(df) != original_len:
            print(
                f"Detected and removed a previous summary row "
                f"(id='__SYSTEM_AVG__'). Rows went from {original_len} to {len(df)}."
            )

    print("Write test succeeded. Continuing to metric computation.")
    print("-" * 80)

    # ---------- GPU / CPU selection (for COMET) ----------
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpus = min(1, torch.cuda.device_count())
    else:
        gpus = 0

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU(s)={gpus}. Device 0: {device_name}")
    else:
        print("CUDA is NOT available. COMET will run on CPU only.")
    print("-" * 80)

    # ---------- Load COMET model ----------
    model = load_comet_model()

    # ---------- Build COMET input data (also normalizes text columns) ----------
    data = prepare_comet_data(df)

    # ---------- Run COMET once over the full dataset ----------
    comet_segment_scores, comet_system_score = run_comet(
        model=model,
        data=data,
        batch_size=BATCH_SIZE,
        gpus=gpus,
    )

    if len(comet_segment_scores) != len(df):
        raise RuntimeError(
            f"Mismatch between number of rows ({len(df)}) and "
            f"number of COMET scores ({len(comet_segment_scores)})."
        )

    # ---------- Compute BLEU ----------
    bleu_segment_scores, bleu_system_score = compute_bleu_scores(df)
    if len(bleu_segment_scores) != len(df):
        raise RuntimeError(
            f"Mismatch between number of rows ({len(df)}) and "
            f"number of BLEU scores ({len(bleu_segment_scores)})."
        )

    # ---------- Compute BLEURT (with graceful fallback) ----------
    bleurt_segment_scores, bleurt_system_score = compute_bleurt_scores(df)
    if len(bleurt_segment_scores) != len(df):
        raise RuntimeError(
            f"Mismatch between number of rows ({len(df)}) and "
            f"number of BLEURT scores ({len(bleurt_segment_scores)})."
        )

    # ---------- Attach scores back to the DataFrame ----------
    df["comet_score"] = comet_segment_scores
    df["bleu_score"] = bleu_segment_scores
    df["bleurt_score"] = bleurt_segment_scores

    # ---------- Append a summary row with system-level averages ----------
    summary_row = {col: None for col in df.columns}
    if "id" in summary_row:
        summary_row["id"] = "__SYSTEM_AVG__"
    summary_row["comet_score"] = comet_system_score
    summary_row["bleu_score"] = bleu_system_score
    summary_row["bleurt_score"] = bleurt_system_score

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    print("Columns just before saving (with metric scores):", list(df.columns))
    print("First 3 rows before final save:")
    print(df.head(3))
    print("-" * 80)

    # ---------- Save CSV (overwrite the same file) ----------
    print(f"Saving updated CSV (overwriting) to: {abs_path}")
    try:
        df.to_csv(abs_path, index=False, encoding="utf-8")
    except PermissionError as e:
        print("\nERROR: Could not write to the CSV file during final save.")
        print("Most likely the file is OPEN in Excel or another program.")
        print("Close the file and run this script again.")
        raise e
    print("CSV saved successfully.")
    print("-" * 80)

    # ---------- Sanity check: reload and confirm columns exist ----------
    print(f"Re-loading CSV from: {abs_path} to verify changes...")
    final_df = pd.read_csv(abs_path)
    print("Columns after final reload:", list(final_df.columns))
    print("First 3 rows after final reload:")
    print(final_df.head(3))

    if "comet_score" not in final_df.columns:
        raise RuntimeError(
            "Sanity check failed: 'comet_score' column not found after saving!"
        )
    if "bleu_score" not in final_df.columns:
        raise RuntimeError(
            "Sanity check failed: 'bleu_score' column not found after saving!"
        )
    if "bleurt_score" not in final_df.columns:
        raise RuntimeError(
            "Sanity check failed: 'bleurt_score' column not found after saving!"
        )

    print("\nDone. Per-sentence scores written to columns:")
    print("  - comet_score")
    print("  - bleu_score")
    print("  - bleurt_score")
    print("A final summary row (id='__SYSTEM_AVG__') contains system-level averages.")

    print("\nSystem-level scores for the whole file:")
    print(f"  COMET : {comet_system_score:.4f}")
    print(f"  BLEU  : {bleu_system_score:.4f}")
    print(f"  BLEURT: {bleurt_system_score:.4f}")


if __name__ == "__main__":
    main()
