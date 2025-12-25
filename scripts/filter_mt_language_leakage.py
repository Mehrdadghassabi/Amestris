#!/usr/bin/env python3
"""
CSV Machine-Translation Leakage Filter (Heuristic)

This script post-processes a CSV file containing machine-translation (MT) outputs and
removes rows that appear to be *untranslated* (or partially untranslated) because the
MT output contains a high proportion of words from an undesired language.

Why this exists
---------------
When translating in batches (e.g., concatenating WMT segments until a budget such as
~1500 "words/tokens" is reached), smaller MT models can occasionally fail and return
the source text (or a mixture) instead of a proper translation. This script provides
a lightweight, repeatable filtering step that flags and prunes those failures.

Approach (lightweight heuristic)
--------------------------------
For each row:
1) Extract word-like tokens from the MT column.
2) Count how many tokens look like the undesired language (e.g., German or English),
   using a simple heuristic detector:
   - common function words
   - typical suffix patterns
   - language-specific characters (for German: äöüß)
3) Compute ratio = undesired_word_count / total_word_count.
4) If ratio >= threshold, the row is considered problematic.

Outputs
-------
Given an input file: /path/data.csv

The script creates, next to the input file:
- data_<mode>_problem_rows.csv   (full problematic rows + optional stats columns)
- data_<mode>_problem_ids.csv    (only the ID column for problematic rows)

It also rewrites the original CSV *without* the problematic rows, after creating a
backup copy:
- data_backup_before_removal.csv (backup of the original input)

Command-line usage examples
---------------------------
Remove rows whose MT output contains too many German words:

  python filter_mt_language_leakage.py \
    --input "/path/to/dataset_wmt14_de_en_train.csv" \
    --mode german \
    --threshold 0.20 \
    --text-column MT \
    --id-column id

Remove rows whose MT output contains too many English words:

  python filter_mt_language_leakage.py \
    --input "/path/to/dataset_wmt14_en_fr_train.csv" \
    --mode english \
    --threshold 0.20

Notes
-----
- This is a heuristic filter, not a full language identification system.
- Threshold selection is task-dependent; 0.20 (20%) is a reasonable starting point.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


# ----------------------------
# Heuristic language detectors
# ----------------------------

GERMAN_COMMON_WORDS = {
    # Articles & pronouns
    "der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "ist", "und", "oder", "nicht", "kein", "keine",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "ihn",
    "mir", "dir", "uns", "euch", "mein", "dein", "sein", "ihr", "unser",
    "euer",
    # Very common words
    "zu", "mit", "auf", "für", "von", "aus", "im", "in", "am", "an", "als",
    "bei", "nach", "vor", "über", "unter", "zwischen", "weil", "wenn",
    "so", "aber", "auch", "noch", "schon", "nur", "sehr", "mehr", "weniger",
    "war", "waren", "hat", "haben", "wird", "werden", "kann", "können",
    "muss", "müssen", "soll", "sollen", "will", "wollen",
}

GERMAN_SUFFIXES: Tuple[str, ...] = (
    "ung", "keit", "heit", "chen", "lein", "schaft", "lich", "isch",
    "los", "bar", "haft", "tum", "sam", "ig",
)


def is_german_word(word: str) -> bool:
    """
    Heuristic check whether a token looks German.

    Signals:
    - Contains umlauts/ß
    - Appears in a list of common German words
    - Matches typical German suffix patterns
    """
    if not word:
        return False

    w = word.lower()

    # Typical German characters
    if any(ch in w for ch in "äöüß"):
        return True

    # Common German words
    if w in GERMAN_COMMON_WORDS:
        return True

    # Typical suffixes
    if len(w) > 4 and any(w.endswith(suf) for suf in GERMAN_SUFFIXES):
        return True

    return False


ENGLISH_COMMON_WORDS = {
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Conjunctions / prepositions
    "and", "or", "but", "if", "because", "while", "though", "although",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "about",
    "as", "into", "over", "under", "between", "through", "during", "after", "before",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    # Common verbs / helpers
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    # Negation / misc
    "not", "no", "yes", "so", "very", "more", "most", "less", "least",
    "there", "here", "then", "than", "when", "where", "what", "which", "who", "whom", "why", "how",
}

ENGLISH_SUFFIXES: Tuple[str, ...] = (
    "ing", "ed", "ly", "tion", "sion", "ment", "ness", "ity",
    "able", "ible", "ous", "ive", "al", "er", "est", "ism", "ist",
)

ENGLISH_CONTRACTION_ENDINGS: Tuple[str, ...] = (
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't",
)


def is_english_word(word: str) -> bool:
    """
    Heuristic check whether a token looks English.

    Signals:
    - ASCII letters (plus apostrophes)
    - Appears in a list of common English words
    - Matches common English suffixes
    - Looks like a contraction (don't, I'm, we're, etc.)
    """
    if not word:
        return False

    w = word.lower().strip("'")
    if not w:
        return False

    # Basic "English-like" filter: letters and apostrophes only
    if any(not (ch.isalpha() or ch == "'") for ch in w):
        return False

    # Reject non-ASCII (e.g., umlauts)
    try:
        w.encode("ascii")
    except UnicodeEncodeError:
        return False

    if w in ENGLISH_COMMON_WORDS:
        return True

    if any(w.endswith(end) for end in ENGLISH_CONTRACTION_ENDINGS) and len(w) > 2:
        return True

    if len(w) > 4 and any(w.endswith(suf) for suf in ENGLISH_SUFFIXES):
        return True

    return False


# ----------------------------
# Tokenization / word extraction
# ----------------------------

_WORD_RE_GERMAN = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_WORD_RE_ENGLISH = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def extract_words(text: str, mode: str) -> List[str]:
    """
    Extract word-like tokens from the given text.

    - german: keeps A-Z plus German umlauts and ß
    - english: keeps ASCII letters and allows internal apostrophes (contractions)
    """
    if not text:
        return []

    if mode == "german":
        return _WORD_RE_GERMAN.findall(text)
    if mode == "english":
        return _WORD_RE_ENGLISH.findall(text)

    # Defensive fallback (should not be reached due to argparse choices)
    return re.findall(r"[A-Za-zÄÖÜäöüß]+", text)


# ----------------------------
# Core processing
# ----------------------------

def _resolve_detector(mode: str) -> Tuple[Callable[[str], bool], str, str]:
    """
    Returns:
      - is_word_fn: token -> bool (undesired language detector)
      - stats_prefix: prefix used for output stats columns
      - report_tag: tag used in output filenames
    """
    if mode == "german":
        return is_german_word, "german", "german"
    if mode == "english":
        return is_english_word, "english", "english"
    raise ValueError(f"Unsupported mode: {mode}")


def filter_csv_language_leakage(
    input_csv_path: str,
    text_column_name: str,
    id_column_name: str,
    mode: str,
    ratio_threshold: float,
    rewrite_input: bool = True,
) -> None:
    """
    Filter a CSV file by detecting undesired-language leakage in the MT column.

    This function:
    - reads the input CSV
    - identifies problematic rows by undesired-language ratio
    - writes two report CSVs (rows + IDs)
    - optionally rewrites the input CSV to remove problematic rows, after a backup
    """
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    if not (0.0 <= ratio_threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1].")

    is_undesired_word, stats_prefix, report_tag = _resolve_detector(mode)

    base, ext = os.path.splitext(input_csv_path)
    output_rows_csv_path = f"{base}_{report_tag}_problem_rows.csv"
    output_ids_csv_path = f"{base}_{report_tag}_problem_ids.csv"
    backup_main_csv_path = f"{base}_backup_before_removal{ext}"

    problematic_rows: List[Dict[str, str]] = []
    problematic_ids: List[str] = []
    non_problematic_rows: List[Dict[str, str]] = []
    total_rows = 0

    with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("No header/fieldnames found in the CSV file.")

        original_fieldnames = list(reader.fieldnames)

        if text_column_name not in original_fieldnames:
            raise ValueError(
                f"Column '{text_column_name}' not found in CSV header. "
                f"Available columns: {original_fieldnames}"
            )
        if id_column_name not in original_fieldnames:
            raise ValueError(
                f"Column '{id_column_name}' not found in CSV header. "
                f"Available columns: {original_fieldnames}"
            )

        # Process each row
        for row in reader:
            total_rows += 1
            text = row.get(text_column_name, "") or ""
            words = extract_words(text, mode=mode)

            if not words:
                # No word-like tokens → keep the row
                non_problematic_rows.append(row)
                continue

            undesired_count = sum(1 for w in words if is_undesired_word(w))
            total_word_count = len(words)
            ratio = undesired_count / total_word_count

            if ratio >= ratio_threshold:
                # Add lightweight diagnostics to the "problem rows" report
                row[f"_{stats_prefix}_word_count"] = str(undesired_count)
                row["_total_word_count"] = str(total_word_count)
                row[f"_{stats_prefix}_ratio"] = f"{ratio:.2f}"
                problematic_rows.append(row)

                problematic_ids.append(row.get(id_column_name, ""))
            else:
                non_problematic_rows.append(row)

    # Determine fieldnames for the problematic-rows report
    if problematic_rows:
        extra_fields = [
            f"_{stats_prefix}_word_count",
            "_total_word_count",
            f"_{stats_prefix}_ratio",
        ]
        # Preserve original columns first, then append stats columns if present
        fieldnames = original_fieldnames + [
            f for f in extra_fields if any(f in r for r in problematic_rows)
        ]
    else:
        fieldnames = original_fieldnames

    # Write report: full problematic rows
    with open(output_rows_csv_path, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in problematic_rows:
            writer.writerow(row)

    # Write report: problematic IDs only
    with open(output_ids_csv_path, "w", encoding="utf-8-sig", newline="") as f_out_ids:
        id_writer = csv.writer(f_out_ids)
        id_writer.writerow([id_column_name])
        for pid in problematic_ids:
            id_writer.writerow([pid])

    # Optionally rewrite main CSV without problematic rows (with a backup)
    if rewrite_input and problematic_rows:
        shutil.copy2(input_csv_path, backup_main_csv_path)

        with open(input_csv_path, "w", encoding="utf-8-sig", newline="") as f_main_out:
            writer = csv.DictWriter(f_main_out, fieldnames=original_fieldnames)
            writer.writeheader()
            for row in non_problematic_rows:
                writer.writerow(row)

        main_file_msg = (
            "Main file cleaned: problematic rows removed.\n"
            f"Backup of original main file: {backup_main_csv_path}"
        )
    elif rewrite_input and not problematic_rows:
        main_file_msg = "Main file not modified (no problematic rows detected)."
    else:
        main_file_msg = "Main file not modified (rewrite disabled)."

    # Console report
    print(f"Input file: {input_csv_path}")
    print(f"Detection mode: {mode}")
    print(f"Text column: {text_column_name}")
    print(f"ID column: {id_column_name}")
    print(f"Threshold: {ratio_threshold:.2f} (i.e., {ratio_threshold * 100:.0f}%)")
    print(f"Output file with problematic rows: {output_rows_csv_path}")
    print(f"Output file with problematic IDs only: {output_ids_csv_path}")
    print(f"Total rows checked: {total_rows}")
    print(f"Problematic rows (>= {ratio_threshold * 100:.0f}% {mode} words): {len(problematic_rows)}")
    print(main_file_msg)

    if problematic_rows:
        print("\nIDs of problematic rows:")
        for pid in problematic_ids:
            print(pid)
    else:
        print("\nNo problematic rows found.")


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filter_mt_language_leakage.py",
        description=(
            "Detect and remove CSV rows whose MT output contains a high ratio of words from an undesired language "
            "(a common sign that the model returned the source text instead of a translation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["german", "english"],
        help=(
            "Undesired-language detector to apply. "
            "Use 'german' to flag German leakage, or 'english' to flag English leakage."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Rows with undesired_word_ratio >= threshold are flagged as problematic.",
    )
    parser.add_argument(
        "--text-column",
        default="MT",
        help="Name of the CSV column containing the MT output to inspect.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Name of the CSV column containing the unique row identifier.",
    )

    rewrite_group = parser.add_mutually_exclusive_group()
    rewrite_group.add_argument(
        "--rewrite-input",
        dest="rewrite_input",
        action="store_true",
        default=True,
        help="Rewrite the input CSV after filtering (creates a backup first).",
    )
    rewrite_group.add_argument(
        "--no-rewrite-input",
        dest="rewrite_input",
        action="store_false",
        help="Do not modify the input CSV; only write the report CSV files.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    filter_csv_language_leakage(
        input_csv_path=args.input,
        text_column_name=args.text_column,
        id_column_name=args.id_column,
        mode=args.mode,
        ratio_threshold=args.threshold,
        rewrite_input=args.rewrite_input,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
