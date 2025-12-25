#!/usr/bin/env python3
"""
Append translation prompts to CSV text columns for MT fine-tuning data.

This script takes a CSV file containing source-language text and prepends a
carefully specified translation prompt to each entry in a given text column.
It is designed for preparing data used to query MT APIs in a consistent,
reproducible way.

The script supports two translation directions:

    - German → English  (direction: de-en)
    - English → German  (direction: en-de)

For each row, the script modifies only the specified text column:

    <PROMPT>

    <original text>

No columns are renamed. By default, the text column is inferred from the
direction:

    - de-en : column "du" (German source)
    - en-de : column "en" (English source)

You can override this with --column-name if needed.

The output file is written next to the input, using the pattern:

    <stem>_with_prompt.csv

Example usage
-------------

German → English (default column "du"):

    python add_translation_prompt.py \
        --input "C:/path/dataset_wmt14_du_en_train_kept_20pct_thr_0.7771.csv" \
        --direction de-en

English → German (default column "en"):

    python add_translation_prompt.py \
        --input "C:/path/dataset_wmt14_fr_en_train_with_answer_kept_20pct_thr_0.7624.csv" \
        --direction en-de

Using a custom text column name:

    python add_translation_prompt.py \
        --input "data.csv" \
        --direction de-en \
        --column-name source_text

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Prompts used for the two translation directions
# ---------------------------------------------------------------------------

DE_TO_EN_PROMPT = """You are an expert professional translator (German → English), working on data for fine-tuning a machine learning model.
Translation requirements:

Preserve all meaning, nuance, and details.

Do not skip, summarize, shorten, merge, or add content.

Use clear, natural, fluent, professional standard English.

Preserve factual and contextual accuracy (dates, numbers, names, product names, etc.).

If something is already in English (words, phrases, acronyms, etc.), keep it as-is unless a simple grammatical adjustment is clearly needed.

If a term cannot be translated reliably (e.g., brand name, typo, very local slang, ambiguous proper noun), keep the original German term and translate around it.

Do not invent extra information.

Do not add explanations or comments; only provide the direct translation.

Structure rules:

Do not add, remove, merge, or reorder objects.

Text to translate:

"""

EN_TO_DE_PROMPT = """You are an expert professional translator (English → German), working on data for fine-tuning a machine learning model.
Translation requirements:

Preserve all meaning, nuance, and details.

Do not skip, summarize, shorten, merge, or add content.

Use clear, natural, fluent, professional standard German.

Preserve factual and contextual accuracy (dates, numbers, names, product names, etc.).

If something is already in German (words, phrases, acronyms, etc.), keep it as-is unless a simple grammatical adjustment is clearly needed.

If a term cannot be translated reliably (e.g., brand name, typo, very local slang, ambiguous proper noun), keep the original English term and translate around it.

Do not invent extra information.

Do not add explanations or comments; only provide the direct translation.

Structure rules:

Do not add, remove, merge, or reorder objects.

Text to translate:

"""


# Default text-column names per direction, matching the original datasets
DEFAULT_COLUMN_BY_DIRECTION = {
    "de-en": "du",  # German source column
    "en-de": "en",  # English source column
}


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def prepend_prompt_to_column(
    input_path: Path,
    output_path: Path,
    direction: str,
    text_column_name: str,
    encoding: str = "utf-8-sig",
) -> None:
    """
    Read the input CSV, prepend the appropriate translation prompt to the
    specified text column, and write the result to output_path.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    output_path : Path
        Path where the modified CSV will be written.
    direction : {"de-en", "en-de"}
        Translation direction. Determines which prompt is used.
    text_column_name : str
        Name of the column containing the source text that will be translated.
        This column is modified in-place (no renaming).
    encoding : str, optional
        File encoding for reading and writing. Default is "utf-8-sig".
    """
    if direction not in ("de-en", "en-de"):
        raise ValueError(f"Unsupported direction: {direction!r}. Expected 'de-en' or 'en-de'.")

    # Select the appropriate prompt for the chosen direction
    if direction == "de-en":
        prompt = DE_TO_EN_PROMPT
    else:  # direction == "en-de"
        prompt = EN_TO_DE_PROMPT

    # Read CSV with explicit string handling:
    # - dtype=str        : keep all cells as strings
    # - keep_default_na=False : do not convert empty strings to NaN
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding=encoding)

    # Validate the existence of the text column
    if text_column_name not in df.columns:
        raise KeyError(
            f'Column "{text_column_name}" not found in input file. '
            f"Available columns: {list(df.columns)}"
        )

    # Prepend the prompt to every row in the chosen column.
    # The original column name is preserved; only the cell contents are modified.
    df[text_column_name] = prompt + "\n\n" + df[text_column_name].fillna("")

    # Write the modified DataFrame back to CSV
    df.to_csv(output_path, index=False, encoding=encoding)

    print(f"Done! Wrote: {output_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="add_translation_prompt.py",
        description=(
            "Prepend a fixed translation prompt to a text column in a CSV file, "
            "for preparing MT data (German ↔ English)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Path to the output CSV file. "
            "If omitted, a file with suffix '_with_prompt.csv' will be created "
            "next to the input file."
        ),
    )

    parser.add_argument(
        "--direction",
        "-d",
        type=str,
        choices=["de-en", "en-de"],
        required=True,
        help=(
            "Translation direction. "
            "'de-en' = German → English, 'en-de' = English → German. "
            "This selects the appropriate prompt."
        ),
    )

    parser.add_argument(
        "--column-name",
        "-c",
        type=str,
        default=None,
        help=(
            "Name of the text column to which the prompt will be prepended. "
            "If not provided, a direction-specific default is used: "
            "'du' for de-en, 'en' for en-de."
        ),
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="Encoding used for reading and writing the CSV files.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        # Default: add "_with_prompt" to the stem, preserve suffix
        output_path = input_path.with_name(input_path.stem + "_with_prompt" + input_path.suffix)

    # Determine column name: either user-specified or default for the direction
    if args.column_name is not None:
        text_column_name = args.column_name
    else:
        text_column_name = DEFAULT_COLUMN_BY_DIRECTION.get(args.direction)
        if text_column_name is None:
            raise ValueError(
                f"No default text column configured for direction {args.direction!r}. "
                f"Please specify --column-name explicitly."
            )

    prepend_prompt_to_column(
        input_path=input_path,
        output_path=output_path,
        direction=args.direction,
        text_column_name=text_column_name,
        encoding=args.encoding,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
