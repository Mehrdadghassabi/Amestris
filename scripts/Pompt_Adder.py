import pandas as pd
from pathlib import Path

# --- Paths ---
input_path = Path(r"C:\Users\HAMAHANG\Desktop\1B\Train\en_to_de\wmt14_en_de_mt_FULL_cleaned_ELBOW_low_thr_0.6349_kept_13.0pct.csv")
output_path = input_path.with_name(input_path.stem + "_with_prompt.csv")

# --- Prompt to prepend ---
PROMPT = """You are an expert professional translator (Endglish to German), working on data for fine-tuning a machine learning model.
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

def main():
    # Read CSV (keep strings; keep empty cells as empty strings)
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    # Ensure column exists
    if "en" not in df.columns:
        raise KeyError(f'Column "du" not found. Available columns: {list(df.columns)}')

    # Rename column
    df = df.rename(columns={"en": "PromptandText"})

    # Prepend prompt to every row in the column
    df["PromptandText"] = PROMPT + "\n\n" + df["PromptandText"].fillna("")

    # Write new CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Done! Wrote: {output_path}")

if __name__ == "__main__":
    main()
