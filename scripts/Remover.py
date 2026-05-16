import os
import pandas as pd

def remove_end_of_turns(
    input_csv: str,
    column_name: str = "mt",
    token: str = "<end_of_turn>",
    output_suffix: str = "_cleaned"
) -> str:
    """
    1) Truncate text in `column_name` at the first occurrence of `token`
       (removes the token and everything after it).
    2) If the resulting text starts with "1." or "2.", remove that prefix.
    Saves as a new CSV and returns the output file path.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    base, ext = os.path.splitext(input_csv)
    output_csv = f"{base}{output_suffix}{ext}"

    df = pd.read_csv(input_csv, dtype=str)

    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found. Available columns: {list(df.columns)}"
        )

    # Fill missing, then truncate at token (keep text before token only)
    s = df[column_name].fillna("")

    # Option A: split at first token (simple + safe)
    s = s.str.split(token, n=1).str[0]

    # NEW: remove leading "1." or "2." (and any spaces after) at the beginning only
    # Examples: "1. Hello" -> "Hello", "2.   Hi" -> "Hi"
    s = s.str.replace(r'^\s*[12]\.\s*', '', regex=True)

    df[column_name] = s

    # Save to a new file (does not modify original)
    df.to_csv(output_csv, index=False)

    return output_csv

if __name__ == "__main__":
    input_path = r"C:\Users\HAMAHANG\Desktop\1B\Train\en_to_de\wmt14_de_en_mt_FULL.csv"
    out_path = remove_end_of_turns(input_path)
    print(f"Saved cleaned copy to:\n{out_path}")
