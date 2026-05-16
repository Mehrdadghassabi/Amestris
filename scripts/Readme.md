# Scripts Documentation

This folder contains the preprocessing and scoring utilities used to transform the raw Gemma 3 1B output into a cleaned, scored, prompt-formatted dataset for preference-based translation training.

The scripts are designed to be run in sequence, although each step can also be used independently when needed.

---

## 1. `Remover.py`

### Purpose

Removes residual generation artifacts from the raw translation file.

### What it cleans

The script truncates the text at the first occurrence of:

- `<end_of_turn>`

It also removes leading numbered prefixes at the beginning of the translation, such as:

- `1. ...`
- `2. ...`

These patterns often appear when a model continues generation beyond the intended translation output.

### Input

A CSV file containing at least one machine translation column, typically:

- `mt`

### Output

A new CSV file with the suffix:

- `_cleaned.csv`

### Behavior

The script:

1. Loads the CSV as strings.
2. Verifies that the target text column exists.
3. Truncates each translation at the first `<end_of_turn>` token.
4. Removes leading `1.` or `2.` prefixes when present.
5. Writes a cleaned copy to disk without modifying the original file.

### Example use

```python
remove_end_of_turns(
    input_csv="wmt14_en_de_mt_raw.csv",
    column_name="mt"
)
```

### Notes

This is a lightweight, deterministic cleanup step intended to remove obvious decoding artifacts before scoring.

---

## 2. `Comet_Score.py`

### Purpose

Computes automatic translation-quality metrics for the cleaned dataset.

### Metrics produced

The script adds the following columns:

- `comet_score`
- `bleu_score`
- `bleurt_score`

It also appends a summary row containing system-level averages.

### Input

A CSV file containing:

- the source sentence column,
- the reference translation column,
- the model output column.

The script is column-configurable, so it can be adapted to the exact naming used in your CSV.

### Output

The input CSV is overwritten by default, unless an alternate output path is configured.

### Behavior

The script performs the following steps:

1. Loads the CSV and confirms the required text columns exist.
2. Normalizes text values to strings and replaces missing values with empty strings.
3. Computes COMET scores using a reference-based COMET model.
4. Computes sentence-level and corpus-level BLEU with SacreBLEU.
5. Attempts to compute BLEURT through `evaluate`.
6. Falls back gracefully if BLEURT is unavailable in the environment.
7. Appends a system-average row with summary scores.
8. Saves the updated CSV and verifies that the score columns were written correctly.

### Notes on BLEURT

BLEURT support depends on the runtime environment.

- In a compatible environment, real BLEURT scores are computed.
- If BLEURT cannot be loaded, the script still completes and leaves `bleurt_score` as `NaN`.

This behavior preserves the rest of the pipeline even when BLEURT is unavailable.

### Notes on COMET

The script uses a Hugging Face COMET checkpoint and batch inference. GPU acceleration is used automatically when available; otherwise, inference runs on CPU.

---

## 3. `Distribution finder.py`

### Purpose

Selects low-quality translation examples from the scored dataset using the **COMET** score distribution.

This script is the rejection-mining stage of the pipeline.

### What it does

The script:

1. Loads the scored CSV.
2. Converts the COMET column to numeric values.
3. Detects an elbow/knee threshold in the score distribution.
4. Keeps the low-scoring subset.
5. Writes a filtered CSV containing the selected rows.
6. Creates a backup copy of the original input file.
7. Optionally displays diagnostic plots.

### Selection strategy

The method is **elbow-based**, not a fixed-percent rule.

The elbow threshold is found from the score distribution and then used to keep the low-quality tail of the data. In the run described in this project, the selected file kept approximately **13.0%** of the data.

### Input

A scored CSV containing a numeric `comet_score` column.

### Output

A filtered CSV whose filename includes:

- `ELBOW`
- the selected threshold
- the kept percentage

### Important settings

- `keep_mode = "low"`  
  Keeps the lower-scoring examples.

- `keep_nans = True`  
  Retains rows with missing scores unless changed.

- `make_plots = True`  
  Produces histogram, ECDF, and elbow visualizations.

### Notes

This script is used to mine translations that are weak enough to serve as rejected examples in the downstream preference dataset.

---

## 4. `Pompt_Adder.py`

### Purpose

Adds a uniform translation instruction prompt to each row and prepares the dataset for prompt-based training.

### Input

A CSV containing an English source column, typically:

- `en`

### Output

A new CSV with the suffix:

- `_with_prompt.csv`

### Behavior

The script:

1. Loads the CSV as strings.
2. Checks that the English source column exists.
3. Renames the source text column to:
   - `PromptandText`
4. Prepends a fixed instruction prompt to each source sentence.
5. Writes the updated CSV to a new file.

### Prompting strategy

The prompt is designed to encourage:

- faithful translation,
- fluent and natural German output,
- preservation of meaning, numbers, names, and factual content,
- no commentary or extra explanation,
- no merging, dropping, or reordering of content.

### Notes

This step makes the translation task explicit and consistent across all training rows. It is especially useful when the final data will be used in a preference-learning or instruction-following setup.

---

## Recommended execution order

The usual order is:

1. `Remover.py`
2. `Comet_Score.py`
3. `Distribution finder.py`
4. `Pompt_Adder.py`

This sequence keeps raw generation artifacts out of the scored dataset and ensures that only filtered examples are passed into the final prompt-formatted training file.

---

## Dependency summary

Typical dependencies across the scripts include:

- `pandas`
- `numpy`
- `matplotlib`
- `torch`
- `sacrebleu`
- `evaluate`
- `unbabel-comet`

Install them in a consistent Python environment before running the pipeline.

---

## Practical recommendation

For the cleanest workflow, treat the scripts folder as a preprocessing toolbox:

- `Remover.py` handles deterministic text cleanup,
- `Comet_Score.py` performs metric scoring,
- `Distribution finder.py` performs rejection mining,
- `Pompt_Adder.py` standardizes the final prompt format.

Together, these scripts transform raw MT outputs into a structured dataset suitable for downstream preference optimization.
