# Gemma-3 4B Preference Data for WMT14 German↔English MT

This repository contains the **data-preparation pipeline** used in our paper to build **preference-style (preferred/rejected) training data** for German↔English machine translation, starting from **WMT14** and using **Gemma-3 (4B parameters)** as the MT system under study.

The goal of this stage is to produce clean, reproducible CSV artifacts that can be directly consumed by preference-based fine-tuning / RL-style methods (e.g., DPO-style “preferred vs. rejected” training).

> **Scope of this repo:** dataset extraction + filtering + MT leakage cleanup + metric scoring (COMET/BLEU) + selection of low-quality “rejected” examples + prompt-wrapping utilities. Model training scripts are *not* included in this folder (this repo is the **data** stage).

---

## Repository contents

### Scripts

- `filter_mt_language_leakage.py`  
  Heuristic filter to detect and remove **failed translations** where the MT output contains too many words from the *undesired* language (typical when batched translation returns source text or a mixed output).

- `mt_score_csv.py`  
  Computes **reference-based COMET** and **BLEU** scores for MT outputs in a CSV, appends per-row scores, and adds a final system-level summary row.

- `select_bottom_comet.py`  
  Selects the **bottom fraction** (default **20%**) of rows by COMET score to represent **rejected** (low-quality) translations. Also reports a diagnostic “knee” score and can plot histograms/ECDF.

- `add_translation_prompt.py`  
  Prepends a strict, professional translation instruction prompt to a chosen text column to ensure consistent MT/API querying.

### Data artifacts (example filenames)


- `dataset_wmt14_du_en_train_source.csv`  
  Raw/merged source used for training extraction (includes an initial feasibility slice and a larger range).

- `dataset_wmt14_*_train_before_filter.csv`  
  Intermediate output after initial constraints (e.g., max-length filtering) and MT generation.

- `dataset_wmt14_*_train_with_scores.csv`  
  Output after metric scoring (COMET + BLEU).

- `dataset_wmt14_*_train_kept_20pct_thr_<X>.csv`  
  Bottom-20% COMET subset used to build **rejected** candidates.

- `dataset_wmt14_*_train_final.csv`  
  Final preference-formatted training CSVs (preferred/rejected columns) ready for training.

- `dataset_wmt14_*_test_source.csv`, `dataset_wmt14_*_test_final.csv`  
  Test split artifacts for evaluation (scored with the same pipeline).

---

## Data schema conventions

Our pipeline assumes CSVs with a unique row identifier and bilingual text columns.

Typical columns (names may vary; the scripts allow mapping via CLI flags):

- `id` — unique row identifier
- `du` — German sentence (Deutsch)
- `en` — English sentence
- `mt` or `MT` — machine translation output from Gemma-3 4B
- `comet_score` — per-segment COMET score (added by `mt_score_csv.py`)
- `bleu_score` — sentence-level BLEU score (added by `mt_score_csv.py`)

The *final* training CSVs used for preference learning typically include:

- `prompt` or prompted-source text (optional)
- `preferred` — preferred translation
- `rejected` — rejected translation
- plus any metadata needed for training (ids, direction tags, scores)

Because teams vary in column naming, all scripts expose flags to specify column names.

---

## Pipeline overview

The full data-prep pipeline is:

1. **Extract a working subset** from WMT14 German↔English.
2. **Apply length constraints** (e.g., max 40 words) to keep examples within a desired range.
3. **Generate MT outputs** using Gemma-3 4B (batched API calls).
4. **Filter translation failures** using `filter_mt_language_leakage.py`.
5. **Score translations** with COMET + BLEU using `mt_score_csv.py`.
6. **Select low-quality outputs** (bottom 20% COMET) as **rejected** candidates via `select_bottom_comet.py`.
7. **Add consistent prompts** (optional but recommended) using `add_translation_prompt.py`.
8. **Assemble final preference pairs** into `*_train_final.csv` for training.

Steps (3) and (8) depend on your MT inference/training stack (API vs. local inference, preference format, etc.). This repo provides the utilities used for filtering/scoring/selection and prompt standardization.

---

## Installation

Recommended environment:

- Python **3.10+**
- CPU is sufficient for BLEU; **COMET is faster with CUDA**, but can run on CPU.

Install dependencies:

```bash
pip install -U pip
pip install pandas numpy matplotlib torch sacrebleu "unbabel-comet>=2.0.0"
```

> If you do not have a GPU, COMET will run on CPU (slower). You can force CPU with `--gpus 0`.

---

## Usage

### 1) Filter MT language leakage (failed / untranslated outputs)

For **German → English** MT outputs: undesired leakage is typically **German** inside the English MT output.

```bash
python filter_mt_language_leakage.py \
  --input dataset_wmt14_du_to_en_train_before_filter.csv \
  --mode german \
  --threshold 0.20 \
  --text-column MT \
  --id-column id
```

For **English → German** MT outputs: undesired leakage is typically **English** inside the German MT output.

```bash
python filter_mt_language_leakage.py \
  --input dataset_wmt14_en_to_du_train_before_filter.csv \
  --mode english \
  --threshold 0.20 \
  --text-column MT \
  --id-column id
```

Outputs (written next to the input file):

- `*_backup_before_removal.csv` — backup of the original
- `*_<mode>_problem_rows.csv` — problematic rows with diagnostic counts/ratios
- `*_<mode>_problem_ids.csv` — IDs of problematic rows
- the input file is optionally rewritten with problematic rows removed (disable with `--no-rewrite-input`)

**Threshold guidance:** `0.20` (20%) is a reasonable starting point; tune depending on how strict you want the leakage filter.

---

### 2) Score MT outputs with COMET + BLEU

Score a German → English file where:

- source column: `du`
- reference column: `en`
- MT column: `mt` (or set `--mt_col MT` if your column is uppercase)

```bash
python mt_score_csv.py \
  --csv dataset_wmt14_du_to_en_train_after_filter.csv \
  --src_col du \
  --ref_col en \
  --mt_col MT \
  --id_col id
```

Score an English → German file:

```bash
python mt_score_csv.py \
  --csv dataset_wmt14_en_to_du_train_after_filter.csv \
  --src_col en \
  --ref_col du \
  --mt_col MT \
  --id_col id
```

What the script does:

- Adds per-row `comet_score` and `bleu_score`
- Appends a final row with system-level scores (id `__SYSTEM_AVG__` when an id column exists)
- Removes a previous summary row automatically if the file was scored before

By default, the input CSV is overwritten. Use `--output_csv` to write to a new file.

---

### 3) Select the bottom 20% by COMET (rejected candidates)

```bash
python select_bottom_comet.py \
  --input dataset_wmt14_du_to_en_train_with_scores.csv \
  --column comet_score \
  --keep-fraction 0.20
```

This writes a new CSV:

- `*_bottom_0.20_thr_<quantile>.csv`

Notes:

- The script also reports a “knee” threshold as a **diagnostic reference**, but selection is **strictly quantile-based**.
- By default, **NaN scores are kept** (can be changed with `--drop-nans`).
- Plots are enabled by default; disable on servers with `--no-plots`.

---

### 4) Add a strict translation prompt (optional, recommended)

German → English (defaults to column `du`):

```bash
python add_translation_prompt.py \
  --input dataset_wmt14_du_to_en_train_kept_20pct_thr_0.7771.csv \
  --direction de-en
```

English → German (defaults to column `en`):

```bash
python add_translation_prompt.py \
  --input dataset_wmt14_en_to_du_train_kept_20pct_thr_0.7624.csv \
  --direction en-de
```

This writes `*_with_prompt.csv` next to the input file.

---

## Reproducing the paper artifacts

To reproduce the exact artifact set used in the paper:

1. Start from the same WMT14 split and extraction range used in our study.
2. Apply the same preprocessing constraints (e.g., max-length filtering).
3. Produce MT outputs with **Gemma-3 4B** using the same prompting strategy.
4. Run leakage filtering → scoring → bottom selection using the commands above.
5. Construct the final preference CSVs (`*_train_final.csv`) in your chosen format.

### Test evaluation

The repo includes test artifacts (`*_test_source.csv` / `*_test_final.csv`). You can score the test outputs with `mt_score_csv.py` using the same column mapping as training.

---

## Practical GitHub notes

- **Large files:** If you plan to publish the large CSVs on GitHub, use **Git LFS**.
  - Otherwise, store them externally and keep only scripts + small samples.

- Consider adding a `.gitignore` for:
  - backups (`*_BACKUP_*.csv`, `*_backup_before_removal.csv`)
  - problem reports (`*_problem_rows.csv`, `*_problem_ids.csv`)

---

## Citation

If you use this repository or the processed datasets in academic work, please cite our paper:

```bibtex
@article{YOURKEY,
  title   = {TODO: Title},
  author  = {TODO: Authors},
  journal = {TODO},
  year    = {2026}
}
```

---

## License

Specify a license appropriate for your release (e.g., Apache-2.0 or MIT for code). Note that **WMT14 data** and **COMET models** have their own licensing terms; include links and comply with upstream licenses.

---

## Acknowledgements

- WMT14 German↔English dataset
- COMET (Unbabel)
- SacreBLEU
- Gemma-3 model family

---

## Contact

For questions about this repository or reproduction details, open an issue or contact the authors listed in the paper.

