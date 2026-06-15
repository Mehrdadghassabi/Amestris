# English→German Translation Dataset and Preference Pipeline

This repository contains the data-preparation workflow used to build an **English→German** translation dataset from **WMT14**, generate model translations with **Gemma 3 1B**, clean generation artifacts, score outputs with automatic metrics, and prepare the data for **preference-based fine-tuning**.

The pipeline in this version is **English→German only**.  

you can download the Data from these links:

https://drive.google.com/file/d/1nUV9zA8s5W5nkRC3V_bCiCBGTijkd2Wg/view?usp=drivesdk

https://drive.google.com/file/d/1Srku-DuWImZDkHOgRJg1G5SNYhJumnU-/view?usp=drivesdk

https://drive.google.com/file/d/1Dut2sFThE6YCflsIB-IctiGjxPGh-nxq/view?usp=drivesdk

https://drive.google.com/file/d/110x7m6l_y1wCxw732Tdo7u-65tZWPApR/view?usp=drivesdk

https://drive.google.com/file/d/16y_zFE68jdYOzo1lzIZjuZYuS6LQHsyR/view?usp=drivesdk


## Project summary

The workflow follows these stages:

1. Download a working subset of **200,000 WMT14 English→German entries**.
2. Extract the English source column and run **Gemma 3 1B** in **Kaggle** using **FP32** inference.
3. Save the raw model outputs to `wmt14_en_de_mt_raw.csv`.
4. Clean residual generation artifacts with `Remover.py`.
5. Score the cleaned translations with **COMET**, **BLEU**, and **BLEURT** using `Comet_Score.py`.
6. Identify low-quality translations with `Distribution finder.py` using the **COMET** distribution.
7. Add a standardized translation prompt with `Pompt_Adder.py`.
8. Produce the final cleaned, scored, prompt-formatted artifact for downstream preference learning.

The final dataset is intended for **preference-optimization training** workflows that use **preferred / rejected** examples, such as DPO-style pipelines or other pairwise optimization methods.

---

## Data source

The starting point for this dataset is **WMT14 English→German**.

From the larger corpus, a subset of **200,000 entries** was downloaded and used as the source pool for generation and filtering. The English source sentences were passed through **Gemma 3 1B** to generate machine translations in German.

---

## Processing pipeline

### 1. Raw model generation

The English source sentences were translated with **Gemma 3 1B** on Kaggle with **FP32** precision.

**Raw output file:**
- `wmt14_en_de_mt_raw.csv`

This file contains the model-generated translations before artifact cleaning.

### 2. Artifact removal

Some generated rows contained residual formatting artifacts, including tokens such as:

- `<end_of_turn>`
- leading numbered prefixes such as `1.` or `2.`

These artifacts were removed using `Remover.py`, producing:

- `wmt14_en_de_mt_raw_cleaned_Before_Comet.csv`

### 3. Metric scoring

The cleaned file was then processed by `Comet_Score.py`, which adds:

- `comet_score`
- `bleu_score`
- `bleurt_score`

A system-average summary row is also appended, typically identified as:

- `__SYSTEM_AVG__`

The output after scoring was:

- `wmt14_en_de_mt_raw_cleaned_After_Comet.csv`

### 4. Low-quality selection

To build a rejected pool for preference learning, the scored file was passed through `Distribution finder.py`.

This script analyzes the **COMET** score distribution and applies an **elbow-based threshold** to retain the weaker translations. In the run described here, the output file was:

- `wmt14_en_de_mt_raw_cleaned_ELBOW_low_thr_0.6349_kept_13.0pct.csv`

This selection step is used to isolate the model outputs that should serve as **rejected** candidates in the downstream preference dataset.

### 5. Prompt standardization

A consistent instruction prompt was added with `Pompt_Adder.py` to make the translation task uniform for downstream training.

The script prepends a translation instruction prompt to the source text and renames the source text column to a training-friendly field name.

The final artifact produced was:

- `wmt14_en_de_mt_raw_cleaned_ELBOW_low_thr_0.6349_kept_13.0pct_with_prompt_Final.csv`

---

## Expected schema

Column names are kept flexible across the scripts, but the typical structure is:

- `id` — unique row identifier
- `en` — English source sentence
- `de` — German reference translation, if available in the source file
- `mt` / `MT` / `answer` — machine translation output from Gemma 3 1B
- `comet_score` — segment-level COMET score
- `bleu_score` — segment-level BLEU score
- `bleurt_score` — segment-level BLEURT score
- `PromptandText` — prompt plus source text, created by the prompt-adder step

For preference-learning exports, the final training format typically uses fields such as:

- `preferred`
- `rejected`

In this project, the **human reference translation** is the natural choice for the **preferred** side, while the filtered Gemma output is used as the **rejected** side.

---

## Installation

The processing scripts were written for Python 3.10+.

Recommended packages:

```bash
pip install -U pip
pip install pandas numpy matplotlib torch sacrebleu "unbabel-comet>=2.0.0" evaluate
```

### Notes on BLEURT

BLEURT may not be available in every environment. In particular, it can be difficult to install on newer Python versions.

- If BLEURT loads successfully, the script computes real BLEURT scores.
- If BLEURT cannot be loaded, the script falls back gracefully and fills the BLEURT column with `NaN`, while still computing COMET and BLEU.

---

## Reproducibility notes

To reproduce the same artifact sequence:

1. Start from the same **WMT14 English→German** source split and extraction range.
2. Generate translations with **Gemma 3 1B** using the same Kaggle/FP32 setup.
3. Clean generation artifacts before scoring.
4. Score with COMET, BLEU, and BLEURT.
5. Apply the same elbow-based COMET filtering strategy.
6. Add the standardized prompt.
7. Convert the resulting file into the preference format required by your downstream training code.

Because the elbow threshold depends on the score distribution, the exact cutoff may change if the generation model, prompt, or preprocessing changes.

---

## File naming convention

The repository uses descriptive filenames that reflect the processing stage:

- `*_raw.csv` — raw model output
- `*_cleaned*.csv` — artifact-cleaned output
- `*_After_Comet.csv` — scored output
- `*_ELBOW_*kept_*pct.csv` — low-quality subset selected from the COMET distribution
- `*_with_prompt_Final.csv` — final prompt-formatted artifact

---

## Recommended downstream use

This dataset is intended for **preference-based fine-tuning** rather than standard supervised translation only.

Typical downstream uses include:

- building `preferred` / `rejected` pairs,
- training a pairwise preference model,
- running DPO-style optimization,
- evaluating translation quality under preference-learning objectives.

The final CSV is therefore a prepared intermediate artifact, not merely a raw translation dump.

---

## Repository contents

See the `scripts/` documentation for a detailed explanation of:

- `Remover.py`
- `Comet_Score.py`
- `Distribution finder.py`
- `Pompt_Adder.py`

Each script is documented with its role, inputs, outputs, and key implementation behavior.
