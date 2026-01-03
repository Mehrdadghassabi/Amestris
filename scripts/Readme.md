### Scripts

- `filter_mt_language_leakage.py`  
  Heuristic filter to detect and remove **failed translations** where the MT output contains too many words from the *undesired* language (typical when batched translation returns source text or a mixed output).

- `mt_score_csv.py`  
  Computes **reference-based COMET** and **BLEU** scores for MT outputs in a CSV, appends per-row scores, and adds a final system-level summary row.

- `select_bottom_comet.py`  
  Selects the **bottom fraction** (default **20%**) of rows by COMET score to represent **rejected** (low-quality) translations. Also reports a diagnostic “knee” score and can plot histograms/ECDF.

- `add_translation_prompt.py`  
  Prepends a strict, professional translation instruction prompt to a chosen text column to ensure consistent MT/API querying.
