import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import shutil

# =========================
# Settings
# =========================
csv_path = Path(r"C:\Users\HAMAHANG\Desktop\1B\Train\en_to_de\wmt14_en_de_mt_FULL_cleaned.csv")
col = "comet_score"

keep_nans = True                 # keep NaNs
keep_mode = "low"                # "low" -> keep scores <= thr | "high" -> keep scores >= thr
elbow_skip_frac = 0.01           # ignore the first/last 1% when searching for elbow (avoids endpoints)
smooth_window = 101              # odd number recommended; set 0 or 1 to disable smoothing

make_plots = True

# =========================
# Helper: elbow (knee) threshold on sorted values
# Method: max deviation from diagonal on normalized sorted curve
# =========================
def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1  # make it odd
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def elbow_threshold(values: np.ndarray, mode: str = "low", skip_frac: float = 0.01, smooth_w: int = 1):
    """
    values: 1D numeric array (no NaNs)
    mode:
      - "low": compute elbow on ascending sorted values (keeps <= thr)
      - "high": compute elbow on descending sorted values (keeps >= thr)
    Returns: (thr, elbow_index, sorted_values_used, distances_used)
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("elbow_threshold: empty values array")

    if mode not in ("low", "high"):
        raise ValueError("mode must be 'low' or 'high'")

    # Sort
    vals_sorted = np.sort(values)
    if mode == "high":
        vals_sorted = vals_sorted[::-1]

    n = vals_sorted.size
    if n < 3:
        thr = float(vals_sorted[-1]) if mode == "low" else float(vals_sorted[-1])
        return thr, int(n // 2), vals_sorted, np.zeros(n, dtype=float)

    vmin = float(vals_sorted.min())
    vmax = float(vals_sorted.max())
    vrng = vmax - vmin
    if vrng == 0:
        # all values equal -> any threshold is identical
        thr = float(vals_sorted[0])
        return thr, n // 2, vals_sorted, np.zeros(n, dtype=float)

    # Normalize to [0,1]
    x = np.linspace(0.0, 1.0, n)
    y = (vals_sorted - vmin) / vrng

    # Optional smoothing (on y)
    y_s = _moving_average(y, smooth_w)

    # Distance to diagonal y=x (absolute to be robust)
    d = np.abs(y_s - x)

    # Avoid trivial elbow at endpoints by skipping a fraction on each side
    skip = int(max(1, round(skip_frac * n)))
    lo = skip
    hi = n - skip
    if hi <= lo + 1:
        lo, hi = 1, n - 1  # fallback

    elbow_rel = int(np.argmax(d[lo:hi]))
    elbow_idx = lo + elbow_rel

    thr = float(vals_sorted[elbow_idx])
    return thr, elbow_idx, vals_sorted, d


# =========================
# 1) Load CSV
# =========================
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path, low_memory=False)

if col not in df.columns:
    raise KeyError(f"Column '{col}' not found. Available columns:\n{df.columns.tolist()}")

numeric_scores = pd.to_numeric(df[col], errors="coerce")

n_total = len(df)
n_nan = int(numeric_scores.isna().sum())
n_numeric = n_total - n_nan

if n_numeric == 0:
    raise ValueError(f"No numeric values found in '{col}' after coercion.")

# =========================
# 2) Find elbow threshold on numeric values
# =========================
vals = numeric_scores.dropna().astype(float).to_numpy()

thr, elbow_idx, vals_sorted_used, dist_used = elbow_threshold(
    vals,
    mode=keep_mode,
    skip_frac=elbow_skip_frac,
    smooth_w=smooth_window
)

# Determine what percent this corresponds to (numeric-only)
if keep_mode == "low":
    kept_numeric = int(np.sum(vals <= thr))
else:
    kept_numeric = int(np.sum(vals >= thr))

kept_total_est = kept_numeric + (n_nan if keep_nans else 0)

print("\n=== Elbow threshold (auto) ===")
print(f"Mode: {keep_mode}  (low: keep <= thr | high: keep >= thr)")
print(f"NaNs: {n_nan} | Numeric: {n_numeric} | Total: {n_total}")
print(f"Elbow index in sorted curve: {elbow_idx} / {len(vals_sorted_used)-1}")
print(f"threshold (elbow) = {thr:.6f}")
print(f"Kept numeric (by threshold): {kept_numeric} ({kept_numeric/n_numeric:.2%} of numeric)")
print(f"Kept total (incl NaNs={keep_nans}): {kept_total_est} ({kept_total_est/n_total:.2%} of total)")

# =========================
# 3) Backup original file
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = csv_path.with_name(csv_path.stem + f"_BACKUP_{timestamp}" + csv_path.suffix)

shutil.copy2(csv_path, backup_path)
print("\n=== Backup created ===")
print(backup_path)

# =========================
# 4) Filter rows by elbow threshold
# =========================
if keep_mode == "low":
    core_mask = numeric_scores.notna() & (numeric_scores <= thr)
else:
    core_mask = numeric_scores.notna() & (numeric_scores >= thr)

if keep_nans:
    mask_keep = numeric_scores.isna() | core_mask
else:
    mask_keep = core_mask

removed = int((~mask_keep).sum())
kept = int(mask_keep.sum())

df_filtered = df.loc[mask_keep].copy()

print("\n=== Filtering result (ELBOW) ===")
print(f"Removed rows: {removed} ({removed/n_total:.2%})")
print(f"Kept rows:    {kept} ({kept/n_total:.2%})")

# =========================
# 5) Save filtered CSV (new file)
# =========================
kept_pct = 100.0 * kept / n_total if n_total else 0.0
out_path = csv_path.with_name(
    csv_path.stem + f"_ELBOW_{keep_mode}_thr_{thr:.4f}_kept_{kept_pct:.1f}pct" + csv_path.suffix
)
df_filtered.to_csv(out_path, index=False)

print("\n=== Saved filtered CSV ===")
print(out_path)

# =========================
# 6) (Optional) Quick plots
# =========================
if make_plots:
    plot_scores = vals.astype(float)

    # Histogram (auto bins via Freedman–Diaconis, fallback)
    q25, q75 = np.percentile(plot_scores, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(plot_scores) ** (-1/3)) if iqr > 0 else None
    bins = max(10, int(np.ceil((plot_scores.max() - plot_scores.min()) / bin_width))) if bin_width and bin_width > 0 else 50

    plt.figure(figsize=(10, 5))
    plt.hist(plot_scores, bins=bins)
    plt.axvline(thr, linewidth=2, label=f"elbow thr = {thr:.4f}")
    plt.title(f"Distribution of {col} (Histogram)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ECDF
    sorted_scores = np.sort(plot_scores)
    ecdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_scores, ecdf)
    plt.axvline(thr, linewidth=2, label=f"elbow thr = {thr:.4f}")
    plt.title(f"ECDF of {col}")
    plt.xlabel(col)
    plt.ylabel("Fraction ≤ score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Elbow plot: normalized sorted curve + diagonal + elbow point
    vals_for_plot = np.sort(plot_scores)
    if keep_mode == "high":
        vals_for_plot = vals_for_plot[::-1]

    vmin = vals_for_plot.min()
    vmax = vals_for_plot.max()
    vrng = vmax - vmin if vmax > vmin else 1.0

    x = np.linspace(0.0, 1.0, len(vals_for_plot))
    y = (vals_for_plot - vmin) / vrng
    y_s = _moving_average(y, smooth_window)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_s, label="normalized sorted curve (smoothed)" if smooth_window > 1 else "normalized sorted curve")
    plt.plot(x, x, linewidth=1, label="diagonal y=x")
    plt.axvline(x[elbow_idx], linewidth=2, label=f"elbow idx = {elbow_idx}")
    plt.title(f"Elbow detection on sorted {col} (mode={keep_mode})")
    plt.xlabel("Normalized index")
    plt.ylabel("Normalized score")
    plt.legend()
    plt.tight_layout()
    plt.show()
