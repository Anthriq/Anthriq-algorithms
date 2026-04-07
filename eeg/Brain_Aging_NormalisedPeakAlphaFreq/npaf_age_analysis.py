"""
src/npaf/npaf_age_analysis.py — N-PAF vs Age / NART correlation analysis.

Port of npafnart1.m. Computes:
    1. Age vs N-PAF and Age vs NART correlations
    2. Simple N-PAF vs NART correlation
    3. Partial correlation (N-PAF vs NART controlling for Age) via residuals
    4. Multiple regression: NART ~ Age + N-PAF
    5. Age-stratified (median split) correlations
    6. 4-panel summary figure

Data can be provided as arrays or loaded from a CSV (e.g., output of npaf_extraction.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import linregress

import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# HARDCODED DATA (from original dataset — 60 subjects, NPAF n=58 complete)
# ============================================================================

_AGE = np.array([
    67, 69, 70, 71, 59, 22, 70, 52, 33, 41,
    64, 44, 30, 74, 38, 27, 20, 78, 30, 52,
    62, 59, 77, 22, 70, 66, 69, 69, 30, 23,
    33, 34, 48, 34, 20, 59, 40, 41, 24, 32,
    40, 68, 27, 56, 41, 27, 38, 73, 50, 30,
    44, 44, 44, 72, 52, 59, 57, 64, 57, 64,
], dtype=float)

_NART = np.array([
    12, 15, 10,  5, 12,  9,  7, 16, 13,  6,
     4, 17, 15,  9, 17, 19,  9, 15, 11, 18,
     7,  8, 10,  9,  7,  6, 10,  6, 10, 21,
    13,  9,  5, 18, 18,  4,  8,  7, 11, 24,
    17,  4, 19,  5,  6,  9, 17, 19,  9, 15,
    23, 17, 44, 72, 52, 59, 57, 64, 57, 64,
], dtype=float)

_NPAF = np.array([
     8.35,  8.21,  8.93,  8.16,  8.61, 10.28,  8.26, 10.36,  9.25,  9.13,
     8.41,  8.93, 10.10,  9.96,  9.85, 10.30, 10.30,  8.86, 10.21,  8.80,
     9.13, 10.98,  8.15, 11.31,  7.71, 11.25,  8.48,  8.73, 11.18, 10.08,
     9.03,  9.98, 10.28,  9.63, 10.03, 10.08,  9.35,  9.63,  8.91, 10.20,
     8.70,  9.55,  9.25, 10.56,  9.85,  9.68,  8.26,  8.82,  8.87,  9.53,
     7.66,  8.36,  9.20,  7.93,  7.81,  9.08,  9.06,  9.86,
], dtype=float)


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pearson r and two-tailed p-value."""
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _ols_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals from simple OLS regression of y on x."""
    slope, intercept, *_ = linregress(x, y)
    return y - (slope * x + intercept)


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Partial correlation of x and y controlling for z (residuals method).

    Regresses both x and z independently on z, then correlates residuals.
    """
    res_x = _ols_residuals(z, x)
    res_y = _ols_residuals(z, y)
    return _pearson(res_x, res_y)


def _multiple_regression(y: np.ndarray, X: np.ndarray) -> dict:
    """
    Multiple OLS via normal equations: y ~ X (X already includes intercept column).

    Returns dict with keys: coefficients, predictions, residuals.
    """
    # beta = (X'X)^{-1} X'y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    residuals = y - y_hat
    return {"coefficients": beta, "predictions": y_hat, "residuals": residuals}


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_npaf_age_analysis(
    age: np.ndarray = None,
    nart: np.ndarray = None,
    npaf: np.ndarray = None,
    out_dir: str = ".",
    figure_name: str = "npaf_age_analysis.png",
) -> dict:
    """
    Run the full N-PAF vs Age / NART correlation analysis.

    Parameters
    ----------
    age, nart, npaf : np.ndarray, optional
        Data arrays. If not provided, the hardcoded dataset is used.
    out_dir : str
        Directory to save the output figure.
    figure_name : str
        Filename for the saved figure.

    Returns
    -------
    dict with all computed statistics.
    """
    # ------------------------------------------------------------------ #
    #  Data preparation                                                    #
    # ------------------------------------------------------------------ #
    age  = np.asarray(age,  dtype=float) if age  is not None else _AGE.copy()
    nart = np.asarray(nart, dtype=float) if nart is not None else _NART.copy()
    npaf = np.asarray(npaf, dtype=float) if npaf is not None else _NPAF.copy()

    print("=== DATA PREPARATION ===")
    print(f"Original lengths — Age: {len(age)}, NART: {len(nart)}, NPAF: {len(npaf)}")

    n = min(len(age), len(nart), len(npaf))
    age, nart, npaf = age[:n], nart[:n], npaf[:n]
    print(f"Using complete cases: n = {n}\n")

    results = {}

    # ------------------------------------------------------------------ #
    #  Step 1 — Known relationships                                        #
    # ------------------------------------------------------------------ #
    print("=== STEP 1: Confirming Known Relationships ===")

    r_age_npaf, p_age_npaf = _pearson(age, npaf)
    r_age_nart, p_age_nart = _pearson(age, nart)

    print(f"Age vs NPAF : r = {r_age_npaf:.3f}, p = {p_age_npaf:.4f}")
    print(f"Age vs NART : r = {r_age_nart:.3f}, p = {p_age_nart:.4f}")

    results.update({
        "r_age_npaf": r_age_npaf, "p_age_npaf": p_age_npaf,
        "r_age_nart": r_age_nart, "p_age_nart": p_age_nart,
    })

    # ------------------------------------------------------------------ #
    #  Step 2 — Simple NPAF vs NART                                        #
    # ------------------------------------------------------------------ #
    print("\n=== STEP 2: Simple NPAF vs NART Correlation ===")

    r_npaf_nart, p_npaf_nart = _pearson(npaf, nart)
    print(f"NPAF vs NART (simple): r = {r_npaf_nart:.3f}, p = {p_npaf_nart:.4f}")

    results.update({"r_npaf_nart": r_npaf_nart, "p_npaf_nart": p_npaf_nart})

    # ------------------------------------------------------------------ #
    #  Step 3 — Partial correlation (NPAF vs NART | Age)                   #
    # ------------------------------------------------------------------ #
    print("\n=== STEP 3: KEY ANALYSIS — Partial Correlation ===")

    r_partial, p_partial = _partial_correlation(npaf, nart, age)
    print(f"NPAF vs NART (controlling for Age): r = {r_partial:.3f}, p = {p_partial:.4f}")

    results.update({"r_partial": r_partial, "p_partial": p_partial})

    # Multiple regression: NART ~ intercept + Age + NPAF
    X = np.column_stack([np.ones(n), age, npaf])
    reg = _multiple_regression(nart, X)
    intercept_coeff, age_coeff, npaf_coeff = reg["coefficients"]

    print("\n--- Multiple Regression: NART ~ Age + NPAF ---")
    print(f"  Intercept : {intercept_coeff:.3f}")
    print(f"  Age       : {age_coeff:.3f}")
    print(f"  NPAF      : {npaf_coeff:.3f}")

    results.update({
        "reg_intercept": intercept_coeff,
        "reg_age_coeff": age_coeff,
        "reg_npaf_coeff": npaf_coeff,
    })

    # ------------------------------------------------------------------ #
    #  Step 4 — Age-stratified analysis                                    #
    # ------------------------------------------------------------------ #
    print("\n=== STEP 4: Age-Stratified Analysis ===")

    median_age = float(np.median(age))
    young_mask = age <= median_age
    old_mask   = age > median_age

    print(f"Median age : {median_age:.1f} years")

    for label, mask in [("YOUNG", young_mask), ("OLD", old_mask)]:
        sub_npaf, sub_nart = npaf[mask], nart[mask]
        if len(sub_npaf) >= 5:
            r_s, p_s = _pearson(sub_npaf, sub_nart)
            print(f"{label} group (n={mask.sum()}): r = {r_s:.3f}, p = {p_s:.4f}")
            results[f"r_{label.lower()}"] = r_s
            results[f"p_{label.lower()}"] = p_s

    # ------------------------------------------------------------------ #
    #  Step 5 — Figure                                                     #
    # ------------------------------------------------------------------ #
    _plot(age, nart, npaf, results, out_dir, figure_name)

    return results


# ============================================================================
# FIGURE
# ============================================================================

def _scatter_with_fit(ax, x, y, color, xlabel, ylabel, title, xlim=None, ylim=None):
    """Scatter plot with OLS regression line."""
    ax.scatter(x, y, s=50, color=color, alpha=0.7)
    slope, intercept, *_ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def _plot(age, nart, npaf, results, out_dir, figure_name):
    """Save a 4-panel summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("N-PAF vs Age / NART Analysis", fontsize=13, fontweight="bold")

    # Panel 1 — Age vs NPAF
    _scatter_with_fit(
        axes[0, 0], age, npaf, color="#3399CC",
        xlabel="Age (years)", ylabel="N-PAF (Hz)",
        title=(f"Age vs N-PAF\nr = {results['r_age_npaf']:.3f}, "
               f"p = {results['p_age_npaf']:.3f}"),
    )

    # Panel 2 — Age vs NART
    _scatter_with_fit(
        axes[0, 1], age, nart, color="#CC6633",
        xlabel="Age (years)", ylabel="NART Score",
        title=(f"Age vs NART\nr = {results['r_age_nart']:.3f}, "
               f"p = {results['p_age_nart']:.3f}"),
    )

    # Panel 3 — NPAF vs NART (simple)
    _scatter_with_fit(
        axes[1, 0], npaf, nart, color="#9933CC",
        xlabel="N-PAF (Hz)", ylabel="NART Score",
        title=(f"N-PAF vs NART (simple)\nr = {results['r_npaf_nart']:.3f}, "
               f"p = {results['p_npaf_nart']:.3f}"),
    )

    # Panel 4 — Age-controlled residuals
    res_npaf = _ols_residuals(age, npaf)
    res_nart = _ols_residuals(age, nart)
    _scatter_with_fit(
        axes[1, 1], res_npaf, res_nart, color="#336633",
        xlabel="N-PAF residuals (age-removed)", ylabel="NART residuals (age-removed)",
        title=(f"Partial correlation (Age controlled)\nr = {results['r_partial']:.3f}, "
               f"p = {results['p_partial']:.3f}"),
    )

    plt.tight_layout()
    out_path = Path(out_dir) / figure_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="N-PAF vs Age/NART correlation analysis")
    parser.add_argument("--csv",  default=None,
                        help="CSV file with columns: age, nart, npaf. "
                             "If not provided, uses hardcoded dataset.")
    parser.add_argument("--out_dir", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--figure",  default="npaf_age_analysis.png")
    args = parser.parse_args()

    if args.csv:
        df   = pd.read_csv(args.csv)
        age  = df["age"].to_numpy()
        nart = df["nart"].to_numpy()
        npaf = df["npaf"].to_numpy()
    else:
        age = nart = npaf = None  # use hardcoded

    stats = run_npaf_age_analysis(age=age, nart=nart, npaf=npaf,
                                  out_dir=args.out_dir, figure_name=args.figure)

    print("\n=== RESULTS SUMMARY ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
