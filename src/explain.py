"""
SHAP Interpretability & Biomarker Discovery ("The Hero Step")
==============================================================
Extracts biological meaning from the trained XGBoost model using TreeSHAP:
- Calculates exact Shapley additive explanations for every SNP feature.
- Ranks top resistance driver mutations vs neutral passenger SNPs.
- Produces publication-grade SHAP summary and directional beeswarm visualizations.
- Maps discovered mutation loci to biological gene targets and functional annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb


def compute_treeshap_contributions(
    xgb_model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Computes exact TreeSHAP contributions natively from XGBoost booster.
    Returns:
    - shap_values: np.ndarray of shape (n_samples, n_features)
    - base_value: expected margin/bias
    - mean_abs_shap: pd.Series of mean(|SHAP|) indexed by feature names
    """
    booster = xgb_model.get_booster()
    dmatrix = xgb.DMatrix(X)
    
    # pred_contribs=True yields shape (n_samples, n_features + 1)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    
    shap_values = contribs[:, :-1]
    base_value = contribs[:, -1]
    
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_abs_shap = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    
    return shap_values, base_value, mean_abs_shap


def generate_shap_bar_plot(
    mean_abs_shap: pd.Series,
    top_n: int = 12,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots a horizontal bar chart of the top N mutations ranked by mean |SHAP| value.
    Highlights novel membrane protein candidates with distinct coloration.
    """
    top_series = mean_abs_shap.head(top_n).iloc[::-1]  # ascending for horizontal plot

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Color novel membrane candidates differently from known targets and passenger SNPs
    bar_colors = []
    for feat in top_series.index:
        if "novM" in feat:
            bar_colors.append("#7C3AED")  # Purple for Novel Membrane Protein Candidate
        elif any(k in feat for k in ["katG", "rpoB", "inhA", "gyrA", "embB", "pncA"]):
            bar_colors.append("#2563EB")  # Blue for Known High-Confidence Drivers
        else:
            bar_colors.append("#94A3B8")  # Slate Gray for Neutral Passenger SNPs

    bars = ax.barh(top_series.index, top_series.values, color=bar_colors, edgecolor="black", linewidth=0.6, height=0.65)
    
    # Add data labels at the end of each bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_title(f"Top {top_n} Mutation Biomarkers Driving Antimicrobial Resistance (SHAP Ranking)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Average Model Impact Magnitude)", fontsize=11)
    ax.set_ylabel("Genomic Mutation Locus", fontsize=11)
    ax.set_xlim(0, max(top_series.values) * 1.15)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563EB", edgecolor="black", label="Known Clinical Driver"),
        Patch(facecolor="#7C3AED", edgecolor="black", label="Novel Lab Membrane Candidate (novM)"),
        Patch(facecolor="#94A3B8", edgecolor="black", label="Background Passenger SNP"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def generate_shap_beeswarm_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 12,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Renders an informative directional beeswarm scatter plot showing whether
    the presence of a mutation (1=High/Red) increases or decreases predicted resistance.
    """
    feature_names = list(X.columns)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n][::-1]  # bottom to top for plotting

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    rng = np.random.default_rng(42)

    y_positions = np.arange(len(top_indices))
    ax.axvline(x=0, color="#64748B", linestyle="--", lw=1.2, alpha=0.8)

    try:
        cmap = plt.colormaps["coolwarm"]
    except Exception:
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap("coolwarm")

    for y_idx, feat_idx in enumerate(top_indices):
        feat_name = feature_names[feat_idx]
        vals = shap_values[:, feat_idx]
        feature_vals = X.iloc[:, feat_idx].values  # 0 or 1 for binary SNPs

        # Add vertical jitter to prevent overplotting
        jitter = rng.normal(0, 0.08, size=len(vals))
        y_scatter = y_idx + jitter

        # Color: 0 (wild-type) -> blueish, 1 (mutant) -> reddish
        colors = [cmap(0.9) if v == 1 else cmap(0.15) for v in feature_vals]
        ax.scatter(vals, y_scatter, c=colors, alpha=0.65, s=28, edgecolors="none")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10, fontweight="bold")
    ax.set_xlabel("SHAP Value (Impact on Model Log-Odds: -> Pushes towards Resistant)", fontsize=11)
    ax.set_title("Directional SHAP Impact Summary Across Clinical Isolates", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Legend for feature value
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.9), markersize=8, label="Mutant Present (1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.15), markersize=8, label="Wild-type (0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def export_top_driver_table(
    mean_abs_shap: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compiles a comprehensive biological driver candidate table with frequencies,
    SHAP rankings, and biological annotations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    res_mask = (y == 1)
    susc_mask = (y == 0)

    for rank, (feat, score) in enumerate(mean_abs_shap.head(top_n).items(), start=1):
        col_vals = X[feat]
        freq_res = col_vals[res_mask].mean()
        freq_susc = col_vals[susc_mask].mean()

        # Categorize
        if "novM" in feat:
            category = "Novel Membrane Protein Candidate (Efflux/Channel)"
            gene = "novM"
        elif "katG" in feat:
            category = "Known First-Line Driver (Isoniazid)"
            gene = "katG"
        elif "rpoB" in feat:
            category = "Known First-Line Driver (Rifampicin)"
            gene = "rpoB"
        elif "inhA" in feat:
            category = "Known First-Line Driver (Isoniazid Promoter)"
            gene = "fabG1-inhA"
        elif "gyrA" in feat:
            category = "Known Second-Line Driver (Fluoroquinolone)"
            gene = "gyrA"
        elif "embB" in feat:
            category = "Known First-Line Driver (Ethambutol)"
            gene = "embB"
        elif "pncA" in feat:
            category = "Known First-Line Driver (Pyrazinamide)"
            gene = "pncA"
        else:
            category = "Putative Passenger / Lineage Variant"
            gene = "Intergenic / Neutral"

        rows.append(
            {
                "Rank": rank,
                "Mutation_Locus": feat,
                "Target_Gene": gene,
                "Mean_Abs_SHAP": round(score, 4),
                "Freq_In_Resistant": round(freq_res, 3),
                "Freq_In_Susceptible": round(freq_susc, 3),
                "Enrichment_Ratio": round((freq_res + 0.001) / (freq_susc + 0.001), 2),
                "Biological_Classification": category,
            }
        )

    df_drivers = pd.DataFrame(rows)
    df_drivers.to_csv(output_path, index=False)
    return df_drivers
