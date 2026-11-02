"""
Model Evaluation & Benchmarking Module
======================================
Computes comprehensive classification metrics and generates publication-quality
performance plots (ROC curves, PR curves, Confusion Matrices).
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Computes key performance metrics for a binary classifier.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(pr_auc),
    }


def plot_roc_pr_curves(
    xgb_model: Any,
    rf_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Plots paired ROC and Precision-Recall curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    rf_auc = roc_auc_score(y_test, rf_proba)

    xgb_prec, xgb_rec, _ = precision_recall_curve(y_test, xgb_proba)
    rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_proba)
    xgb_pr_auc = auc(xgb_rec, xgb_prec)
    rf_pr_auc = auc(rf_rec, rf_prec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. ROC Curve
    axes[0].plot(xgb_fpr, xgb_tpr, color="#2563EB", lw=2.2, label=f"XGBoost (AUC = {xgb_auc:.3f})")
    axes[0].plot(rf_fpr, rf_tpr, color="#059669", lw=2.2, label=f"Random Forest (AUC = {rf_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", lw=1.5)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)

    # 2. Precision-Recall Curve
    baseline = np.sum(y_test == 1) / len(y_test)
    axes[1].plot(xgb_rec, xgb_prec, color="#2563EB", lw=2.2, label=f"XGBoost (PR-AUC = {xgb_pr_auc:.3f})")
    axes[1].plot(rf_rec, rf_prec, color="#059669", lw=2.2, label=f"Random Forest (PR-AUC = {rf_pr_auc:.3f})")
    axes[1].axhline(y=baseline, color="#9CA3AF", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline:.2f})")
    axes[1].set_title("Precision-Recall Curve (Imbalance-Aware)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(
    xgb_model: Any,
    rf_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Plots confusion matrices for both classifiers side by side.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_xgb = confusion_matrix(y_test, xgb_model.predict(X_test))
    cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    for ax, cm, title in zip(
        axes,
        [cm_xgb, cm_rf],
        ["XGBoost Confusion Matrix", "Random Forest Confusion Matrix"],
    ):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Susceptible (0)", "Resistant (1)"],
            yticklabels=["Susceptible (0)", "Resistant (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Phenotype", fontsize=11)
        ax.set_ylabel("True Laboratory DST Phenotype", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
