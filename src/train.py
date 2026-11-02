"""
Model Training Module
=====================
Trains and validates ensemble machine learning models (XGBoost & Random Forest)
for binary Antimicrobial Resistance (AMR) classification from genomic variant data.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[xgb.XGBClassifier, RandomForestClassifier, Dict[str, Any]]:
    """
    Trains XGBoost and Random Forest classifiers with Stratified K-Fold CV.
    Returns: (xgb_model, rf_model, cv_results_dict)
    """
    # 1. Handle class imbalance
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = n_neg / max(n_pos, 1)

    # 2. Configure XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,   # L1 regularization for sparse genomic feature selection
        reg_lambda=1.5,  # L2 regularization to prevent overfitting on passenger SNPs
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    # 3. Configure Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    # 4. Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    xgb_cv_roc = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="roc_auc")
    xgb_cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="f1")

    rf_cv_roc = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="roc_auc")
    rf_cv_f1 = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="f1")

    # Fit both models on the full training partition
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    cv_results = {
        "xgboost": {
            "cv_roc_auc_mean": float(np.mean(xgb_cv_roc)),
            "cv_roc_auc_std": float(np.std(xgb_cv_roc)),
            "cv_f1_mean": float(np.mean(xgb_cv_f1)),
            "cv_f1_std": float(np.std(xgb_cv_f1)),
        },
        "random_forest": {
            "cv_roc_auc_mean": float(np.mean(rf_cv_roc)),
            "cv_roc_auc_std": float(np.std(rf_cv_roc)),
            "cv_f1_mean": float(np.mean(rf_cv_f1)),
            "cv_f1_std": float(np.std(rf_cv_f1)),
        },
    }

    return xgb_model, rf_model, cv_results


def save_models(
    xgb_model: xgb.XGBClassifier,
    rf_model: RandomForestClassifier,
    output_dir: Path,
    prefix: str = "",
) -> Tuple[Path, Path]:
    """
    Saves trained model artifacts to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_str = f"{prefix}_" if prefix else ""
    xgb_path = output_dir / f"xgboost_{prefix_str}amr.json"
    rf_path = output_dir / f"rf_{prefix_str}amr.joblib"

    xgb_model.save_model(str(xgb_path))
    joblib.dump(rf_model, rf_path)

    # Also save standard name for fallback compatibility
    xgb_model.save_model(str(output_dir / "xgboost_amr.json"))
    joblib.dump(rf_model, output_dir / "rf_amr.joblib")

    return xgb_path, rf_path
