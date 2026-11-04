"""
TB-AMR-Variant-ML: End-to-End Execution Pipeline
=================================================
Runs the full machine learning and genomic biomarker discovery workflow on:
- 100% REAL clinical M. tuberculosis patient isolates from NCBI SRA / BV-BRC
- OR the hypothesis-testing synthetic cohort containing candidate novM membrane mutations.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrices,
    plot_roc_pr_curves,
)
from src.explain import (
    compute_treeshap_contributions,
    export_top_driver_table,
    generate_shap_bar_plot,
    generate_shap_beeswarm_plot,
)
from src.feature_engineering import prepare_train_test_data
from src.real_data_loader import load_real_tb_cohort
from src.synthetic_generator import save_cohort_dataset
from src.train import save_models, train_models


def run_pipeline(data_mode: str = "real", target_drug: str = "isoniazid", max_isolates: int = 800):
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    figures_dir = project_root / "results" / "figures"
    tables_dir = project_root / "results" / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"TB-AMR-Variant-ML Pipeline [{data_mode.upper()} DATA MODE - {target_drug.upper()}]")
    print("=" * 78)

    # Step 1: Data Acquisition
    if data_mode.lower() == "real":
        print(f"\n[Step 1/5] Ingesting REAL clinical M. tuberculosis WGS cohort (NCBI SRA / BV-BRC)...")
        real_processed = data_dir / "processed" / f"real_cohort_{target_drug.lower()}.csv"
        if not real_processed.exists():
            X, y, metadata, df_annot = load_real_tb_cohort(
                data_dir, target_drug=target_drug, max_isolates=max_isolates, min_maf=0.025
            )
        else:
            df_full = pd.read_csv(real_processed, index_col=0)
            y = df_full["amr_phenotype"]
            X = df_full.drop(columns=["amr_phenotype"])
    else:
        print(f"\n[Step 1/5] Ingesting Synthetic M. tb cohort with novel membrane candidate (novM)...")
        snp_matrix_file = data_dir / "processed" / "snp_matrix.csv"
        if not snp_matrix_file.exists():
            save_cohort_dataset(data_dir, n_samples=600, seed=42)
        df_full = pd.read_csv(snp_matrix_file, index_col=0)
        y = df_full["amr_phenotype"]
        X = df_full.drop(columns=["amr_phenotype"])

    print(f"  - Total Patient Isolates: {len(X)}")
    print(f"  - Total Filtered SNP Loci: {X.shape[1]}")
    print(f"  - Resistant Prevalence: {y.mean():.1%} ({y.sum()} resistant / {len(y) - y.sum()} susceptible)")

    # Step 2: Quality Filtering & Stratification
    print("\n[Step 2/5] Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        X, y, test_size=0.20, min_maf=0.01, random_state=42
    )
    print(f"  - Training Set: {X_train.shape[0]} isolates, {X_train.shape[1]} SNPs")
    print(f"  - Test Set:     {X_test.shape[0]} isolates (held out for unbiased evaluation)")

    # Step 3: Model Training
    print("\n[Step 3/5] Training regularized tree ensembles (XGBoost & Random Forest)...")
    xgb_model, rf_model, cv_results = train_models(X_train, y_train, random_state=42)
    print(f"  - XGBoost 5-Fold CV ROC-AUC:       {cv_results['xgboost']['cv_roc_auc_mean']:.3f} +/- {cv_results['xgboost']['cv_roc_auc_std']:.3f}")
    print(f"  - Random Forest 5-Fold CV ROC-AUC: {cv_results['random_forest']['cv_roc_auc_mean']:.3f} +/- {cv_results['random_forest']['cv_roc_auc_std']:.3f}")

    save_models(xgb_model, rf_model, models_dir, prefix=data_mode)

    # Step 4: Model Evaluation
    print("\n[Step 4/5] Evaluating performance on withheld clinical test set...")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")

    df_metrics = pd.DataFrame([xgb_metrics, rf_metrics])
    prefix = f"{data_mode}_{target_drug}"
    df_metrics.to_csv(tables_dir / f"model_performance_{prefix}.csv", index=False)

    print("\nTest Set Benchmarking Results:")
    print(df_metrics[["model", "accuracy", "f1_score", "roc_auc", "pr_auc"]].to_string(index=False))

    plot_roc_pr_curves(xgb_model, rf_model, X_test, y_test, figures_dir / f"roc_pr_curves_{prefix}.png")
    plot_confusion_matrices(xgb_model, rf_model, X_test, y_test, figures_dir / f"confusion_matrices_{prefix}.png")

    # Step 5: SHAP Interpretability
    print("\n[Step 5/5] Computing TreeSHAP values for biological discovery on real patient genomes...")
    shap_vals, base_val, mean_abs_shap = compute_treeshap_contributions(xgb_model, X_test)

    generate_shap_bar_plot(mean_abs_shap, top_n=10, output_path=figures_dir / f"shap_summary_bar_{prefix}.png")
    generate_shap_beeswarm_plot(shap_vals, X_test, top_n=10, output_path=figures_dir / f"shap_beeswarm_{prefix}.png")

    drivers_df = export_top_driver_table(
        mean_abs_shap, X_test, y_test, output_path=tables_dir / f"top_driver_mutations_{prefix}.csv", top_n=10
    )

    print("\nTop Discovered Resistance Drivers (SHAP Ranking):")
    cols_to_print = ["Rank", "Mutation_Locus", "Target_Gene", "Mean_Abs_SHAP", "Enrichment_Ratio", "Biological_Classification"]
    print(drivers_df[cols_to_print].to_string(index=False))

    print("\n" + "=" * 78)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"Data Source Used: {'100% Real NCBI SRA / BV-BRC Clinical TB Isolates' if data_mode == 'real' else 'Laboratory Simulation Cohort'}")
    print(f"Artifacts saved in: {figures_dir} and {tables_dir}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TB AMR Variant ML Pipeline")
    parser.add_argument("--data", choices=["real", "synthetic"], default="real", help="Data source: real or synthetic")
    parser.add_argument("--drug", default="isoniazid", help="Target antibiotic (e.g. isoniazid, rifampicin)")
    args = parser.parse_args()

    run_pipeline(data_mode=args.data, target_drug=args.drug)
