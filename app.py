"""
TB-AMR-Variant-ML: Interactive Web Application
==============================================
Streamlit application for exploring Mycobacterium tuberculosis genomic variants,
inspecting XGBoost antimicrobial resistance models, and querying SHAP driver discoveries
across BOTH 100% Real Clinical Data (NCBI / BV-BRC) and Laboratory Candidate Simulation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="M. tuberculosis AMR Variant Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .source-badge-real {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .source-badge-sim {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


def safe_show_dataframe(df: pd.DataFrame):
    """Safely renders DataFrame with automatic fallback."""
    try:
        st.dataframe(df)
    except Exception:
        try:
            st.table(df)
        except Exception:
            st.write(df)


# Sidebar Selection
st.sidebar.title("🧬 Settings & Data")
data_mode = st.sidebar.selectbox(
    "Choose Dataset Mode:",
    [
        "🟢 Real Clinical Data (NCBI SRA / BV-BRC)",
        "🧪 Lab Simulation (Novel novM Candidate)",
    ],
)
is_real = "Real" in data_mode

# Sidebar Navigation
menu = st.sidebar.radio(
    "Go to",
    [
        "1. Project Overview & Background",
        "2. Clinical Cohort Exploration",
        "3. Model Performance & Benchmarks",
        "4. SHAP Biomarker Discovery",
        "5. Live Resistance Predictor",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Context:**\n"
    "Computational genomics pipeline for predicting antimicrobial resistance in *M. tuberculosis*.\n"
    "Supports real NCBI WGS benchmarking and genomic variant prioritization."
)


@st.cache_data
def load_dataset(use_real: bool):
    if use_real:
        data_file = DATA_DIR / "processed" / "real_cohort_isoniazid.csv"
        drivers_file = RESULTS_DIR / "tables" / "top_driver_mutations_real_isoniazid.csv"
        metrics_file = RESULTS_DIR / "tables" / "model_performance_real_isoniazid.csv"
        fig_roc = RESULTS_DIR / "figures" / "roc_pr_curves_real_isoniazid.png"
        fig_cm = RESULTS_DIR / "figures" / "confusion_matrices_real_isoniazid.png"
        fig_bar = RESULTS_DIR / "figures" / "shap_summary_bar_real_isoniazid.png"
        fig_bee = RESULTS_DIR / "figures" / "shap_beeswarm_real_isoniazid.png"
    else:
        data_file = DATA_DIR / "processed" / "snp_matrix.csv"
        drivers_file = RESULTS_DIR / "tables" / "top_driver_mutations.csv"
        metrics_file = RESULTS_DIR / "tables" / "model_performance.csv"
        fig_roc = RESULTS_DIR / "figures" / "roc_pr_curves.png"
        fig_cm = RESULTS_DIR / "figures" / "confusion_matrices.png"
        fig_bar = RESULTS_DIR / "figures" / "shap_summary_bar.png"
        fig_bee = RESULTS_DIR / "figures" / "shap_beeswarm.png"

    df = pd.read_csv(data_file, index_col=0) if data_file.exists() else None
    df_drivers = pd.read_csv(drivers_file) if drivers_file.exists() else None
    df_metrics = pd.read_csv(metrics_file) if metrics_file.exists() else None

    return df, df_drivers, df_metrics, fig_roc, fig_cm, fig_bar, fig_bee


@st.cache_resource
def load_xgb_model(use_real: bool):
    filename = "xgboost_real_amr.json" if use_real else "xgboost_synthetic_amr.json"
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        model_path = MODELS_DIR / "xgboost_amr.json"
    if not model_path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


df_data, df_drivers, df_metrics, fig_roc, fig_cm, fig_bar, fig_bee = load_dataset(is_real)
model = load_xgb_model(is_real)

# ---------------- PAGE 1 ----------------
if menu == "1. Project Overview & Background":
    st.markdown('<div class="main-title">Antimicrobial Resistance (AMR) Prediction from WGS Variant Data</div>', unsafe_allow_html=True)
    if is_real:
        st.markdown('<span class="source-badge-real">ACTIVE MODE: 100% Real Clinical Data (NCBI SRA / BV-BRC Global Cohort)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="source-badge-sim">ACTIVE MODE: Lab Hypothesis Simulation (Novel novM Membrane Protein)</span>', unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown(
            """
            ### Research Motivation
            * *Mycobacterium tuberculosis* ($M. tb$) is a pathogen where drug resistance develops primarily through chromosomal mutations and membrane transport adaptations.
            * In clinical settings, **thousands of neutral single nucleotide polymorphisms (SNPs)** occur across patients as genetic background noise.
            * Finding which specific mutation sites actually cause treatment failure requires machine learning algorithms that:
              1. **Penalize sparse noise:** $L_1$ and $L_2$ regularized tree boosting (XGBoost).
              2. **Capture non-linear epistasis:** Interaction between multiple mutations (e.g. efflux pump activation + target mutation).
              3. **Explain predictions biologically:** Using game-theoretic **TreeSHAP** to rank candidate driver mutations without treating the model as a black box.
            """
        )
    with col2:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Mycobacterium_tuberculosis_8524_lores.jpg/640px-Mycobacterium_tuberculosis_8524_lores.jpg",
            caption="Mycobacterium tuberculosis bacilli (CDC)",
            use_container_width=True,
        )

# ---------------- PAGE 2 ----------------
elif menu == "2. Clinical Cohort Exploration":
    st.markdown('<div class="main-title">Clinical Cohort & Genomic Feature Matrix</div>', unsafe_allow_html=True)
    if is_real:
        st.markdown('<span class="source-badge-real">Dataset: Real NCBI SRA Patient Isolates with Laboratory Isoniazid Phenotypes</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="source-badge-sim">Dataset: Synthetic Cohort with novM Candidate</span>', unsafe_allow_html=True)

    if df_data is not None:
        y = df_data["amr_phenotype"]
        X = df_data.drop(columns=["amr_phenotype"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patient Isolates", len(df_data))
        c2.metric("Total SNP Loci Filtered", X.shape[1])
        res_count = int(y.sum())
        c3.metric("Resistant Strains", f"{res_count} ({res_count/len(df_data):.1%})")
        c4.metric("Susceptible Strains", f"{len(df_data)-res_count} ({(len(df_data)-res_count)/len(df_data):.1%})")

        st.subheader("Genomic Matrix Preview (Patient Isolates x SNP Loci)")
        safe_show_dataframe(df_data.head(15))
    else:
        st.warning("Dataset not loaded. Run `python main.py` first.")

# ---------------- PAGE 3 ----------------
elif menu == "3. Model Performance & Benchmarks":
    st.markdown('<div class="main-title">Model Performance & Test Set Benchmarking</div>', unsafe_allow_html=True)
    if df_metrics is not None:
        safe_show_dataframe(df_metrics)

    c1, c2 = st.columns(2)
    with c1:
        if fig_roc.exists():
            st.image(str(fig_roc), caption="ROC & Precision-Recall Curves", use_container_width=True)
    with c2:
        if fig_cm.exists():
            st.image(str(fig_cm), caption="Confusion Matrices", use_container_width=True)

# ---------------- PAGE 4 ----------------
elif menu == "4. SHAP Biomarker Discovery":
    st.markdown('<div class="main-title">SHAP Interpretability: "The Hero Step"</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Extracting biological candidates: Distinguishing causal drivers from passenger noise.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if fig_bar.exists():
            st.image(str(fig_bar), caption="Top Mutations Ranked by Mean |SHAP|", use_container_width=True)
    with c2:
        if fig_bee.exists():
            st.image(str(fig_bee), caption="Directional Impact Across Clinical Isolates", use_container_width=True)

    if df_drivers is not None:
        st.subheader("Top Discovered Driver Mutation Loci")
        safe_show_dataframe(df_drivers)

# ---------------- PAGE 5 ----------------
elif menu == "5. Live Resistance Predictor":
    st.markdown('<div class="main-title">Interactive Clinical Strain Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Toggle mutation profiles and observe model resistance probabilities in real time.</div>', unsafe_allow_html=True)

    if model is not None and df_data is not None:
        feature_cols = [c for c in df_data.columns if c != "amr_phenotype"]
        st.markdown("### Select Mutations Detected in Clinical Isolate:")

        colA, colB = st.columns(2)
        selected_mutations = {}

        if is_real:
            # Top real drivers from NCBI
            top_real_muts = [
                ("katG_pos_2155168_C>G", "katG Codon 315 (Ser315Thr - Primary Isoniazid driver)"),
                ("rpoB_pos_761155_C>T", "rpoB Codon 450 (Ser450Leu - Rifampicin/MDR co-driver)"),
                ("inhA_pos_1673425_C>T", "inhA Promoter c.-15C>T (Low-level Isoniazid driver)"),
                ("gyrA_pos_7582_A>G", "gyrA Asp94Gly (Fluoroquinolone driver)"),
                ("pos_781687_A>G", "Background Genomic Variant (pos 781,687)"),
                ("pos_839534_A>C", "Background Genomic Variant (pos 839,534)"),
            ]
            half = len(top_real_muts) // 2
            with colA:
                for m_id, label in top_real_muts[:half]:
                    selected_mutations[m_id] = st.checkbox(f"**{m_id}**: {label}", value=False)
            with colB:
                for m_id, label in top_real_muts[half:]:
                    selected_mutations[m_id] = st.checkbox(f"**{m_id}**: {label}", value=False)
        else:
            sim_muts = [
                ("katG_S315T", "katG Ser315Thr (High-confidence Isoniazid driver)"),
                ("rpoB_S450L", "rpoB Ser450Leu (High-confidence Rifampicin driver)"),
                ("novM_G214R_Efflux", "novM Gly214Arg (Novel Membrane Candidate / Efflux activation)"),
                ("gyrA_D94G", "gyrA Asp94Gly (Fluoroquinolone driver)"),
                ("inhA_promoter_C-15T", "inhA Promoter c.-15C>T (Low-level INH resistance)"),
                ("novM_L108P_TM2", "novM Leu108Pro (Novel Membrane TM2 helix destabilization)"),
            ]
            half = len(sim_muts) // 2
            with colA:
                for m_id, label in sim_muts[:half]:
                    selected_mutations[m_id] = st.checkbox(f"**{m_id}**: {label}", value=False)
            with colB:
                for m_id, label in sim_muts[half:]:
                    selected_mutations[m_id] = st.checkbox(f"**{m_id}**: {label}", value=False)

        if st.button("🔮 Predict Resistance Probability", type="primary"):
            # Ensure features match model's expected inputs
            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
            else:
                model_features = feature_cols

            sample_vec = pd.DataFrame(0, index=[0], columns=model_features)
            for m_id, val in selected_mutations.items():
                if m_id in sample_vec.columns and val:
                    sample_vec.at[0, m_id] = 1

            prob_res = float(model.predict_proba(sample_vec)[0, 1])
            pred_class = "Resistant" if prob_res >= 0.50 else "Susceptible"

            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted Phenotype", pred_class)
                st.metric("Resistance Probability", f"{prob_res:.1%}")
            with c2:
                if prob_res >= 0.50:
                    st.error(f"⚠️ **High Clinical Resistance Risk ({prob_res:.1%})**\n\nDriver mutations detected.")
                else:
                    st.success(f"✅ **Susceptible Profile ({prob_res:.1%})**\n\nNo primary driver mutations active.")
    else:
        st.warning("Model not found.")
