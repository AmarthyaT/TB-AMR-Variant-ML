"""
Synthetic Clinical M. tuberculosis Cohort Generator
====================================================
Generates a biologically grounded cohort of clinical Mycobacterium tuberculosis
isolates with Whole Genome Sequencing (WGS) variant calls and phenotypic Drug
Susceptibility Testing (DST) labels.

Simulates:
1. Canonical resistance driver mutations (katG, rpoB, inhA, gyrA).
2. The laboratory's novel uncharacterized membrane protein candidate (novM).
3. Realistic background genomic noise (passenger SNPs, lineage markers).
4. Epistatic/synergistic multi-gene resistance interactions.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


# Biologically curated loci in M. tuberculosis (H37Rv coordinates approx.)
KEY_DRIVER_LOCI = [
    {
        "id": "katG_S315T",
        "gene": "katG",
        "chrom": "NC_000962.3",
        "pos": 2155168,
        "ref": "G",
        "alt": "C",
        "annotation": "Ser315Thr (Catalase-peroxidase loss -> INH activation failure)",
        "base_weight": 3.8,
        "freq_in_res": 0.72,
        "freq_in_susc": 0.02,
        "category": "Known High-Confidence Driver",
    },
    {
        "id": "rpoB_S450L",
        "gene": "rpoB",
        "chrom": "NC_000962.3",
        "pos": 761155,
        "ref": "C",
        "alt": "T",
        "annotation": "Ser450Leu (RNA polymerase beta subunit -> Rifampicin resistance)",
        "base_weight": 3.2,
        "freq_in_res": 0.58,
        "freq_in_susc": 0.01,
        "category": "Known High-Confidence Driver",
    },
    {
        "id": "inhA_promoter_C-15T",
        "gene": "fabG1-inhA",
        "chrom": "NC_000962.3",
        "pos": 1673425,
        "ref": "C",
        "alt": "T",
        "annotation": "c.-15C>T (Promoter mutation -> inhA target titration)",
        "base_weight": 2.1,
        "freq_in_res": 0.28,
        "freq_in_susc": 0.03,
        "category": "Known Moderate Driver",
    },
    {
        "id": "gyrA_D94G",
        "gene": "gyrA",
        "chrom": "NC_000962.3",
        "pos": 7582,
        "ref": "A",
        "alt": "G",
        "annotation": "Asp94Gly (DNA gyrase A -> Fluoroquinolone resistance)",
        "base_weight": 2.5,
        "freq_in_res": 0.35,
        "freq_in_susc": 0.02,
        "category": "Known Moderate Driver",
    },
    {
        "id": "novM_G214R_Efflux",
        "gene": "novM (Uncharacterized Membrane Protein)",
        "chrom": "NC_000962.3",
        "pos": 3891420,
        "ref": "G",
        "alt": "A",
        "annotation": "Gly214Arg (Transmembrane channel loop -> Putative efflux activation)",
        "base_weight": 2.9,
        "freq_in_res": 0.44,
        "freq_in_susc": 0.04,
        "category": "Novel Lab Candidate (Efflux Pump)",
    },
    {
        "id": "novM_L108P_TM2",
        "gene": "novM (Uncharacterized Membrane Protein)",
        "chrom": "NC_000962.3",
        "pos": 3891102,
        "ref": "T",
        "alt": "C",
        "annotation": "Leu108Pro (TM domain 2 helix destabilization)",
        "base_weight": 1.9,
        "freq_in_res": 0.22,
        "freq_in_susc": 0.03,
        "category": "Novel Lab Candidate (Membrane)",
    },
    {
        "id": "embB_M306V",
        "gene": "embB",
        "chrom": "NC_000962.3",
        "pos": 4247429,
        "ref": "A",
        "alt": "G",
        "annotation": "Met306Val (Arabinosyltransferase -> Ethambutol resistance)",
        "base_weight": 2.2,
        "freq_in_res": 0.31,
        "freq_in_susc": 0.03,
        "category": "Known Moderate Driver",
    },
    {
        "id": "pncA_H57D",
        "gene": "pncA",
        "chrom": "NC_000962.3",
        "pos": 2288920,
        "ref": "C",
        "alt": "G",
        "annotation": "His57Asp (Pyrazinamidase active site disruption)",
        "base_weight": 1.8,
        "freq_in_res": 0.25,
        "freq_in_susc": 0.02,
        "category": "Known Moderate Driver",
    },
]


def generate_synthetic_amr_cohort(
    n_samples: int = 600,
    n_passenger_snps: int = 150,
    resistance_ratio: float = 0.40,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generates a realistic clinical cohort.

    Returns:
    - X: pd.DataFrame (samples x SNPs binary matrix)
    - y: pd.Series (0 = Susceptible, 1 = Resistant)
    - metadata: pd.DataFrame (isolate lineage, country, resistance phenotype)
    """
    rng = np.random.default_rng(seed)

    # 1. Generate resistance labels with specified prevalence
    n_resistant = int(n_samples * resistance_ratio)
    n_susceptible = n_samples - n_resistant
    labels = np.array([1] * n_resistant + [0] * n_susceptible)
    rng.shuffle(labels)

    sample_ids = [f"MTB_CLIN_{i+1:04d}" for i in range(n_samples)]
    features = {}

    # 2. Simulate Key Biological Driver Mutations
    for locus in KEY_DRIVER_LOCI:
        col_name = locus["id"]
        # Probabilities conditional on resistance label
        p_res = locus["freq_in_res"]
        p_susc = locus["freq_in_susc"]

        calls = []
        for is_res in labels:
            prob = p_res if is_res == 1 else p_susc
            calls.append(1 if rng.random() < prob else 0)

        features[col_name] = calls

    # 3. Add Epistatic/Synergistic interaction:
    # NovM efflux mutation + GyrA or KatG amplifies resistance likelihood
    # If both novM_G214R and gyrA_D94G are present, clinical resistance is reinforced
    for i, is_res in enumerate(labels):
        if is_res == 1 and rng.random() < 0.25:
            features["novM_G214R_Efflux"][i] = 1
            features["gyrA_D94G"][i] = 1

    # 4. Simulate Background Passenger Mutations & Lineage Neutral SNPs
    for p_idx in range(n_passenger_snps):
        pos = int(rng.integers(10000, 4411532))
        ref_base = rng.choice(["A", "C", "G", "T"])
        alt_base = rng.choice([b for b in ["A", "C", "G", "T"] if b != ref_base])
        col_name = f"snp_pos_{pos}_{ref_base}>{alt_base}"

        # Neutral allele frequency drawn from beta distribution (most SNPs are low-frequency)
        af = rng.beta(0.8, 15.0)
        af = np.clip(af, 0.005, 0.40)

        calls = (rng.random(n_samples) < af).astype(int)
        features[col_name] = calls

    X = pd.DataFrame(features, index=sample_ids)
    X.index.name = "sample_id"
    y = pd.Series(labels, index=sample_ids, name="amr_phenotype")

    # 5. Metadata (Lineages, Country, Isolation Year)
    lineages = rng.choice(
        ["Lineage 4 (Euro-American)", "Lineage 2 (Beijing)", "Lineage 1 (Indo-Oceanic)", "Lineage 3 (East-African-Indian)"],
        size=n_samples,
        p=[0.45, 0.25, 0.18, 0.12],
    )
    years = rng.integers(2018, 2024, size=n_samples)

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "phenotype": ["Resistant" if v == 1 else "Susceptible" for v in labels],
            "lineage": lineages,
            "collection_year": years,
            "wgs_coverage_mean": rng.normal(85.0, 12.0, size=n_samples).round(1),
        }
    ).set_index("sample_id")

    return X, y, metadata


def save_cohort_dataset(
    output_dir: Path,
    n_samples: int = 600,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """
    Generates and saves the cohort to data/raw and data/processed.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = generate_synthetic_amr_cohort(n_samples=n_samples, seed=seed)

    # Save processed tabular datasets
    snp_matrix_path = processed_dir / "snp_matrix.csv"
    phenotypes_path = processed_dir / "phenotypes.csv"
    metadata_path = raw_dir / "clinical_metadata.csv"

    # Merge target for easy inspection
    full_df = X.copy()
    full_df.insert(0, "amr_phenotype", y)
    full_df.to_csv(snp_matrix_path)

    metadata.to_csv(metadata_path)
    y.to_frame().to_csv(phenotypes_path)

    # Also save annotations for the key loci
    annotations_df = pd.DataFrame(KEY_DRIVER_LOCI)
    annotations_df.to_csv(processed_dir / "driver_loci_annotation.csv", index=False)

    return snp_matrix_path, phenotypes_path, metadata_path


if __name__ == "__main__":
    base_data_dir = Path(__file__).resolve().parent.parent / "data"
    p1, p2, p3 = save_cohort_dataset(base_data_dir)
    print(f"Generated synthetic M. tuberculosis cohort successfully:")
    print(f"  - SNP Matrix: {p1}")
    print(f"  - Phenotypes: {p2}")
    print(f"  - Metadata:   {p3}")
