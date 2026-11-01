"""
Feature Engineering & Genomic Preprocessing
============================================
Performs quality filtering and encoding on genomic variant data:
- Minor Allele Frequency (MAF) filtering (removing ultra-rare singleton noise).
- Variance threshold filtering.
- K-mer frequency representation for raw FASTA sequence inputs.
- Stratified train-test splitting.
"""

from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_by_maf(df_snps: pd.DataFrame, min_maf: float = 0.01) -> pd.DataFrame:
    """
    Filters out SNP features with Minor Allele Frequency (MAF) below threshold.
    Helps eliminate uninformative singleton variants and reduces genomic dimensionality.
    """
    allele_frequencies = df_snps.mean(axis=0)
    # Since binary 0/1: MAF is min(p, 1-p)
    maf = np.minimum(allele_frequencies, 1.0 - allele_frequencies)
    keep_cols = maf[maf >= min_maf].index
    return df_snps[keep_cols]


def extract_kmers_from_sequence(seq: str, k: int = 5) -> Dict[str, int]:
    """
    Extracts k-mer frequency counts from a biological nucleotide sequence.
    Useful when raw FASTA genomic reads/genes are provided instead of pre-called SNPs.
    """
    seq = seq.upper()
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    return dict(Counter(kmers))


def prepare_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    min_maf: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans features and performs stratified train/test split.
    """
    # Filter features based on training variance / MAF
    X_filtered = filter_by_maf(X, min_maf=min_maf)

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
