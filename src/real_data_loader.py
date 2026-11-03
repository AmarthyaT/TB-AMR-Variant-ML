"""
Real Clinical M. tuberculosis Genomic Dataset Loader
====================================================
Loads and processes 100% real clinical Whole Genome Sequencing (WGS) variant
matrices and phenotypic Drug Susceptibility Testing (DST) data from the
global NCBI SRA / BV-BRC tuberculosis benchmark (Saberian et al., ACM).

Matches:
- Real patient isolate IDs (e.g., SRR1166318, ERR176810, etc.)
- Real genomic SNP positions mapped to the M. tuberculosis H37Rv reference genome
- Real laboratory DST phenotypes (Isoniazid, Rifampicin, etc.)
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

BASE_URL = "https://raw.githubusercontent.com/AmirHoseinSafari/M.tuberculosis-dataset-for-drug-resistant/master"

URLS = {
    "labels": f"{BASE_URL}/AllLabels.csv",
    "isolates": f"{BASE_URL}/iso_list.csv",
    "sparse_snps": f"{BASE_URL}/sparsetableFeb27.npz",
    "snp_list": f"{BASE_URL}/SNPList.csv",
}

# Known clinical target genomic intervals on M. tb H37Rv (NC_000962.3)
GENE_COORDINATES = [
    ("katG", 2153889, 2156111, "Catalase-peroxidase (Isoniazid activation)"),
    ("rpoB", 759807, 763325, "RNA Polymerase Beta (Rifampicin binding)"),
    ("inhA", 1673000, 1675200, "Enoyl-ACP reductase / Promoter (Isoniazid target)"),
    ("gyrA", 7302, 9818, "DNA Gyrase Subunit A (Fluoroquinolones)"),
    ("pncA", 2288681, 2289241, "Pyrazinamidase (Pyrazinamide activation)"),
    ("embB", 4246514, 4249813, "Arabinosyltransferase (Ethambutol target)"),
    ("efpA", 3184000, 3187000, "Efflux Protein A (Membrane Efflux Pump)"),
    ("rv1819c_bacA", 2059000, 2063000, "ABC Transporter / Membrane Transport"),
]


def annotate_snp_coordinate(pos: int) -> Tuple[str, str]:
    """Annotates a genomic coordinate with its gene name and functional role."""
    for gene, start, end, desc in GENE_COORDINATES:
        if start <= pos <= end:
            return gene, f"{gene} ({desc})"
    return "Intergenic/Other", "Background Genomic Locus"


def download_file_if_missing(url: str, dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists() or dest_path.stat().st_size == 0:
        print(f"Downloading real dataset: {dest_path.name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())
        print(f"Saved: {dest_path.name} ({dest_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return dest_path


def load_snp_definitions(snp_list_file: Path) -> List[Tuple[int, str]]:
    """Loads all 742,620 real SNP definitions from SNPList.csv."""
    with open(snp_list_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        row = next(reader)
    items = row[1:]  # skip column 0 title

    definitions = []
    for item in items:
        parts = [p.strip() for p in item.split(",")]
        try:
            pos = int(parts[0])
            ref = parts[1] if len(parts) > 1 else "N"
            alt = parts[2] if len(parts) > 2 else "N"
            label = f"pos_{pos}_{ref}>{alt}"
        except Exception:
            pos = 0
            label = "unknown"
        definitions.append((pos, label))
    return definitions


def load_real_tb_cohort(
    data_dir: Path,
    target_drug: str = "isoniazid",
    max_isolates: int = 1000,
    min_maf: float = 0.02,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Loads real clinical M. tuberculosis isolates with phenotypic DST.
    
    Returns:
    - X: pd.DataFrame (samples x filtered real SNPs)
    - y: pd.Series (0 = Susceptible, 1 = Resistant)
    - metadata: pd.DataFrame (SRA IDs, phenotype, drug)
    - locus_annotations: pd.DataFrame (Locus, Gene, Description)
    """
    raw_dir = Path(data_dir) / "raw"
    processed_dir = Path(data_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    labels_file = raw_dir / "real_AllLabels.csv"
    isolates_file = raw_dir / "real_iso_list.csv"
    sparse_file = raw_dir / "real_sparsetableFeb27.npz"
    snp_list_file = raw_dir / "real_SNPList.csv"

    download_file_if_missing(URLS["labels"], labels_file)
    download_file_if_missing(URLS["isolates"], isolates_file)
    download_file_if_missing(URLS["sparse_snps"], sparse_file)
    download_file_if_missing(URLS["snp_list"], snp_list_file)

    # 1. Load isolate identifiers
    df_iso = pd.read_csv(isolates_file)
    iso_col = "name/ pos- ref- alt" if "name/ pos- ref- alt" in df_iso.columns else df_iso.columns[-1]
    isolate_ids = df_iso[iso_col].astype(str).str.strip().tolist()

    # 2. Load clinical DST labels
    df_labels = pd.read_csv(labels_file).set_index("id")

    drug_col = target_drug.lower()
    if drug_col not in [c.lower() for c in df_labels.columns]:
        drug_col = "isoniazid"

    # Filter to isolates with both sequencing and DST phenotypes
    valid_indices = []
    matched_ids = []
    phenotypes = []

    for idx, iso in enumerate(isolate_ids):
        if iso in df_labels.index:
            val = df_labels.loc[iso, drug_col]
            if pd.notnull(val) and val in [0.0, 1.0, 0, 1]:
                valid_indices.append(idx)
                matched_ids.append(iso)
                phenotypes.append(int(val))

    print(f"Matched {len(valid_indices)} real clinical isolates with {drug_col.upper()} DST phenotypes.")

    # Subsample if max_isolates requested
    rng = np.random.default_rng(random_seed)
    valid_indices = np.array(valid_indices)
    matched_ids = np.array(matched_ids)
    phenotypes = np.array(phenotypes)

    if len(valid_indices) > max_isolates:
        idx_res = np.where(phenotypes == 1)[0]
        idx_susc = np.where(phenotypes == 0)[0]
        ratio_res = len(idx_res) / len(phenotypes)
        n_res = int(max_isolates * ratio_res)
        n_susc = max_isolates - n_res

        sample_res = rng.choice(idx_res, size=n_res, replace=False)
        sample_susc = rng.choice(idx_susc, size=n_susc, replace=False)
        selected_sub = np.sort(np.concatenate([sample_res, sample_susc]))

        valid_indices = valid_indices[selected_sub]
        matched_ids = matched_ids[selected_sub]
        phenotypes = phenotypes[selected_sub]

    # 3. Load Sparse SNP Matrix
    print("Loading real sparse genomic variant matrix (coo_matrix)...")
    npz = np.load(sparse_file)
    row_idx = npz["row"]
    col_idx = npz["col"]
    data_vals = npz["data"]
    shape = tuple(npz["shape"])

    coo = coo_matrix((data_vals, (row_idx, col_idx)), shape=shape, dtype=np.uint8)
    csr = coo.tocsr()
    sub_csr = csr[valid_indices, :]

    # 4. Filter by Minor Allele Frequency
    col_counts = np.array(sub_csr.sum(axis=0)).flatten()
    n_samples = len(valid_indices)
    col_freq = col_counts / max(n_samples, 1)
    maf = np.minimum(col_freq, 1.0 - col_freq)

    informative_cols = np.where(maf >= min_maf)[0]
    print(f"Retained {len(informative_cols)} informative clinical SNPs (MAF >= {min_maf}).")

    sub_filtered = sub_csr[:, informative_cols].toarray()

    # 5. Map SNP definitions to real gene names
    print("Mapping real genomic coordinates and annotating gene targets...")
    snp_defs = load_snp_definitions(snp_list_file)

    feature_names = []
    annotations = []

    for c in informative_cols:
        if c < len(snp_defs):
            pos, label = snp_defs[c]
        else:
            pos, label = 0, f"SNP_locus_{c}"

        gene, role = annotate_snp_coordinate(pos)
        feat_name = f"{gene}_{label}" if gene != "Intergenic/Other" else label
        feature_names.append(feat_name)
        annotations.append({"feature": feat_name, "position": pos, "gene": gene, "description": role})

    X = pd.DataFrame(sub_filtered, index=matched_ids, columns=feature_names)
    X.index.name = "isolate_id"
    y = pd.Series(phenotypes, index=matched_ids, name="amr_phenotype")

    metadata = pd.DataFrame(
        {
            "isolate_id": matched_ids,
            "phenotype": ["Resistant" if v == 1 else "Susceptible" for v in phenotypes],
            "target_drug": drug_col.capitalize(),
            "data_source": "NCBI SRA / BV-BRC Clinical TB Isolates (Real)",
        }
    ).set_index("isolate_id")

    df_annot = pd.DataFrame(annotations).set_index("feature")

    # Cache to processed
    real_processed_file = processed_dir / f"real_cohort_{drug_col}.csv"
    merged = X.copy()
    merged.insert(0, "amr_phenotype", y)
    merged.to_csv(real_processed_file)
    print(f"Saved processed real cohort to: {real_processed_file}")

    return X, y, metadata, df_annot


if __name__ == "__main__":
    p_dir = Path(__file__).resolve().parent.parent / "data"
    X, y, meta, ann = load_real_tb_cohort(p_dir, target_drug="isoniazid", max_isolates=800, min_maf=0.025)
    print("Real Data X shape:", X.shape)
    print("Real Data y counts:", y.value_counts().to_dict())
    print("Sample features:", list(X.columns[:10]))
