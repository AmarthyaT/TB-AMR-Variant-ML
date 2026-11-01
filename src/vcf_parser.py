"""
VCF Parser & Variant Matrix Builder
===================================
Parses Variant Call Format (VCF) files and variant call tables into
a binary feature matrix (Samples x SNP Mutation Loci) suitable for
Machine Learning classifiers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def parse_vcf_line(line: str) -> Optional[Tuple[str, str, str, List[str]]]:
    """
    Parse a single VCF data line.
    Returns (chrom, pos, ref, alt_list).
    """
    if line.startswith("#") or not line.strip():
        return None
    fields = line.strip().split("\t")
    if len(fields) < 8:
        return None
    chrom = fields[0]
    pos = fields[1]
    ref = fields[3]
    alt = fields[4].split(",")
    return chrom, pos, ref, alt


def parse_multisample_vcf(vcf_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parses a multi-sample VCF into a binary SNP matrix.
    Rows = Sample IDs
    Columns = Variant identifiers (e.g., 'CHROM_POS_REF_ALT')
    Values = 1 (variant present) or 0 (wild-type/reference)
    """
    vcf_path = Path(vcf_path)
    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    sample_names = []
    variant_records = []  # will store (var_id, sample_calls)

    with open(vcf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_parts = line.strip().split("\t")
                if len(header_parts) > 9:
                    sample_names = header_parts[9:]
                else:
                    sample_names = ["Sample_01"]
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0]
            pos = parts[1]
            ref = parts[3]
            alts = parts[4].split(",")
            format_col = parts[8] if len(parts) > 8 else "GT"
            sample_data = parts[9:] if len(parts) > 9 else []

            gt_idx = 0
            format_keys = format_col.split(":")
            if "GT" in format_keys:
                gt_idx = format_keys.index("GT")

            for alt_allele in alts:
                var_id = f"{chrom}_{pos}_{ref}_{alt_allele}"
                calls = []
                for s in sample_data:
                    subfields = s.split(":")
                    gt = subfields[gt_idx] if len(subfields) > gt_idx else "0"
                    # In bacterial haploid genomics, calls are typically 0 or 1, or 0/0, 1/1
                    is_mutant = 1 if ("1" in gt or "2" in gt) else 0
                    calls.append(is_mutant)

                if sample_names and len(calls) == len(sample_names):
                    variant_records.append((var_id, calls))

    if not sample_names or not variant_records:
        return pd.DataFrame()

    var_ids = [r[0] for r in variant_records]
    matrix = np.array([r[1] for r in variant_records]).T  # shape: (n_samples, n_variants)

    df = pd.DataFrame(matrix, index=sample_names, columns=var_ids)
    df.index.name = "sample_id"
    return df


def write_sample_vcf(output_path: Union[str, Path], df_matrix: pd.DataFrame) -> None:
    """
    Writes a binary SNP matrix back to a standard mock VCF format
    for testing and benchmarking interoperability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(df_matrix.index)
    columns = list(df_matrix.columns)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=TB-AMR-Variant-ML\n")
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        
        header_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
        f.write("\t".join(header_cols) + "\n")

        for col in columns:
            parts = col.split("_")
            if len(parts) >= 4:
                chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
            else:
                chrom, pos, ref, alt = "NC_000962.3", "1000", "A", "G"
            
            genotypes = []
            for s in samples:
                val = df_matrix.loc[s, col]
                gt = "1/1" if val == 1 else "0/0"
                genotypes.append(gt)

            row = [chrom, str(pos), col, ref, alt, "100", "PASS", "AF=0.5", "GT"] + genotypes
            f.write("\t".join(row) + "\n")
