"""
GDSC Drug Response Prediction — 3-Stage Cancer-Specific Comparison
======================================================================
Per Astha's plan:
  Stage A: all 40 genes (original set)
  Stage B: cancer-specific priority subset (from her gene-mapping doc)
  Stage C: priority subset + missing genes (from her doc, new download)

Compares R2/RMSE/MAE across all 3 stages, for each of the 5 cancers, so we
can see whether gains come from biologically relevant gene selection rather
than just adding more genes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH,
)

EXPRESSION_49_PATH = "Expression_(Short-read)_Public_26Q1_subsetted (2).csv"

ORIGINAL_40 = [
    "BRCA1", "ROS1", "MLH1", "PGR", "ABCB1", "ESR1", "MSH2", "CDK6", "MET", "GATA3",
    "CCND1", "VEGFA", "STK11", "PIK3CA", "FOXA1", "TOP2A", "KRAS", "APC", "CDK4", "MDM2",
    "MYC", "BRCA2", "RB1", "TP53", "SMAD4", "ERBB2", "AKT1", "EGFR", "CDKN2A", "MKI67",
    "ATM", "BRAF", "ALK", "PTEN", "HRAS", "CHEK2", "NF1", "MTOR", "TOP1", "NRAS",
]

# From Astha's gene-mapping doc
CANCER_CONFIG = {
    "Breast": {
        "tcga": ["BRCA"],
        "priority": ["ESR1", "PGR", "ERBB2", "PIK3CA", "PTEN", "AKT1", "BRCA1",
                     "BRCA2", "CDK4", "CDK6", "CCND1", "FOXA1", "GATA3", "TP53"],
        "missing": ["CCNE1", "PALB2", "RAD51", "RAD51C", "RAD51D"],
    },
    "Lung/NSCLC": {
        "tcga": ["LUAD", "LUSC"],
        "priority": ["EGFR", "ALK", "ROS1", "MET", "KRAS", "BRAF", "STK11", "NF1",
                     "ERBB2", "PIK3CA", "PTEN", "AKT1", "MTOR", "TP53", "CDKN2A"],
        "missing": ["RET", "KEAP1"],
    },
    "Colorectal": {
        "tcga": ["COREAD"],
        "priority": ["APC", "KRAS", "BRAF", "EGFR", "PIK3CA", "PTEN", "AKT1",
                     "MTOR", "SMAD4", "MLH1", "MSH2", "TP53", "ERBB2"],
        "missing": ["CTNNB1"],
    },
    "Melanoma": {
        "tcga": ["SKCM"],
        "priority": ["BRAF", "NRAS", "NF1", "KRAS", "HRAS", "PTEN", "AKT1", "MTOR",
                     "CDK4", "CDK6", "CDKN2A", "CCND1", "MDM2", "TP53"],
        "missing": ["KIT"],
    },
    "Ovarian": {
        "tcga": ["OV"],
        "priority": ["BRCA1", "BRCA2", "TP53", "ATM", "CHEK2", "PIK3CA", "PTEN",
                     "AKT1", "MTOR", "MLH1", "MSH2", "VEGFA", "TOP1", "TOP2A"],
        "missing": ["CCNE1", "PALB2", "RAD51", "RAD51C", "RAD51D"],
    },
}


def load_expression_49(path):
    expr = pd.read_csv(path)
    meta_cols = {"depmap_id", "cell_line_display_name",
                 "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    gene_cols = [c for c in expr.columns if c not in meta_cols]
    expr["name_norm"] = expr["cell_line_display_name"].apply(normalize_name)
    expr = expr.drop_duplicates(subset="name_norm", keep="first")
    return expr, gene_cols


def build_subset(df, expr, tcga_codes, gene_cols):
    subset = df[df["TCGA_DESC"].isin(tcga_codes)].copy()
    subset["name_norm"] = subset["CELL_LINE_NAME"].apply(normalize_name)
    merged = subset.merge(expr[["name_norm"] + gene_cols], on="name_norm", how="inner")
    return merged


def train_and_eval(merged, gene_cols):
    all_features = CAT_FEATURES + gene_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in gene_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    if len(X) < 50:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    _, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return metrics


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, all_gene_cols = load_expression_49(EXPRESSION_49_PATH)
    print(f"Loaded GDSC: {len(df):,} rows | Expression: {len(expr):,} cell lines, {len(all_gene_cols)} genes\n")

    rows = []

    for cancer, cfg in CANCER_CONFIG.items():
        print("=" * 70)
        print(cancer)
        print("=" * 70)

        merged = build_subset(df, expr, cfg["tcga"], all_gene_cols)
        n_cell_lines = merged["CELL_LINE_NAME"].nunique()
        print(f"Matched cell lines: {n_cell_lines} | Rows: {len(merged):,}")

        stages = {
            "A: All 40 genes": ORIGINAL_40,
            "B: Priority subset": cfg["priority"],
            "C: Priority + missing": cfg["priority"] + cfg["missing"],
        }

        for stage_name, gene_list in stages.items():
            metrics = train_and_eval(merged, gene_list)
            if metrics is None:
                print(f"  {stage_name}: too few rows, skipped")
                continue
            print(f"  {stage_name} ({len(gene_list)} genes): "
                  f"R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}")
            rows.append({
                "Cancer": cancer, "Stage": stage_name, "NumGenes": len(gene_list),
                "CellLines": n_cell_lines, "Rows": len(merged),
                "R2": metrics["R2"], "RMSE": metrics["RMSE"], "MAE": metrics["MAE"],
            })
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv("cancer_3stage_summary.csv", index=False)
    print("=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nSaved to cancer_3stage_summary.csv")


if __name__ == "__main__":
    main()
