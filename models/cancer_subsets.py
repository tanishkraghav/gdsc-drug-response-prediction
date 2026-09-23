"""
GDSC Drug Response Prediction — Cancer-Specific Subsets (Stage 1)
====================================================================
Astha's plan: current 40-gene model -> 5 cancer-specific subsets -> compare
R2/RMSE/MAE, before testing whether adding missing genes improves further.

Cancer type mapping (TCGA_DESC codes in GDSC):
- Breast    -> BRCA
- Lung/NSCLC-> LUAD + LUSC (NSCLC = non-small-cell; SCLC is a different
               disease biologically and is excluded)
- Colorectal-> COREAD
- Melanoma  -> SKCM
- Ovarian   -> OV

For each cancer: reports matched cell lines, response rows, and drug
compounds after filtering (as Astha asked), then trains the same
expression-enriched XGBoost model per subset and reports R2/RMSE/MAE.

Common genes (TP53, PIK3CA, PTEN, AKT1, MTOR) are kept in every subset per
Astha's note, since they matter across cancers even if not cancer-specific.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb, load_expression,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH, EXPRESSION_PATH,
)

CANCER_TCGA_MAP = {
    "Breast": ["BRCA"],
    "Lung/NSCLC": ["LUAD", "LUSC"],
    "Colorectal": ["COREAD"],
    "Melanoma": ["SKCM"],
    "Ovarian": ["OV"],
}


def build_cancer_subset(df, expr, gene_cols, tcga_codes):
    subset = df[df["TCGA_DESC"].isin(tcga_codes)].copy()
    subset["name_norm"] = subset["CELL_LINE_NAME"].apply(normalize_name)
    merged = subset.merge(expr[["name_norm"] + gene_cols], on="name_norm", how="inner")
    return merged


def train_on_subset(merged, gene_cols):
    all_features = CAT_FEATURES + gene_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in gene_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    if len(X) < 50:
        return None, None  # too small to split meaningfully

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return model, metrics


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, gene_cols = load_expression(EXPRESSION_PATH)
    print(f"Loaded GDSC: {len(df):,} rows | Expression: {len(expr):,} cell lines, {len(gene_cols)} genes\n")

    summary_rows = []

    for cancer, tcga_codes in CANCER_TCGA_MAP.items():
        print("=" * 60)
        print(f"{cancer}  (TCGA: {', '.join(tcga_codes)})")
        print("=" * 60)

        merged = build_cancer_subset(df, expr, gene_cols, tcga_codes)
        n_cell_lines = merged["CELL_LINE_NAME"].nunique()
        n_rows = len(merged)
        n_drugs = merged["DRUG_NAME"].nunique()

        print(f"Matched cell lines: {n_cell_lines}")
        print(f"Response rows:      {n_rows:,}")
        print(f"Drug compounds:     {n_drugs}")

        if n_rows < 50:
            print("Too few rows to train a meaningful model — skipping.\n")
            summary_rows.append({
                "Cancer": cancer, "CellLines": n_cell_lines, "Rows": n_rows,
                "Drugs": n_drugs, "R2": None, "RMSE": None, "MAE": None,
            })
            continue

        model, metrics = train_on_subset(merged, gene_cols)
        print(f"R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}\n")

        summary_rows.append({
            "Cancer": cancer, "CellLines": n_cell_lines, "Rows": n_rows,
            "Drugs": n_drugs, "R2": metrics["R2"], "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
        })

    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("cancer_subset_summary.csv", index=False)
    print("\nSaved to cancer_subset_summary.csv")


if __name__ == "__main__":
    main()
