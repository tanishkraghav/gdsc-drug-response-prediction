"""
Test: AREG/EREG for Colorectal, MITF/SOX10 for Melanoma
============================================================
Astha's candidates, tested one biological-info-type-at-a-time as agreed:
- AREG, EREG: associated with response to EGFR-targeted treatment
  (e.g. cetuximab) in CRC literature
- MITF, SOX10: melanocytic lineage/biology markers for melanoma

Each is added on top of the Stage A baseline (all 40 genes) for its
respective cancer, so we can isolate whether these specific genes help.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH,
)
from cancer_3stage import build_subset, ORIGINAL_40

EXPRESSION_53_PATH = "Expression_(Short-read)_Public_26Q1_subsetted (3).csv"


def load_expression_53(path):
    expr = pd.read_csv(path)
    meta_cols = {"depmap_id", "cell_line_display_name",
                 "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    gene_cols = [c for c in expr.columns if c not in meta_cols]
    expr["name_norm"] = expr["cell_line_display_name"].apply(normalize_name)
    expr = expr.drop_duplicates(subset="name_norm", keep="first")
    return expr, gene_cols


def train_and_eval(merged, gene_cols):
    all_features = CAT_FEATURES + gene_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in gene_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return model, metrics, all_features


def run_test(cancer_name, tcga_codes, new_genes, df, expr, all_gene_cols):
    print("=" * 70)
    print(f"{cancer_name}  (testing: {', '.join(new_genes)})")
    print("=" * 70)

    merged = build_subset(df, expr, tcga_codes, all_gene_cols)
    print(f"Cell lines: {merged['CELL_LINE_NAME'].nunique()} | Rows: {len(merged):,}\n")

    # Baseline: all 40 genes, no new genes
    _, metrics_base, _ = train_and_eval(merged, ORIGINAL_40)
    print(f"Baseline (all 40):        R2={metrics_base['R2']:.4f}  RMSE={metrics_base['RMSE']:.4f}")

    # With new genes added
    model, metrics_new, all_features = train_and_eval(merged, ORIGINAL_40 + new_genes)
    print(f"With {'/'.join(new_genes)} added: R2={metrics_new['R2']:.4f}  RMSE={metrics_new['RMSE']:.4f}")

    delta = metrics_new['R2'] - metrics_base['R2']
    print(f"Delta: {delta:+.4f}")

    importances = sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1])
    print(f"\nWhere do the new genes rank?")
    for gene in new_genes:
        rank = next(i for i, (f, _) in enumerate(importances) if f == gene) + 1
        imp = dict(importances)[gene]
        print(f"  {gene}: importance={imp:.4f}, rank #{rank} of {len(all_features)}")

    return {
        "Cancer": cancer_name, "NewGenes": "/".join(new_genes),
        "R2_baseline": metrics_base["R2"], "R2_with_new": metrics_new["R2"],
        "Delta": delta,
    }


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, all_gene_cols = load_expression_53(EXPRESSION_53_PATH)
    print(f"Loaded GDSC: {len(df):,} rows | Expression: {len(expr):,} cell lines\n")

    results = []
    results.append(run_test("Colorectal", ["COREAD"], ["AREG", "EREG"], df, expr, all_gene_cols))
    print()
    results.append(run_test("Melanoma", ["SKCM"], ["MITF", "SOX10"], df, expr, all_gene_cols))

    summary = pd.DataFrame(results)
    summary.to_csv("crc_melanoma_new_genes_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
