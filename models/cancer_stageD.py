"""
Stage D: All 40 genes + missing genes (additive — nothing removed)
=====================================================================
Tests whether the R2 drop seen for Colorectal/Melanoma in the priority-subset
and priority+missing runs was caused by REMOVING genes (fewer features overall)
rather than the missing genes themselves being unhelpful.

Also prints feature importance for Colorectal and Melanoma specifically, to
see what's actually driving predictions there vs Breast/Lung.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH,
)
from cancer_3stage import (
    load_expression_49, build_subset, ORIGINAL_40, CANCER_CONFIG,
    EXPRESSION_49_PATH,
)


def train_and_eval_with_model(merged, gene_cols):
    all_features = CAT_FEATURES + gene_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in gene_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    if len(X) < 50:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return model, metrics


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, all_gene_cols = load_expression_49(EXPRESSION_49_PATH)
    print(f"Loaded GDSC: {len(df):,} rows | Expression: {len(expr):,} cell lines\n")

    rows = []
    importance_report = {}

    for cancer, cfg in CANCER_CONFIG.items():
        print("=" * 70)
        print(cancer)
        print("=" * 70)

        merged = build_subset(df, expr, cfg["tcga"], all_gene_cols)
        n_cell_lines = merged["CELL_LINE_NAME"].nunique()

        # Stage D: all 40 + missing, nothing removed
        stage_d_genes = ORIGINAL_40 + cfg["missing"]
        # de-dup while preserving order, in case of overlap
        stage_d_genes = list(dict.fromkeys(stage_d_genes))

        model, metrics = train_and_eval_with_model(merged, stage_d_genes)
        if metrics is None:
            print("  Too few rows, skipped")
            continue

        print(f"  D: All 40 + missing ({len(stage_d_genes)} genes): "
              f"R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}")
        rows.append({
            "Cancer": cancer, "Stage": "D: All 40 + missing (additive)",
            "NumGenes": len(stage_d_genes), "CellLines": n_cell_lines,
            "Rows": len(merged), "R2": metrics["R2"], "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
        })

        if cancer in ("Colorectal", "Melanoma"):
            importances = sorted(
                zip(CAT_FEATURES + stage_d_genes, model.feature_importances_),
                key=lambda x: -x[1]
            )
            importance_report[cancer] = importances[:15]

    summary = pd.DataFrame(rows)
    summary.to_csv("cancer_stageD_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("STAGE D SUMMARY (compare to Stage A/B/C in cancer_3stage_summary.csv)")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — Colorectal & Melanoma (Stage D model)")
    print("=" * 70)
    for cancer, importances in importance_report.items():
        print(f"\n{cancer}:")
        for feat, imp in importances:
            is_gene = feat not in CAT_FEATURES
            print(f"  {feat}{' (gene)' if is_gene else ''}: {imp:.4f}")

    print("\nSaved to cancer_stageD_summary.csv")


if __name__ == "__main__":
    main()
