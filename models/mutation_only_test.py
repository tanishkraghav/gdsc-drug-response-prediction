"""
Mutation-only test: BRAF/KRAS/NRAS mutation status, NO expression features
================================================================================
Isolates how much signal mutation status carries entirely on its own for
Colorectal and Melanoma, without 40+ expression genes diluting its
contribution. Compares against:
  - Categorical-only baseline (drug/cell line/pathway, no biology at all)
  - Full expression baseline (Stage A, all 40 genes)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH,
)
from cancer_3stage import build_subset, ORIGINAL_40
from crc_melanoma_new_genes import load_expression_53, EXPRESSION_53_PATH
from crc_melanoma_mutations import load_mutations, MUTATION_PATH, MUTATION_COLS

STAGE_A_R2 = {"Colorectal": 0.7602, "Melanoma": 0.7466}


def train_and_eval(merged, feature_cols):
    data = merged[feature_cols + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    numeric_cols = [c for c in feature_cols if c not in CAT_FEATURES]
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[feature_cols], data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return model, metrics


def run_test(cancer_name, tcga_codes, df, expr, all_gene_cols, mut):
    print("=" * 70)
    print(cancer_name)
    print("=" * 70)

    merged = build_subset(df, expr, tcga_codes, all_gene_cols)
    merged = merged.merge(mut[["name_norm"] + MUTATION_COLS], on="name_norm", how="left")

    # Categorical-only (no biology at all)
    _, metrics_cat = train_and_eval(merged, CAT_FEATURES)
    print(f"Categorical only (no biology):     R2={metrics_cat['R2']:.4f}")

    # Mutation-only (categorical + mutation status, NO expression genes)
    _, metrics_mut_only = train_and_eval(merged, CAT_FEATURES + MUTATION_COLS)
    print(f"Mutation status only (no expr):    R2={metrics_mut_only['R2']:.4f}")

    # Reference: full expression baseline (Stage A, from earlier run)
    print(f"Full expression baseline (Stage A): R2={STAGE_A_R2[cancer_name]:.4f}  (reference, not re-run here)")

    print(f"\nMutation-only vs categorical-only: {metrics_mut_only['R2'] - metrics_cat['R2']:+.4f}")
    print(f"Mutation-only vs full expression:   {metrics_mut_only['R2'] - STAGE_A_R2[cancer_name]:+.4f}")

    return {
        "Cancer": cancer_name,
        "R2_categorical_only": metrics_cat["R2"],
        "R2_mutation_only": metrics_mut_only["R2"],
        "R2_full_expression_baseline": STAGE_A_R2[cancer_name],
    }


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, all_gene_cols = load_expression_53(EXPRESSION_53_PATH)
    mut = load_mutations(MUTATION_PATH)

    results = []
    results.append(run_test("Colorectal", ["COREAD"], df, expr, all_gene_cols, mut))
    print()
    results.append(run_test("Melanoma", ["SKCM"], df, expr, all_gene_cols, mut))

    summary = pd.DataFrame(results)
    summary.to_csv("mutation_only_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
