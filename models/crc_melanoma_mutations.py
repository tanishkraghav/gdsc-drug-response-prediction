"""
Test: BRAF/KRAS/NRAS mutation status for Colorectal and Melanoma
=====================================================================
Astha's hypothesis: expression of BRAF/KRAS/NRAS didn't show much signal
for CRC/melanoma, but mutation status of these genes (which drive these
cancers clinically) might carry complementary information that expression
alone doesn't capture.

Mutation data: DepMap Hotspot Mutations + Damaging Mutations (Public 26Q1),
for BRAF/KRAS/NRAS. Merged onto the Stage A baseline (all 40 expression
genes) as additional numeric features (0/1/2 = number of mutated alleles
detected).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gdsc_expression_model import (
    normalize_name, encode_categoricals, train_xgb,
    CAT_FEATURES, TARGET_COL, RANDOM_STATE, GDSC_PATH,
)
from cancer_3stage import build_subset, ORIGINAL_40
from crc_melanoma_new_genes import load_expression_53, EXPRESSION_53_PATH

MUTATION_PATH = "mutation_data.csv"
MUTATION_COLS = [
    "Hotspot Mutations (Public 26Q1) NRAS",
    "Hotspot Mutations (Public 26Q1) KRAS",
    "Hotspot Mutations (Public 26Q1) BRAF",
    "Damaging Mutations (Public 26Q1) NRAS",
    "Damaging Mutations (Public 26Q1) KRAS",
    "Damaging Mutations (Public 26Q1) BRAF",
]


def load_mutations(path):
    mut = pd.read_csv(path)
    mut["name_norm"] = mut["cell_line_display_name"].apply(normalize_name)
    mut = mut.drop_duplicates(subset="name_norm", keep="first")
    return mut


def train_and_eval(merged, gene_cols, mutation_cols=None):
    numeric_cols = gene_cols + (mutation_cols or [])
    all_features = CAT_FEATURES + numeric_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, metrics = train_xgb(X_train, y_train, X_test, y_test, n_estimators=150)
    return model, metrics, all_features


def run_test(cancer_name, tcga_codes, df, expr, all_gene_cols, mut):
    print("=" * 70)
    print(cancer_name)
    print("=" * 70)

    merged = build_subset(df, expr, tcga_codes, all_gene_cols)
    merged = merged.merge(mut[["name_norm"] + MUTATION_COLS], on="name_norm", how="left")
    n_with_mutation_data = merged[MUTATION_COLS[0]].notna().sum()
    print(f"Cell lines: {merged['CELL_LINE_NAME'].nunique()} | Rows: {len(merged):,} | "
          f"Rows with mutation data: {n_with_mutation_data:,}\n")

    # Baseline: all 40 expression genes only
    _, metrics_base, _ = train_and_eval(merged, ORIGINAL_40)
    print(f"Baseline (expression only):     R2={metrics_base['R2']:.4f}  RMSE={metrics_base['RMSE']:.4f}")

    # With mutation status added
    model, metrics_mut, all_features = train_and_eval(merged, ORIGINAL_40, MUTATION_COLS)
    print(f"With BRAF/KRAS/NRAS mutations:   R2={metrics_mut['R2']:.4f}  RMSE={metrics_mut['RMSE']:.4f}")

    delta = metrics_mut['R2'] - metrics_base['R2']
    print(f"Delta: {delta:+.4f}")

    importances = sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1])
    print(f"\nWhere do mutation features rank?")
    for col in MUTATION_COLS:
        rank = next(i for i, (f, _) in enumerate(importances) if f == col) + 1
        imp = dict(importances)[col]
        print(f"  {col}: importance={imp:.4f}, rank #{rank} of {len(all_features)}")

    return {
        "Cancer": cancer_name, "R2_baseline": metrics_base["R2"],
        "R2_with_mutations": metrics_mut["R2"], "Delta": delta,
    }


def main():
    df = pd.read_csv(GDSC_PATH)
    expr, all_gene_cols = load_expression_53(EXPRESSION_53_PATH)
    mut = load_mutations(MUTATION_PATH)
    print(f"Loaded GDSC: {len(df):,} rows | Expression: {len(expr):,} cell lines | "
          f"Mutation data: {len(mut):,} cell lines\n")

    results = []
    results.append(run_test("Colorectal", ["COREAD"], df, expr, all_gene_cols, mut))
    print()
    results.append(run_test("Melanoma", ["SKCM"], df, expr, all_gene_cols, mut))

    summary = pd.DataFrame(results)
    summary.to_csv("crc_melanoma_mutation_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
