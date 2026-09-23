"""
GDSC Drug Response Prediction — Baseline + Gene Expression Model
==================================================================
Compares two models for predicting drug sensitivity (LN_IC50):

1. Categorical-only baseline: cell line, cancer type, drug, pathway, tissue
2. Categorical + gene expression: adds 40 cancer-relevant gene expression
   values (DepMap Expression Short-read Public 26Q1), merged in by matching
   normalized cell line names between GDSC and DepMap.

Files needed in the same folder (or update the paths below):
- gdsc_dataset.csv
- Expression_(Short-read)_Public_26Q1_subsetted (1).csv
"""

import re
import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

RANDOM_STATE = 42
GDSC_PATH = "gdsc_dataset.csv"
EXPRESSION_PATH = "Expression_(Short-read)_Public_26Q1_subsetted (1).csv"

CAT_FEATURES = [
    "CELL_LINE_NAME", "TCGA_DESC", "DRUG_NAME", "TARGET_PATHWAY",
    "GDSC Tissue descriptor 1", "GDSC Tissue descriptor 2",
]
TARGET_COL = "LN_IC50"


def normalize_name(name):
    """Strip everything but letters/digits and uppercase, so 'SK-ES-1' and
    'SKES1' match — cell line naming is inconsistent across databases."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def encode_categoricals(data, cat_cols):
    for col in cat_cols:
        data[col] = data[col].fillna("Unknown").astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    data[cat_cols] = encoder.fit_transform(data[cat_cols])
    return data


def train_xgb(X_train, y_train, X_test, y_test, n_estimators=150):
    """Single-threaded, hist-method XGBoost — fast even on 1 CPU core."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Trained in {time.time() - t0:.1f}s")

    metrics = {
        "RMSE": root_mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }
    return model, metrics


def run_categorical_baseline(df):
    print("\n" + "=" * 60)
    print("PART 1: Categorical-only baseline")
    print("=" * 60)

    data = df[CAT_FEATURES + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)

    X, y = data[CAT_FEATURES], data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training on {len(X_train):,} rows, {len(CAT_FEATURES)} features...")
    model, metrics = train_xgb(X_train, y_train, X_test, y_test)

    print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    return model, metrics


def load_expression(path):
    expr = pd.read_csv(path)
    meta_cols = {"depmap_id", "cell_line_display_name",
                 "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    gene_cols = [c for c in expr.columns if c not in meta_cols]
    expr["name_norm"] = expr["cell_line_display_name"].apply(normalize_name)
    expr = expr.drop_duplicates(subset="name_norm", keep="first")
    return expr, gene_cols


def run_expression_model(df, expr, gene_cols):
    print("\n" + "=" * 60)
    print("PART 2: Categorical + Gene Expression")
    print("=" * 60)
    print(f"Expression data: {len(expr):,} cell lines, {len(gene_cols)} genes")

    df = df.copy()
    df["name_norm"] = df["CELL_LINE_NAME"].apply(normalize_name)
    merged = df.merge(expr[["name_norm"] + gene_cols], on="name_norm", how="inner")
    print(f"Rows after merging: {len(merged):,} "
          f"(dropped {len(df) - len(merged):,} rows with no expression match)")

    all_features = CAT_FEATURES + gene_cols
    data = merged[all_features + [TARGET_COL]].copy()
    data = encode_categoricals(data, CAT_FEATURES)
    for col in gene_cols:
        data[col] = data[col].fillna(data[col].mean())

    X, y = data[all_features], data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training on {len(X_train):,} rows, {len(all_features)} features...")
    model, metrics = train_xgb(X_train, y_train, X_test, y_test)

    print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

    importances = sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 15 features:")
    for col, imp in importances[:15]:
        tag = " (gene)" if col in gene_cols else ""
        print(f"  {col}{tag}: {imp:.4f}")

    return model, metrics, len(merged)


def main():
    df = pd.read_csv(GDSC_PATH)
    print(f"Loaded GDSC: {df.shape[0]:,} rows, {df.shape[1]} columns")

    _, cat_metrics = run_categorical_baseline(df)

    expr, gene_cols = load_expression(EXPRESSION_PATH)
    _, expr_metrics, n_merged = run_expression_model(df, expr, gene_cols)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Categorical-only : R2={cat_metrics['R2']:.4f}  (n={len(df):,} rows)")
    print(f"With expression  : R2={expr_metrics['R2']:.4f}  (n={n_merged:,} rows, "
          f"{len(expr):,} cell lines had expression data)")


if __name__ == "__main__":
    main()
