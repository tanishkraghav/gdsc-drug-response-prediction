# GDSC Drug Response Prediction

Predicting cancer cell-line drug sensitivity (LN_IC50) from gene expression,
mutation status, and clinical markers — across five cancer types (breast,
lung/NSCLC, colorectal, melanoma, ovarian).

**[Live dashboard →](https://incredible-bavarois-510e17.netlify.app/)**

A collaborative bioinformatics + ML project: cancer-wise gene mapping,
dataset sourcing, and biological interpretation on one side; feature
engineering, model training, and experiment design on the other.

## Structure

```
├── datasets/     # source data + README on where to get it
├── models/       # all model/experiment scripts (Python, XGBoost)
├── results/      # output CSVs from each experiment
└── dashboard/    # the published HTML dashboard
```

## Summary of findings

- Baseline model (categorical features only, drug + cell line + pathway):
  **R² = 0.78** across all 242K rows
- Adding 40-gene expression data: **R² = 0.78–0.79** depending on cancer type
- Breast, Lung/NSCLC, and Ovarian responded well to curated/expanded gene
  sets — best results up to **R² = 0.79**
- Colorectal and Melanoma plateaued regardless of additional expression
  genes; MSI status gave a small real gain for Colorectal; mutation status
  (BRAF/KRAS/NRAS) showed strong *individual* feature importance for both
  but didn't lift the overall model — suggesting expression and mutation
  carry complementary, not additive, information for these two cancers

See `results/full_experiment_summary.csv` and the dashboard for the full
breakdown.

## Running the models

```bash
pip install pandas scikit-learn xgboost

cd models
python gdsc_expression_model.py   # full-dataset baseline + expression
python cancer_3stage.py           # per-cancer gene-selection comparison
python cancer_stageD.py           # additive gene test
python crc_msi_test.py            # MSI status for colorectal
python crc_melanoma_new_genes.py  # AREG/EREG, MITF/SOX10
python crc_melanoma_mutations.py  # BRAF/KRAS/NRAS mutation status
python full_summary.py            # consolidates all results
```

Each script expects the datasets described in `datasets/README.md` to be
present in the same folder (or update the path constants at the top of
each script).

## Datasets

- [GDSC](https://www.cancerrxgene.org/) — drug sensitivity data
- [DepMap](https://depmap.org/portal/) — gene expression (Short-read
  Public 26Q1) and mutation data (Hotspot/Damaging Mutations Public 26Q1)

See `datasets/README.md` for exact download instructions and the full
gene list used.
