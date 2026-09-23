# Datasets

## `gdsc_dataset.csv` (not included — see below)

The core drug-response dataset from **GDSC (Genomics of Drug Sensitivity in
Cancer)**. ~242,035 rows — 969 cancer cell lines tested against 286 drugs,
with response measured as LN_IC50, AUC, and Z-score. Also includes TCGA
cancer-type labels and Microsatellite Instability (MSI) status per cell
line.

Not committed to this repo (~36.5MB, easily re-downloaded). To get it:
- Source: https://www.cancerrxgene.org/downloads/bulk_download
- Download the drug sensitivity / cell line data export and place it here
  as `gdsc_dataset.csv`

## `Expression__Short-read__Public_26Q1_subsetted__3_.csv`

Gene expression values (RNA-seq, DepMap **Expression Short-read Public
26Q1**) for a curated 53-gene panel, across 1,719 cancer cell lines. Built
up over three iterations as candidate genes were added based on literature
review and per-cancer relevance (40 → 49 → 53 genes).

Source: https://depmap.org/portal/data_page/ — custom download using
"Use custom gene/compounds list" under the Expression dataset.

Full gene list (53): BRCA1, ROS1, MLH1, PGR, ABCB1, ESR1, MSH2, CDK6, MET,
GATA3, CCND1, VEGFA, STK11, PIK3CA, FOXA1, TOP2A, KRAS, APC, CDK4, MDM2,
MYC, BRCA2, RB1, TP53, SMAD4, ERBB2, AKT1, EGFR, CDKN2A, MKI67, ATM, BRAF,
ALK, PTEN, HRAS, CHEK2, NF1, MTOR, TOP1, NRAS, CTNNB1, CCNE1, PALB2, RAD51,
RAD51C, RAD51D, RET, KEAP1, KIT, AREG, EREG, MITF, SOX10

## `mutation_data.csv`

BRAF, KRAS, and NRAS mutation status (Hotspot Mutations + Damaging
Mutations, DepMap **Public 26Q1**) for 1,968 cell lines. Used to test
whether mutation status carries signal that gene expression alone
doesn't capture, for Colorectal and Melanoma specifically.

Source: same DepMap portal, "Mutations" section — Hotspot Mutations
(Public 26Q1) and Damaging Mutations (Public 26Q1), custom gene list
(BRAF, KRAS, NRAS).

## Matching cell lines across datasets

GDSC and DepMap name cell lines inconsistently (e.g. `SK-ES-1` vs
`SKES1`). All merges in this project normalize names first — stripping
non-alphanumeric characters and uppercasing — before joining. See
`normalize_name()` in `models/gdsc_expression_model.py`.
