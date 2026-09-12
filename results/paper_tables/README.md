# Thesis result tables and data

This directory contains the result evidence presented in the submitted
thesis. It is intentionally limited to the three paper datasets and does not
contain raw observations.

## Table map

| Thesis table | Dataset / comparison | Table image | Structured result data |
|---|---|---|---|
| Table 1 | Variables and target definitions | `thesis_table_01_variables.png` | — |
| Tables 2–4 | RSW Gun: fixed 0.5, F2-only, and OMMA comparisons | `thesis_table_02`–`04` | `rsw_gun_test_results.csv` |
| Tables 5–7 | WM811K_01: fixed 0.5, F2-only, and OMMA comparisons | `thesis_table_05`–`07` | `wm811k_test_results.csv` |
| Tables 8–10 | CNC Milling Tool Life 03: fixed 0.5, F2-only, and OMMA comparisons | `thesis_table_08`–`10` | `cnc_milling_tool_life_test_results.csv` |

The PNG files are the table images embedded in the submitted thesis PDF. The
CSV files contain the corresponding held-out test results for each model and
policy, including threshold, Precision, Recall, F1, F2, G-mean, FPR, and
confusion-matrix counts (`TP`, `FP`, `FN`, `TN`).

The CSV files are result summaries, not the underlying raw datasets. To rerun
the experiments, keep the three raw datasets outside Git and follow
[`docs/data_sources.md`](../../docs/data_sources.md).

## Interpretation guide

- **RSW Gun:** The proposed policy improves the balance between defect
  detection and normal-class preservation. The thesis highlights CatBoost with
  Recall 0.959, F2 0.733, and G-mean 0.954 in the OMMA comparison.
- **WM811K_01:** The proposed policy lowers the threshold and generally trades
  some Precision for higher Recall and G-mean. This is why the paper reads F2
  and G-mean together with Precision rather than treating Recall alone as the
  objective.
- **CNC Milling Tool Life 03:** The proposed sequential policy improves the
  defect-sensitive metrics over the fixed threshold and is reported to exceed
  OMMA across the compared metrics and base models.

[Table 1 — variables and target definitions](thesis_table_01_variables.png)

## RSW Gun

- [Table 2 — fixed 0.5 vs proposed](thesis_table_02_rsw_gun_05_vs_proposed.png)
- [Table 3 — F2-only vs proposed](thesis_table_03_rsw_gun_f2_vs_proposed.png)
- [Table 4 — OMMA vs proposed](thesis_table_04_rsw_gun_omma_vs_proposed.png)
- [Structured test results](rsw_gun_test_results.csv)

## WM811K

- [Table 5 — fixed 0.5 vs proposed](thesis_table_05_wm811k_05_vs_proposed.png)
- [Table 6 — F2-only vs proposed](thesis_table_06_wm811k_f2_vs_proposed.png)
- [Table 7 — OMMA vs proposed](thesis_table_07_wm811k_omma_vs_proposed.png)
- [Structured test results](wm811k_test_results.csv)

## CNC Milling Tool Life

- [Table 8 — fixed 0.5 vs proposed](thesis_table_08_cnc_milling_tool_life_05_vs_proposed.png)
- [Table 9 — F2-only vs proposed](thesis_table_09_cnc_milling_tool_life_f2_vs_proposed.png)
- [Table 10 — OMMA vs proposed](thesis_table_10_cnc_milling_tool_life_omma_vs_proposed.png)
- [Structured test results](cnc_milling_tool_life_test_results.csv)
