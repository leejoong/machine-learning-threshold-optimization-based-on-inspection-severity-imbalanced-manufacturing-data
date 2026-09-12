# Results interpretation

The tracked tables summarize the three datasets used in the thesis: RSW Gun,
WM811K, and CNC Milling Tool Life. The repository does not redraw or extend the
paper's results. Its visual gallery is limited to the two thesis figures that
explain the procedure and the evaluation language.

## What the tables test

The experiments compare the proposed state-dependent threshold policy with a
fixed 0.5 threshold, a validation F2-only threshold, and OMMA. XGBoost,
LightGBM, and CatBoost are evaluated under the same policy comparisons. The
main performance table reports Precision, Recall, F1, F2, and G-mean; the other
tables record risk concentration, bootstrap intervals, and the FN:FP = 10:1
cost scenario.

## Reading the three datasets

- **RSW Gun:** The proposed policy is designed to raise defect sensitivity
  when recent lots indicate risk, while retaining a higher-quality balance than
  a permanently aggressive threshold.
- **WM811K:** The policy illustrates the precision–recall trade-off most
  clearly. Lowering the decision threshold improves defect capture, but the
  resulting increase in false positives must be interpreted alongside F2 and
  G-mean rather than Precision alone.
- **CNC Milling Tool Life:** The sequential lot policy concentrates tightened
  inspection around periods with more rejected lots. This dataset is useful for
  reading the state-transition logic in Figure 1 as an operational inspection
  sequence.

## Findings stated in the thesis

The thesis reports the following patterns in its dataset-specific comparison
tables:

- For **RSW Gun**, the proposed policy improves Precision, Recall, F2, and
  G-mean over the fixed 0.5 threshold for all three base models. Against OMMA,
  the CatBoost version is highlighted as a balanced result with Recall 0.959,
  F2 0.733, and G-mean 0.954.
- For **WM811K_01**, the proposed policy lowers the threshold relative to 0.5.
  For XGBoost, the thesis reports Recall increasing from 0.590 to 0.617,
  F2 from 0.494 to 0.496, and G-mean from 0.728 to 0.738, while Precision
  decreases. Against OMMA, OMMA has higher Recall but much lower Precision
  (0.110, 0.110, and 0.122 for XGBoost, LightGBM, and CatBoost), so the
  proposed policy gives the better overall Precision–Recall balance.
- For **CNC_Milling_Tool_Life_03**, the proposed policy improves Recall, F2,
  and G-mean over the fixed 0.5 threshold for all three models. The thesis
  also reports that the proposed method outperforms OMMA on all reported
  metrics for XGBoost, LightGBM, and CatBoost.

These are interpretations of the thesis experiments, not claims of a causal
plant intervention. The observed effect depends on the dataset order, split
assumptions, model scores, and the selected cost scenario.

## Source tables

- [`journal_table_1_dataset_characteristics.csv`](../results/tables/journal_table_1_dataset_characteristics.csv)
- [`journal_table_2_main_performance_average.csv`](../results/tables/journal_table_2_main_performance_average.csv)
- [`journal_table_3_risk_concentration.csv`](../results/tables/journal_table_3_risk_concentration.csv)
- [`journal_table_6_bootstrap_ci.csv`](../results/tables/journal_table_6_bootstrap_ci.csv)
- [`journal_table_7_cost_sensitivity_fn10.csv`](../results/tables/journal_table_7_cost_sensitivity_fn10.csv)

## Boundaries

The results support a post-score inspection-policy interpretation. They do not
establish a causal plant intervention effect, and they should not be read as a
claim that tightened inspection removes all false positives or false negatives.
