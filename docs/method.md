# Method and reproducibility notes

The public code follows the submitted thesis. The repository contains the
paper's two main figures in [`assets/figures`](../assets/figures): Figure 1
shows the train/validation/test procedure and Figure 2 defines the confusion
matrix used by the reported metrics.

The proposed method is a decision-policy layer placed after a probabilistic
classifier. The classifier is trained once; the policy changes the decision
threshold according to the recent inspection history.

## Policy rule

- Normal state uses `theta_normal = 0.5`.
- Tightened state searches `theta_tight` over `0.001, 0.002, ..., 0.500`.
- The current lot is assigned its state before its outcome is observed.
- At least two rejected lots in the most recent five lots switch the next lot
  to Tightened.
- Five accepted lots while Tightened return the next lot to Normal.

This timing prevents the current outcome from determining the current lot's
threshold. The validation split selects `theta_tight`; the test split is used
only for final evaluation.

## How to read Figure 1

Figure 1 separates model learning from policy execution. The classifier first
produces a score for each row or lot. The policy then assigns the current lot a
Normal or Tightened state using only prior lot history. Normal uses a threshold
of 0.5; Tightened uses the validation-selected `theta_tight`. After the current
lot is classified, its reject result updates the state used by the next lot.
This sequence is important: the current lot's observed label cannot influence
its own threshold.

## How to read Figure 2

Figure 2 maps actual and predicted classes to TP, FP, FN, and TN. The thesis
uses these four counts to calculate Precision, Recall, F2, G-mean, and related
diagnostics. In an imbalanced inspection problem, Recall is especially
important because FN represents a missed defect. F2 therefore gives Recall
greater weight than Precision, while G-mean checks that positive detection does
not come at the cost of completely losing negative-class specificity.

## Datasets in this public package

Only the three datasets used in the paper are supported:

1. CNC Milling Tool Life
2. Resistance Spot Welding (RSW Gun)
3. WM811K Wafer Map

Raw data stays outside Git. Set `THESIS_DATA_ROOT` to the folder containing
these dataset directories. The default is the parent directory of this
repository on the author's local D drive.

## Models and metrics

The experiment runners compare XGBoost, LightGBM, and CatBoost using the same
trained classifier scores across decision policies. Reported metrics include
precision, recall, F1, F2, G-mean, FPR, confusion-matrix counts, and an
expected-cost scenario where false negatives are ten times false positives.

## Reproduction order

```powershell
python -m pytest
python scripts/verify_thesis_assets.py
python -m experiments.run_cnc_milling_tool_life
python -m experiments.run_rsw_gun_532s_rowlevel
python -m experiments.run_wm811k
```

The full runners can require substantial memory and time. They write caches,
models, and generated results to `outputs/`, which is intentionally ignored by
Git.
