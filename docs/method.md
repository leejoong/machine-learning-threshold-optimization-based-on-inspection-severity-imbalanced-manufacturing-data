# Method and reproducibility notes

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
python scripts/build_figures.py
python -m experiments.run_cnc_milling_tool_life
python -m experiments.run_rsw_gun_532s_rowlevel
python -m experiments.run_wm811k
```

The full runners can require substantial memory and time. They write caches,
models, and generated results to `outputs/`, which is intentionally ignored by
Git.
