# Results interpretation

The tracked tables summarize the final three-dataset experiment. The figures
in [`assets/figures`](../assets/figures) are generated from the tracked tables
by `scripts/build_figures.py`.

## Risk concentration

The proposed policy captures most positive rows inside Tightened inspection for
CNC and RSW while keeping the tightened workload relatively small. WM811K is a
different regime: nearly the whole process enters Tightened inspection, which
should be interpreted as broad process instability rather than localized risk.

See [`journal_table_3_risk_concentration.csv`](../results/tables/journal_table_3_risk_concentration.csv)
and [`public_risk_concentration.png`](../assets/figures/public_risk_concentration.png).

## Cost sensitivity

The cost table uses FN:FP = 10:1 to represent a setting where missed defects
are materially more expensive than additional inspection alarms. This is a
scenario analysis, not a universal economic claim; users should substitute
their own costs.

See [`journal_table_7_cost_sensitivity_fn10.csv`](../results/tables/journal_table_7_cost_sensitivity_fn10.csv)
and [`public_cost_sensitivity.png`](../assets/figures/public_cost_sensitivity.png).

## Boundaries

The results support a post-score inspection-policy interpretation. They do not
establish a causal plant intervention effect, and they should not be read as a
claim that tightened inspection removes all false positives or false negatives.
