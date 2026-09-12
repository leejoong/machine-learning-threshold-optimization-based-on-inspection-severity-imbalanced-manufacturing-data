# Data sources and scope

The public repository contains code and small summary tables, not raw data.
Raw datasets should be obtained from their original public sources and stored
outside this repository. The local code expects the following directory names:

| Paper dataset | Expected local path | Loader |
|---|---|---|
| CNC Milling Tool Life | `cnc_milling_tool_life_2025/FeatureAndMetadata_Milling.csv` | `experiments/run_cnc_milling_tool_life.py` |
| Resistance Spot Welding | `rsw_gun/*.csv` | `experiments/run_rsw_gun_532s_rowlevel.py` |
| WM811K Wafer Map | `wm811k/LSWMD.pkl` | `experiments/run_wm811k.py` |

Processing follows the paper's split logic: chronological/group-aware
train-validation-test partitions, with validation used for threshold
selection and test data held out for final evaluation.

Before publication, the final manuscript should cite the original article and
official distribution page for each dataset and record the access date where
the source requires it.
