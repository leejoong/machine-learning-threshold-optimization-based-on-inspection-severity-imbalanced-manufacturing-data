"""Build WM811K lot features and run the three-model policy experiment."""

from __future__ import annotations

import pickle
import re
import sys
import types

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.srs_ta_common import CACHE_ROOT, DATA_ROOT
from experiments.srs_ta_group_common import run_group_srs_experiment


SEED = 42


def lot_number(value):
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def split_groups(df, group_col, target_col="target"):
    group_target = (
        df.groupby(group_col, as_index=False)
        .agg(target=(target_col, "max"), order=("order", "min"))
        .sort_values("order")
        .reset_index(drop=True)
    )
    n = len(group_target)
    chronological = group_target.copy()
    chronological["split"] = "test"
    chronological.loc[: int(n * 0.6) - 1, "split"] = "train"
    chronological.loc[int(n * 0.6) : int(n * 0.8) - 1, "split"] = "val"
    if chronological.groupby("split")["target"].nunique().min() >= 2:
        split_map = dict(zip(chronological[group_col], chronological["split"]))
        out = df.copy()
        out["split"] = out[group_col].map(split_map)
        return out, "chronological_group_60_20_20"

    trainval, test = train_test_split(
        group_target,
        test_size=0.2,
        random_state=SEED,
        stratify=group_target["target"],
    )
    train, val = train_test_split(
        trainval,
        test_size=0.25,
        random_state=SEED,
        stratify=trainval["target"],
    )
    split_map = {group: "train" for group in train[group_col]}
    split_map.update({group: "val" for group in val[group_col]})
    split_map.update({group: "test" for group in test[group_col]})
    out = df.copy()
    out["split"] = out[group_col].map(split_map)
    return out, "stratified_group_60_20_20"


def install_old_pandas_pickle_shims():
    """Allow the public loader to read older WM811K pandas pickles."""

    from pandas.core.indexes.base import _new_Index

    indexes_mod = types.ModuleType("pandas.indexes")
    base_mod = types.ModuleType("pandas.indexes.base")
    base_mod.Index = pd.Index
    base_mod._new_Index = _new_Index

    numeric_mod = types.ModuleType("pandas.indexes.numeric")
    numeric_mod.Index = pd.Index
    numeric_mod.Int64Index = pd.Index
    numeric_mod.UInt64Index = pd.Index
    numeric_mod.Float64Index = pd.Index

    range_mod = types.ModuleType("pandas.indexes.range")
    range_mod.RangeIndex = pd.RangeIndex

    multi_mod = types.ModuleType("pandas.indexes.multi")
    multi_mod.MultiIndex = pd.MultiIndex

    category_mod = types.ModuleType("pandas.indexes.category")
    category_mod.CategoricalIndex = pd.CategoricalIndex

    sys.modules["pandas.indexes"] = indexes_mod
    sys.modules["pandas.indexes.base"] = base_mod
    sys.modules["pandas.indexes.numeric"] = numeric_mod
    sys.modules["pandas.indexes.range"] = range_mod
    sys.modules["pandas.indexes.multi"] = multi_mod
    sys.modules["pandas.indexes.category"] = category_mod


def failure_label(value):
    arr = np.asarray(value)
    return "" if arr.size == 0 else str(arr.ravel()[0])


def wafer_feature_row(wafer_map):
    arr = np.asarray(wafer_map)
    if arr.ndim != 2:
        return {name: np.nan for name in (
            "map_rows", "map_cols", "die_count", "good_count", "defect_count",
            "defect_ratio", "edge_defect_ratio", "center_defect_ratio",
            "defect_row_mean", "defect_col_mean", "defect_row_std", "defect_col_std",
        )}

    active = arr > 0
    good = arr == 1
    defect = arr == 2
    rows, cols = arr.shape
    die_count = int(active.sum())
    defect_count = int(defect.sum())

    edge_width = max(1, int(round(min(rows, cols) * 0.15)))
    edge = np.zeros_like(active, dtype=bool)
    edge[:edge_width, :] = True
    edge[-edge_width:, :] = True
    edge[:, :edge_width] = True
    edge[:, -edge_width:] = True

    rr, cc = np.indices(arr.shape)
    center_r = (rows - 1) / 2
    center_c = (cols - 1) / 2
    radius = np.sqrt(((rr - center_r) / max(rows, 1)) ** 2 + ((cc - center_c) / max(cols, 1)) ** 2)
    center = radius <= np.nanpercentile(radius[active], 35) if die_count else np.zeros_like(active, dtype=bool)

    defect_pos = np.argwhere(defect)
    if len(defect_pos):
        r_mean = float(defect_pos[:, 0].mean() / max(rows - 1, 1))
        c_mean = float(defect_pos[:, 1].mean() / max(cols - 1, 1))
        r_std = float(defect_pos[:, 0].std() / max(rows - 1, 1))
        c_std = float(defect_pos[:, 1].std() / max(cols - 1, 1))
    else:
        r_mean = c_mean = r_std = c_std = 0.0

    return {
        "map_rows": rows,
        "map_cols": cols,
        "die_count": die_count,
        "good_count": int(good.sum()),
        "defect_count": defect_count,
        "defect_ratio": defect_count / die_count if die_count else 0.0,
        "edge_defect_ratio": int((defect & edge).sum()) / defect_count if defect_count else 0.0,
        "center_defect_ratio": int((defect & center).sum()) / defect_count if defect_count else 0.0,
        "defect_row_mean": r_mean,
        "defect_col_mean": c_mean,
        "defect_row_std": r_std,
        "defect_col_std": c_std,
    }


def build_wm811k():
    dataset = "WM811K_lot"
    cache_path = CACHE_ROOT / f"{dataset}.pkl"
    if cache_path.exists():
        print(f"loading cache: {cache_path}")
        return dataset, pd.read_pickle(cache_path)

    install_old_pandas_pickle_shims()
    path = DATA_ROOT / "wm811k" / "LSWMD.pkl"
    print(f"loading WM811K pickle: {path}", flush=True)
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")

    raw = raw.copy()
    raw["label_text"] = raw["failureType"].map(failure_label)
    raw = raw[raw["label_text"].ne("")].reset_index(drop=True)
    raw["target"] = raw["label_text"].ne("none").astype(int)
    raw["lot_order"] = raw["lotName"].map(lot_number)
    print(f"extracting wafer features: {len(raw)} labeled wafers", flush=True)

    features = pd.DataFrame([wafer_feature_row(value) for value in raw["waferMap"]])
    df = pd.concat(
        [features, raw[["dieSize", "waferIndex", "lotName", "lot_order", "target", "label_text"]].reset_index(drop=True)],
        axis=1,
    )
    df["order"] = df["lot_order"].astype(float) * 100 + df["waferIndex"].astype(float)
    df, strategy = split_groups(df, "lotName")
    df["split_strategy"] = strategy
    df.to_pickle(cache_path)
    return dataset, df


def main():
    dataset, df = build_wm811k()
    exclude_cols = {"target", "split", "order", "lotName", "lot_order", "label_text", "split_strategy"}
    feature_cols = [column for column in df.columns if column not in exclude_cols and pd.api.types.is_numeric_dtype(df[column])]
    run_group_srs_experiment(dataset, df, feature_cols, subgroup_col="lotName")


if __name__ == "__main__":
    main()
