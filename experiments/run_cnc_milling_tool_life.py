import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.srs_ta_common import (
    DATA_ROOT,
    METRICS,
    MODEL_ROOT,
    RESULT_ROOT,
    classification_metrics,
    make_models,
    model_scores,
    online_sliding_f2_predict,
    prepare_xy,
    srs_predict,
    state_sequence,
    best_static_threshold,
    best_srs_threshold,
    write_markdown_summary,
)

DATASET = "CNC_Milling_Tool_Life_2025"
DATA_PATH = DATA_ROOT / "cnc_milling_tool_life_2025" / "FeatureAndMetadata_Milling.csv"
TARGET_THRESHOLD = 0.2
MIN_RUN_ROWS = 10


def build_dataset():
    df = pd.read_csv(DATA_PATH, sep=";", header=1, decimal=",")
    df = df.rename(columns={"TollIndex": "ToolIndex"})
    df["target"] = (df["CycleToFailureNormalized"] <= TARGET_THRESHOLD).astype(int)
    df["run_id"] = df["ToolIndex"].astype(int)

    run_sizes = df.groupby("run_id").size()
    keep_runs = set(run_sizes[run_sizes >= MIN_RUN_ROWS].index)
    df = df[df["run_id"].isin(keep_runs)].copy()

    tool_order = sorted(df["run_id"].unique())
    n_tools = len(tool_order)
    train_tools = set(tool_order[: int(n_tools * 0.6)])
    val_tools = set(tool_order[int(n_tools * 0.6) : int(n_tools * 0.8)])

    df["split"] = "test"
    df.loc[df["run_id"].isin(train_tools), "split"] = "train"
    df.loc[df["run_id"].isin(val_tools), "split"] = "val"
    df["order"] = df["run_id"].astype(float) * 1_000_000 + df["NumberOfCycle"].astype(float)

    non_features = {
        "FileName",
        "SampleIndex",
        "ToolIndex",
        "run_id",
        "target",
        "split",
        "order",
        "CycleToFailure",
        "CycleToFailureNormalized",
    }
    feature_cols = [c for c in df.columns if c not in non_features]
    return df, feature_cols


def grouped_states(part):
    states = np.empty(len(part), dtype=object)
    for _, idx in part.groupby("run_id", sort=False).groups.items():
        y = part.loc[idx, "target"].to_numpy(dtype=int)
        states[np.asarray(idx)] = state_sequence(y)
    return states


def grouped_online_sliding_f2(part, scores):
    pred = np.zeros(len(part), dtype=int)
    thresholds = np.full(len(part), 0.5, dtype=float)
    for _, idx in part.groupby("run_id", sort=False).groups.items():
        group_idx = np.asarray(idx)
        group_pred, group_thresholds = online_sliding_f2_predict(
            np.asarray(scores)[group_idx],
            part.loc[group_idx, "target"].to_numpy(dtype=int),
        )
        pred[group_idx] = group_pred
        thresholds[group_idx] = group_thresholds
    return pred, thresholds


def run_experiment():
    df, feature_cols = build_dataset()
    out_dir = RESULT_ROOT / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    run_audit = (
        df.groupby(["split", "run_id"], sort=True)
        .agg(rows=("target", "size"), positive=("target", "sum"), min_cycle=("NumberOfCycle", "min"), max_cycle=("NumberOfCycle", "max"))
        .reset_index()
    )
    run_audit.to_csv(out_dir / "run_audit.csv", index=False, encoding="utf-8-sig")

    data, audit = prepare_xy(df, feature_cols)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")

    parts = {
        split: df[df["split"].eq(split)].sort_values("order").reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    states = {split: grouped_states(parts[split]) for split in ["train", "val", "test"]}
    pd.DataFrame(
        [
            {
                "split": split,
                "normal": int((arr == "Normal").sum()),
                "tightened": int((arr == "Tightened").sum()),
            }
            for split, arr in states.items()
        ]
    ).to_csv(out_dir / "state_counts.csv", index=False, encoding="utf-8-sig")

    model_dir = MODEL_ROOT / DATASET
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_rows = []

    for model_name, model in make_models(data["y_train"]).items():
        model_path = model_dir / f"{model_name}.pkl"
        if model_path.exists():
            with model_path.open("rb") as f:
                model = pickle.load(f)
            print(f"loaded model: {DATASET} | {model_name}", flush=True)
        else:
            print(f"training: {DATASET} | {model_name}", flush=True)
            model.fit(data["X_train"], data["y_train"])
            with model_path.open("wb") as f:
                pickle.dump(model, f)

        val_scores = model_scores(model, data["X_val"])
        theta_f2, val_f2 = best_static_threshold(data["y_val"], val_scores)
        theta_srs, val_srs_f2 = best_srs_threshold(data["y_val"], val_scores, states["val"])
        threshold_rows.append(
            {
                "model": model_name,
                "theta_f2": theta_f2,
                "theta_srs_tight": theta_srs,
                "val_f2_static": val_f2,
                "val_f2_srs": val_srs_f2,
            }
        )

        for split in ["train", "val", "test"]:
            y = data[f"y_{split}"]
            scores = model_scores(model, data[f"X_{split}"])
            online_pred, online_thresholds = grouped_online_sliding_f2(parts[split], scores)
            methods = {
                "theta=0.5": (0.5, (scores >= 0.5).astype(int)),
                "F2-only": (theta_f2, (scores >= theta_f2).astype(int)),
                "UnitOnline-F2-TA": (float(np.mean(online_thresholds)), online_pred),
                "SRS-TA": (theta_srs, srs_predict(scores, states[split], theta_srs)),
            }
            for method, (theta, pred) in methods.items():
                row = {"dataset": DATASET, "model": model_name, "split": split, "method": method, "threshold": theta}
                row.update(classification_metrics(y, pred))
                rows.append(row)

    results = pd.DataFrame(rows)
    thresholds = pd.DataFrame(threshold_rows)
    results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    test = results[results["split"].eq("test")].copy()
    test.to_csv(out_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(test, out_dir / "test_results_bold.md")
    print(test[["model", "method", "threshold", *METRICS]].to_string(index=False), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    run_experiment()
