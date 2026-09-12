import pickle

import numpy as np
import pandas as pd

from experiments.srs_ta_common import (
    METRICS,
    MODEL_ROOT,
    RESULT_ROOT,
    best_srs_threshold,
    best_static_threshold,
    best_static_threshold_or_default,
    classification_metrics,
    make_models,
    model_scores,
    prepare_xy,
    srs_predict,
    write_markdown_summary,
)


def subgroup_states(part, subgroup_col):
    groups = (
        part.groupby(subgroup_col, sort=False)
        .agg(group_order=("order", "min"), reject=("target", "max"))
        .sort_values("group_order")
    )
    history = []
    state = "Normal"
    state_by_group = {}
    for group_id, row in groups.iterrows():
        state_by_group[group_id] = state
        history.append(int(row["reject"]))
        if len(history) >= 5:
            recent = sum(history[-5:])
            if state == "Normal" and recent >= 2:
                state = "Tightened"
            elif state == "Tightened" and recent == 0:
                state = "Normal"
    return part[subgroup_col].map(state_by_group).to_numpy()


def prepare_group_states(df, subgroup_col):
    states = {}
    counts = []
    for split in ["train", "val", "test"]:
        part = df[df["split"].eq(split)].sort_values("order").reset_index(drop=True)
        arr = subgroup_states(part, subgroup_col)
        states[split] = arr
        counts.append(
            {
                "split": split,
                "groups": int(part[subgroup_col].nunique()),
                "normal_rows": int((arr == "Normal").sum()),
                "tightened_rows": int((arr == "Tightened").sum()),
            }
        )
    return states, pd.DataFrame(counts)


def unit_online_group_f2_predict(part, scores, subgroup_col, window=5, default_theta=0.5):
    scores = np.asarray(scores)
    y_true = part["target"].to_numpy(dtype=int)
    pred = np.zeros(len(part), dtype=int)
    thresholds = np.full(len(part), float(default_theta), dtype=float)
    groups = (
        part.groupby(subgroup_col, sort=False)
        .agg(group_order=("order", "min"))
        .sort_values("group_order")
    )
    history_scores = []
    history_rejects = []
    for group_id in groups.index:
        idx = part.index[part[subgroup_col].eq(group_id)].to_numpy()
        if len(history_rejects) >= window:
            theta = best_static_threshold_or_default(
                np.asarray(history_rejects[-window:], dtype=int),
                np.asarray(history_scores[-window:], dtype=float),
                default_theta,
            )
            thresholds[idx] = theta
        pred[idx] = (scores[idx] >= thresholds[idx]).astype(int)
        history_scores.append(float(np.max(scores[idx])))
        history_rejects.append(int(np.max(y_true[idx])))
    return pred, thresholds


def run_group_srs_experiment(dataset_name, df, feature_cols, subgroup_col, model_cache_name=None):
    out_dir = RESULT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data, audit = prepare_xy(df, feature_cols)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")

    for split in ["train", "val", "test"]:
        if len(np.unique(data[f"y_{split}"])) < 2:
            raise ValueError(f"{dataset_name} {split} split has only one class. Adjust split rule.")

    parts = {
        split: df[df["split"].eq(split)].sort_values("order").reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    states, state_counts = prepare_group_states(df, subgroup_col)
    state_counts.to_csv(out_dir / "state_counts.csv", index=False, encoding="utf-8-sig")

    cache_name = model_cache_name or dataset_name
    model_dir = MODEL_ROOT / cache_name
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    threshold_rows = []
    for model_name, model in make_models(data["y_train"]).items():
        model_path = model_dir / f"{model_name}.pkl"
        if model_path.exists():
            with model_path.open("rb") as f:
                model = pickle.load(f)
            print(f"loaded model: {cache_name} | {model_name}", flush=True)
        else:
            print(f"training: {cache_name} | {model_name}", flush=True)
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
            online_pred, online_thresholds = unit_online_group_f2_predict(parts[split], scores, subgroup_col)
            methods = {
                "theta=0.5": (0.5, (scores >= 0.5).astype(int)),
                "F2-only": (theta_f2, (scores >= theta_f2).astype(int)),
                "UnitOnline-F2-TA": (float(np.mean(online_thresholds)), online_pred),
                "SRS-TA": (theta_srs, srs_predict(scores, states[split], theta_srs)),
            }
            for method, (theta, pred) in methods.items():
                row = {"dataset": dataset_name, "model": model_name, "split": split, "method": method, "threshold": theta}
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
