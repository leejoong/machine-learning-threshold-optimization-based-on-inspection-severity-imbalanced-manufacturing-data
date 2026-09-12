import math
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from thesis_policy.config import (
    CACHE_ROOT,
    DATA_ROOT,
    MODEL_ROOT,
    RESULT_ROOT,
    ensure_output_dirs,
)

SEED = 42
ensure_output_dirs()

METRICS = ["precision", "recall", "f1", "f2", "gmean", "fpr"]
THRESHOLD_GRID = np.arange(0.001, 1.0, 0.001)


def classification_metrics(y_true, pred):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "gmean": math.sqrt(recall * specificity),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def state_sequence(y_true):
    states = []
    history = []
    state = "Normal"
    for y in np.asarray(y_true).astype(int):
        states.append(state)
        history.append(int(y))
        if len(history) >= 5:
            recent = sum(history[-5:])
            if state == "Normal" and recent >= 2:
                state = "Tightened"
            elif state == "Tightened" and recent == 0:
                state = "Normal"
    return np.asarray(states)


def srs_predict(scores, states, theta_tight):
    thresholds = np.where(states == "Tightened", theta_tight, 0.5)
    return (np.asarray(scores) >= thresholds).astype(int)


def best_static_threshold(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    pred = scores[None, :] >= THRESHOLD_GRID[:, None]
    tp = pred @ y_true
    fp = pred.sum(axis=1) - tp
    fn = y_true.sum() - tp
    denom = 5 * tp + 4 * fn + fp
    f2 = np.divide(5 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    best_idx = int(np.argmax(f2))
    return float(THRESHOLD_GRID[best_idx]), float(f2[best_idx])


def best_static_threshold_or_default(y_true, scores, default_theta=0.5):
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float(default_theta)
    theta, _ = best_static_threshold(y_true, scores)
    return theta


def online_sliding_f2_predict(scores, y_true, window=5, default_theta=0.5):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)
    pred = np.zeros(len(scores), dtype=int)
    thresholds = np.full(len(scores), float(default_theta), dtype=float)
    for i in range(len(scores)):
        if i >= window:
            theta = best_static_threshold_or_default(y_true[i - window : i], scores[i - window : i], default_theta)
            thresholds[i] = theta
        pred[i] = int(scores[i] >= thresholds[i])
    return pred, thresholds


def best_srs_threshold(y_true, scores, states):
    best_theta, best_f2 = 0.5, -1.0
    for theta in np.arange(0.001, 0.501, 0.001):
        f2 = classification_metrics(y_true, srs_predict(scores, states, theta))["f2"]
        if f2 > best_f2:
            best_theta, best_f2 = float(theta), float(f2)
    return best_theta, best_f2


def make_models(y_train):
    y_train = np.asarray(y_train).astype(int)
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos
    return {
        "XGBoost": XGBClassifier(
            random_state=SEED,
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=SEED,
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            random_seed=SEED,
            iterations=180,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
        ),
    }


def model_scores(model, x):
    return model.predict_proba(x)[:, 1]


def prepare_xy(df, feature_cols):
    data = {}
    parts = {
        split: df[df["split"].eq(split)].sort_values("order").reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    x_train_raw = parts["train"][feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(x_train_raw)
    data["X_train"] = scaler.fit_transform(x_train)
    data["y_train"] = parts["train"]["target"].to_numpy(dtype=int)
    for split in ["val", "test"]:
        x_raw = parts[split][feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        data[f"X_{split}"] = scaler.transform(imputer.transform(x_raw))
        data[f"y_{split}"] = parts[split]["target"].to_numpy(dtype=int)

    audit = []
    for split, part in parts.items():
        y = part["target"].to_numpy(dtype=int)
        audit.append(
            {
                "split": split,
                "rows": int(len(part)),
                "positive": int(y.sum()),
                "positive_rate": float(y.mean()) if len(y) else 0.0,
            }
        )
    return data, pd.DataFrame(audit)


def run_srs_experiment(dataset_name, df, feature_cols):
    out_dir = RESULT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data, audit = prepare_xy(df, feature_cols)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")

    for split in ["train", "val", "test"]:
        if len(np.unique(data[f"y_{split}"])) < 2:
            raise ValueError(f"{dataset_name} {split} split has only one class. Adjust split rule.")

    states = {split: state_sequence(data[f"y_{split}"]) for split in ["train", "val", "test"]}
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

    model_dir = MODEL_ROOT / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_rows = []
    for model_name, model in make_models(data["y_train"]).items():
        model_path = model_dir / f"{model_name}.pkl"
        if model_path.exists():
            with model_path.open("rb") as f:
                model = pickle.load(f)
            print(f"loaded model: {dataset_name} | {model_name}", flush=True)
        else:
            print(f"training: {dataset_name} | {model_name}", flush=True)
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
            online_pred, online_thresholds = online_sliding_f2_predict(scores, y)
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

        pd.DataFrame(rows).to_csv(out_dir / "all_results_partial.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(threshold_rows).to_csv(out_dir / "thresholds_partial.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(rows)
    thresholds = pd.DataFrame(threshold_rows)
    results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    test = results[results["split"].eq("test")].copy()
    test.to_csv(out_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(test, out_dir / "test_results_bold.md")
    print(test[["model", "method", "threshold", *METRICS]].to_string(index=False), flush=True)
    print(f"saved: {out_dir}", flush=True)


def write_markdown_summary(test, path):
    cols = ["model", "method", "threshold", *METRICS]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in test.iterrows():
        group = test[test["model"].eq(row["model"])]
        vals = [row["model"], row["method"], f"{row['threshold']:.3f}"]
        for metric in METRICS:
            best = group[metric].min() if metric == "fpr" else group[metric].max()
            text = f"{row[metric]:.4f}"
            if np.isclose(row[metric], best):
                text = f"**{text}**"
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")
