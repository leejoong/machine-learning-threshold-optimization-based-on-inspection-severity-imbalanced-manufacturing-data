from __future__ import annotations

import pickle
import time
import gc

import numpy as np
import pandas as pd

from experiments.srs_ta_common import (
    CACHE_ROOT,
    DATA_ROOT,
    METRICS,
    MODEL_ROOT,
    RESULT_ROOT,
    THRESHOLD_GRID,
    classification_metrics,
    make_models,
    model_scores,
    write_markdown_summary,
)


DATASET = "RSW_Gun_532s_RowLevel"
WINDOW_SECONDS = 532
OFTA_WINDOW = 100

RAW_FEATURES = [f"c{i}" for i in range(1, 10)] + [f"c{i}" for i in range(11, 20)] + ["c10_on"]
RAW_COLUMNS = ["time", *[f"c{i}" for i in range(1, 20)], "error"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_start_time(path: Path) -> pd.Timestamp:
    sample = pd.read_csv(path, usecols=["time"], nrows=1)
    return pd.to_datetime(sample["time"].iloc[0], utc=True, errors="coerce")


def build_dataset() -> pd.DataFrame:
    cache_path = CACHE_ROOT / f"{DATASET}.pkl"
    if cache_path.exists():
        log(f"loading cache: {cache_path}")
        return pd.read_pickle(cache_path)

    root = DATA_ROOT / "rsw_gun"
    files = sorted(p for p in root.glob("*.csv") if not p.name.endswith("_demo.csv"))
    file_order = pd.DataFrame(
        [{"source_file": p.name, "path": p, "file_start": file_start_time(p)} for p in files]
    ).sort_values("file_start").reset_index(drop=True)
    file_order["file_seq"] = np.arange(len(file_order), dtype=np.int32)

    n_files = len(file_order)
    train_cut = int(n_files * 0.6)
    val_cut = int(n_files * 0.8)

    frames = []
    for row in file_order.itertuples(index=False):
        seq = int(row.file_seq)
        split = "train" if seq < train_cut else "val" if seq < val_cut else "test"
        log(f"reading {seq + 1}/{n_files} {row.source_file} -> {split}")

        df = pd.read_csv(row.path, usecols=RAW_COLUMNS, on_bad_lines="skip", low_memory=False)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        elapsed = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
        window_no = np.floor(elapsed / WINDOW_SECONDS).astype(np.int32)

        out = pd.DataFrame(index=df.index)
        for col in [f"c{i}" for i in range(1, 10)] + [f"c{i}" for i in range(11, 20)]:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        out["c10_on"] = df["c10"].astype(str).str.lower().eq("on").astype("float32")
        out["target"] = df["error"].astype(str).str.strip().ne("0").astype("int8")
        out["split"] = split
        out["source_file"] = row.source_file
        out["file_seq"] = np.int32(seq)
        out["window_no"] = window_no
        out["lot_id"] = (np.int64(seq) * 1_000_000 + window_no.astype(np.int64)).astype(np.int64)
        out["order"] = (np.int64(seq) * 1_000_000_000_000 + np.arange(len(out), dtype=np.int64)).astype(np.int64)
        frames.append(out)

    df_all = pd.concat(frames, ignore_index=True)
    log(f"saving cache: rows={len(df_all):,}, features={len(RAW_FEATURES)}")
    df_all.to_pickle(cache_path)
    return df_all


def split_cache_paths() -> dict[str, Path]:
    return {split: CACHE_ROOT / f"{DATASET}_{split}.pkl" for split in ["train", "val", "test"]}


def build_split_caches() -> dict[str, Path]:
    paths = split_cache_paths()
    if all(path.exists() for path in paths.values()):
        for split, path in paths.items():
            log(f"split cache exists: {split} -> {path}")
        return paths

    root = DATA_ROOT / "rsw_gun"
    files = sorted(p for p in root.glob("*.csv") if not p.name.endswith("_demo.csv"))
    file_order = pd.DataFrame(
        [{"source_file": p.name, "path": p, "file_start": file_start_time(p)} for p in files]
    ).sort_values("file_start").reset_index(drop=True)
    file_order["file_seq"] = np.arange(len(file_order), dtype=np.int32)

    n_files = len(file_order)
    train_cut = int(n_files * 0.6)
    val_cut = int(n_files * 0.8)
    frames = {"train": [], "val": [], "test": []}

    for row in file_order.itertuples(index=False):
        seq = int(row.file_seq)
        split = "train" if seq < train_cut else "val" if seq < val_cut else "test"
        log(f"split-cache reading {seq + 1}/{n_files} {row.source_file} -> {split}")
        df = pd.read_csv(row.path, usecols=RAW_COLUMNS, on_bad_lines="skip", low_memory=False)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        elapsed = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
        window_no = np.floor(elapsed / WINDOW_SECONDS).astype(np.int32)

        out = pd.DataFrame(index=df.index)
        for col in [f"c{i}" for i in range(1, 10)] + [f"c{i}" for i in range(11, 20)]:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        out["c10_on"] = df["c10"].astype(str).str.lower().eq("on").astype("float32")
        out["target"] = df["error"].astype(str).str.strip().ne("0").astype("int8")
        out["source_file"] = row.source_file
        out["file_seq"] = np.int32(seq)
        out["window_no"] = window_no
        out["lot_id"] = (np.int64(seq) * 1_000_000 + window_no.astype(np.int64)).astype(np.int64)
        out["order"] = (np.int64(seq) * 1_000_000_000_000 + np.arange(len(out), dtype=np.int64)).astype(np.int64)
        frames[split].append(out)

    for split, split_frames in frames.items():
        part = pd.concat(split_frames, ignore_index=True).sort_values("order").reset_index(drop=True)
        log(f"saving split cache: {split}, rows={len(part):,}, lots={part['lot_id'].nunique():,}")
        part.to_pickle(paths[split])
        del part
        gc.collect()
    return paths


def load_part(paths: dict[str, Path], split: str) -> pd.DataFrame:
    log(f"loading split: {split}")
    return pd.read_pickle(paths[split])


def best_threshold_sorted(y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y, dtype=np.int64)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    n = len(scores)

    best_theta, best_f2 = 0.5, -1.0
    for theta in grid:
        idx = int(np.searchsorted(sorted_scores, theta, side="left"))
        pred_pos = n - idx
        tp = total_pos - (int(cum_pos[idx - 1]) if idx > 0 else 0)
        fp = pred_pos - tp
        fn = total_pos - tp
        denom = 5 * tp + 4 * fn + fp
        f2 = (5 * tp / denom) if denom else 0.0
        if f2 > best_f2:
            best_theta, best_f2 = float(theta), float(f2)
    return best_theta, best_f2


def lot_reject_table(part: pd.DataFrame) -> pd.DataFrame:
    return (
        part.groupby("lot_id", sort=False)
        .agg(lot_order=("order", "min"), reject=("target", "max"), row_count=("target", "size"))
        .sort_values("lot_order")
    )


def states_by_lot(part: pd.DataFrame) -> np.ndarray:
    lots = lot_reject_table(part)
    state = "Normal"
    accept_count_tight = 0
    reject_history = []
    state_by_lot = {}
    for lot_id, row in lots.iterrows():
        state_by_lot[lot_id] = state
        reject = int(row["reject"])
        reject_history.append(reject)

        if state == "Tightened" and reject == 0:
            accept_count_tight += 1
        elif reject == 1 or state == "Normal":
            accept_count_tight = 0

        if len(reject_history) >= 5:
            recent = sum(reject_history[-5:])
            if state == "Normal" and recent >= 2:
                state = "Tightened"
                accept_count_tight = 0
            elif state == "Tightened" and accept_count_tight >= 5:
                state = "Normal"
                accept_count_tight = 0

    return part["lot_id"].map(state_by_lot).to_numpy()


def srs_predict(scores: np.ndarray, states: np.ndarray, theta_tight: float) -> np.ndarray:
    thresholds = np.where(np.asarray(states) == "Tightened", theta_tight, 0.5)
    return (np.asarray(scores) >= thresholds).astype(np.int8)


def best_srs_threshold_sorted(y_true: np.ndarray, scores: np.ndarray, states: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    states = np.asarray(states)

    normal_mask = states == "Normal"
    tight_mask = ~normal_mask
    normal_pred = scores[normal_mask] >= 0.5
    normal_y = y_true[normal_mask]
    base_tp = int(np.sum(normal_pred & (normal_y == 1)))
    base_fp = int(np.sum(normal_pred & (normal_y == 0)))
    base_fn = int(np.sum((~normal_pred) & (normal_y == 1)))

    tight_y = y_true[tight_mask]
    tight_scores = scores[tight_mask]
    if len(tight_y) == 0:
        f2 = classification_metrics(y_true, scores >= 0.5)["f2"]
        return 0.5, f2

    order = np.argsort(tight_scores)
    sorted_scores = tight_scores[order]
    sorted_y = tight_y[order]
    cum_pos = np.cumsum(sorted_y, dtype=np.int64)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    n = len(tight_scores)

    best_theta, best_f2 = 0.5, -1.0
    for theta in np.arange(0.001, 0.501, 0.001):
        idx = int(np.searchsorted(sorted_scores, theta, side="left"))
        pred_pos = n - idx
        tight_tp = total_pos - (int(cum_pos[idx - 1]) if idx > 0 else 0)
        tight_fp = pred_pos - tight_tp
        tight_fn = total_pos - tight_tp
        tp = base_tp + tight_tp
        fp = base_fp + tight_fp
        fn = base_fn + tight_fn
        denom = 5 * tp + 4 * fn + fp
        f2 = (5 * tp / denom) if denom else 0.0
        if f2 > best_f2:
            best_theta, best_f2 = float(theta), float(f2)
    return best_theta, best_f2


def threshold_or_default(y_true: np.ndarray, scores: np.ndarray, default_theta: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float(default_theta)
    theta, _ = best_threshold_sorted(y_true, scores)
    return theta


def ofta_predict(part: pd.DataFrame, scores: np.ndarray, window: int = OFTA_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    y_true = part["target"].to_numpy(dtype=np.int8)
    pred = np.zeros(len(part), dtype=np.int8)
    thresholds = np.full(len(part), 0.5, dtype=np.float32)

    starts = part.groupby("lot_id", sort=False).indices
    lots = lot_reject_table(part)
    history_scores: list[float] = []
    history_rejects: list[int] = []
    for lot_id in lots.index:
        idx = np.asarray(starts[lot_id], dtype=np.int64)
        theta = 0.5
        if len(history_rejects) >= window:
            theta = threshold_or_default(
                np.asarray(history_rejects[-window:], dtype=np.int8),
                np.asarray(history_scores[-window:], dtype=np.float32),
                0.5,
            )
            thresholds[idx] = theta
        pred[idx] = (scores[idx] >= theta).astype(np.int8)
        history_scores.append(float(np.max(scores[idx])))
        history_rejects.append(int(np.max(y_true[idx])))
    return pred, thresholds


def x_y(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = part[RAW_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = part["target"].to_numpy(dtype=np.int8)
    return x, y


def run() -> None:
    out_dir = RESULT_ROOT / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = build_split_caches()
    audit_rows = []
    for split in ["train", "val", "test"]:
        part = load_part(paths, split)
        audit_rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "positive": int(part["target"].sum()),
                "positive_rate": float(part["target"].mean()),
                "lots": int(part["lot_id"].nunique()),
            }
        )
        del part
        gc.collect()
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")
    log("\\n" + audit.to_string(index=False))

    train_audit = audit[audit["split"].eq("train")].iloc[0]
    train_pos = int(train_audit["positive"])
    train_neg = int(train_audit["rows"]) - train_pos
    y_train_for_weight = np.concatenate(
        [np.zeros(train_neg, dtype=np.int8), np.ones(train_pos, dtype=np.int8)]
    )

    model_dir = MODEL_ROOT / DATASET
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_rows = []

    for model_name, model in make_models(y_train_for_weight).items():
        model_path = model_dir / f"{model_name}.pkl"
        if model_path.exists():
            log(f"loaded model: {model_name}")
            with model_path.open("rb") as f:
                model = pickle.load(f)
        else:
            train_part = load_part(paths, "train")
            x_train, y_train = x_y(train_part)
            log(f"training model: {model_name}")
            model.fit(x_train, y_train)
            with model_path.open("wb") as f:
                pickle.dump(model, f)
            del train_part, x_train, y_train
            gc.collect()

        val_part = load_part(paths, "val")
        x_val, y_val = x_y(val_part)
        val_states = states_by_lot(val_part)
        log(f"scoring validation: {model_name}")
        val_scores = model_scores(model, x_val).astype(np.float32)
        theta_f2, val_f2 = best_threshold_sorted(y_val, val_scores)
        theta_srs, val_srs_f2 = best_srs_threshold_sorted(y_val, val_scores, val_states)
        threshold_rows.append(
            {
                "model": model_name,
                "theta_f2": theta_f2,
                "theta_tight": theta_srs,
                "val_f2_static": val_f2,
                "val_f2_srs": val_srs_f2,
            }
        )
        del val_part, x_val, y_val, val_scores, val_states
        gc.collect()

        test_part = load_part(paths, "test")
        x_test, y_test = x_y(test_part)
        test_states = states_by_lot(test_part)
        log(f"evaluating {model_name} | test")
        scores = model_scores(model, x_test).astype(np.float32)
        ofta_pred, ofta_thresholds = ofta_predict(test_part, scores, window=OFTA_WINDOW)
        methods = {
            "0.5": (0.5, (scores >= 0.5).astype(np.int8)),
            "F2최적화": (theta_f2, (scores >= theta_f2).astype(np.int8)),
            "OFTA": (float(np.mean(ofta_thresholds)), ofta_pred),
            "제안": (theta_srs, srs_predict(scores, test_states, theta_srs)),
        }
        for method, (theta, pred) in methods.items():
            row = {"dataset": DATASET, "model": model_name, "split": "test", "method": method, "threshold": theta}
            row.update(classification_metrics(y_test, pred))
            rows.append(row)
        del test_part, x_test, y_test, test_states, scores, ofta_pred, ofta_thresholds
        gc.collect()

        pd.DataFrame(rows).to_csv(out_dir / "all_results_partial.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(threshold_rows).to_csv(out_dir / "thresholds_partial.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(rows)
    thresholds = pd.DataFrame(threshold_rows)
    results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    test = results[results["split"].eq("test")].copy()
    test.to_csv(out_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(test, out_dir / "test_results_bold.md")
    log("\\n" + test[["model", "method", "threshold", *METRICS]].to_string(index=False))
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    run()
