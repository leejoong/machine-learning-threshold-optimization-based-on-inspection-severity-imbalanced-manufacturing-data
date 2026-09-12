import numpy as np

from thesis_policy.omma_final_utils import (
    classification_metrics,
    omma_predict,
    omma_threshold,
)


def test_classification_metrics_from_balanced_counts():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f2"] == 0.5
    assert metrics["gmean"] == 0.5


def test_omma_threshold_is_bounded():
    threshold = omma_threshold("f2", tp=3, fp=2, fn=4, tn=20)
    assert 0.0 <= threshold <= 1.0


def test_omma_predict_is_single_pass_and_aligned():
    y_true = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.7, 0.9])
    predictions, thresholds = omma_predict(y_true, scores, objective="f2")
    assert predictions.shape == y_true.shape
    assert thresholds.shape == scores.shape
    assert set(predictions.tolist()) <= {0, 1}
    assert np.all((thresholds >= 0.0) & (thresholds <= 1.0))
