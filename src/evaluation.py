"""Shared evaluation for every model (text, image, multimodal), so results are directly comparable.

Conventions
- Scores are probabilities or any "higher = more likely fake" number.
- Thresholds are chosen on validation and then applied unchanged to test.
- The official test set is ~67% fake. Real-world prevalence is far lower, so `prevalence_table` re-expresses
  results at realistic fake rates. Recall (TPR) and false-positive rate do not depend on prevalence, so
  precision and alert volumes at any prevalence follow from them exactly, without resampling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

DEFAULT_PREVALENCES = (0.001, 0.01, 0.05)


def equal_error_rate(y_true, score):
    """Error rate at the threshold where the false-positive rate equals the false-negative rate (lower is better)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def best_f1_threshold(y_true, score):
    """Threshold that maximises F1. Choose it on validation, never on test."""
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y_true, score >= t) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def binary_metrics(y_true, score, threshold=0.5):
    """Threshold-free metrics (AUC, PR-AUC, EER) plus thresholded ones (accuracy, precision, recall, F1)."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pred = score >= threshold
    return {
        "AUC": roc_auc_score(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
        "EER": equal_error_rate(y_true, score),
        "ACC": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def combine_fake_score(p_image, p_text):
    """P(pair is fake) from the two per-modality probabilities: fake if either modality is manipulated.
    Noisy-OR, i.e. assumes the two predictions are independent."""
    return 1 - (1 - np.asarray(p_image)) * (1 - np.asarray(p_text))


def evaluate(labels, p_image, p_text, thresholds=None):
    """Main results table: one row per target (is_fake, image_fake, text_fake).

    labels: DataFrame with is_fake, image_fake, text_fake columns, aligned with the predictions.
    thresholds: dict target -> threshold (from validation); defaults to 0.5 for each.
    """
    thresholds = thresholds or {}
    scores = {"is_fake": combine_fake_score(p_image, p_text), "image_fake": p_image, "text_fake": p_text}
    rows = {target: binary_metrics(labels[target], s, thresholds.get(target, 0.5)) for target, s in scores.items()}
    return pd.DataFrame(rows).T


def detection_rate_by_type(labels, score, threshold, group_col="fake_cls"):
    """Share of each group flagged as fake. For `orig` this is the false-positive rate, for every other
    group it is the recall on that manipulation type."""
    flagged = pd.Series(np.asarray(score) >= threshold, index=labels.index)
    out = flagged.groupby(labels[group_col]).agg(["mean", "size"])
    out.columns = ["flagged_rate", "n"]
    return out.sort_values("flagged_rate")


def prevalence_table(y_true, score, threshold, prevalences=DEFAULT_PREVALENCES, volume=100_000):
    """What the model would do on `volume` posts at realistic fake rates.

    Uses the recall and false-positive rate measured at `threshold`; these are the same at any prevalence.
    """
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(score) >= threshold
    tpr = pred[y_true].mean()
    fpr = pred[~y_true].mean()
    rows = []
    for p in prevalences:
        fakes = volume * p
        caught = tpr * fakes
        false_alarms = fpr * (volume - fakes)
        rows.append({
            "fake_rate": p,
            "fakes_per_100k": fakes,
            "alerts": caught + false_alarms,
            "fakes_caught": caught,
            "false_alarms": false_alarms,
            "precision": caught / (caught + false_alarms) if caught + false_alarms else np.nan,
            "recall": tpr,
        })
    return pd.DataFrame(rows).set_index("fake_rate")
