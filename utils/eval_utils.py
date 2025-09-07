from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        y_pred = predictions.argmax(axis=1)
    else:
        y_pred = (predictions.ravel() > 0).astype(int)

    y_true = labels.astype(int)
    assert y_true.shape[0] == y_pred.shape[0]

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    accuracy = float((y_true == y_pred).mean())

    recalls = []
    f1s = []
    eps = 1e-12
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        prec_c = tp / (tp + fp + eps)
        rec_c = tp / (tp + fn + eps)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
        recalls.append(rec_c)
        f1s.append(f1_c)

    recall_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
    }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _log_confidence_metrics(
    writer: SummaryWriter,
    tag_prefix: str,
    logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    global_step: Optional[int] = None,
) -> None:
    if logits.ndim == 1:
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.stack([1 - probs, probs], axis=1)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probs = np.stack([1 - probs, probs], axis=1)
    else:
        probs = _softmax(logits, axis=1)

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    y_true = labels.astype(int)

    overall_acc = float((preds == y_true).mean())
    mean_conf = float(confidences.mean())
    writer.add_scalar(f"{tag_prefix}/confidence/overall_accuracy", overall_acc, global_step or 0)
    writer.add_scalar(f"{tag_prefix}/confidence/mean_confidence", mean_conf, global_step or 0)
    writer.add_histogram(f"{tag_prefix}/confidence/hist", confidences, global_step or 0)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == 0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float((preds[mask] == y_true[mask]).mean())
        frac = float(mask.mean())
        ece += frac * abs(bin_acc - bin_conf)
        writer.add_scalars(
            f"{tag_prefix}/confidence/reliability/bin_{i}",
            {"acc": bin_acc, "conf": bin_conf, "frac": frac},
            global_step or 0,
        )

    writer.add_scalar(f"{tag_prefix}/confidence/ECE", float(ece), global_step or 0)


def run_and_log_confidence(
    trainer,
    eval_dataset,
    writer: SummaryWriter,
    tag_prefix: str,
    num_bins: int = 10,
) -> None:
    pred_output = trainer.predict(eval_dataset)
    logits_np = pred_output.predictions if not isinstance(pred_output.predictions, (list, tuple)) else pred_output.predictions[0]
    labels_np = pred_output.label_ids
    _log_confidence_metrics(
        writer=writer,
        tag_prefix=tag_prefix,
        logits=logits_np,
        labels=labels_np,
        num_bins=num_bins,
        global_step=getattr(trainer.state, "global_step", 0),
    )


