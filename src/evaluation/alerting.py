import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve


def apply_alarm_cooldown(y_pred: NDArray[np.int_], cooldown_steps: int) -> NDArray[np.int_]:
    """Suppresses consecutive alarm predictions for a set number of steps after each trigger."""
    filtered_pred = np.copy(y_pred)
    cooldown_counter = 0

    for i in range(1, len(filtered_pred)):
        if cooldown_counter > 0:
            filtered_pred[i] = 0
            cooldown_counter -= 1
        elif filtered_pred[i] == 1:
            cooldown_counter = cooldown_steps

    return filtered_pred


def find_optimal_threshold(y_true: NDArray[np.int_], y_probs: NDArray[np.float64], target_precision: float = 0.90) -> float:
    """Finds the lowest threshold achieving the target precision on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(precisions >= target_precision)[0]

    if len(valid_idx) == 0:
        logging.warning(f"Target precision {target_precision} not reached. Defaulting to 0.5")
        return 0.5

    best_idx = valid_idx[0]
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    return thresholds[best_idx]


def find_f1_optimal_threshold(y_true: NDArray[np.int_], y_probs: NDArray[np.float64]) -> float:
    """Finds the threshold that maximizes F1 score on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    return thresholds[best_idx]
