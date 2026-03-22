import numpy as np
from numpy.typing import NDArray
import joblib
import os
import logging
import config
from visualization import plot_predictions, plot_feature_importances, plot_pr_curve, plot_confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def apply_alarm_cooldown(y_pred: NDArray[np.int_], cooldown_steps: int) -> NDArray[np.int_]:
    """Suppresses consecutive alarm predictions for a set number of steps after each initial trigger."""
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
    """Finds the lowest probability threshold that achieves the target precision on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(precisions >= target_precision)[0]

    if len(valid_idx) == 0:
        logging.warning(f"Target precision {target_precision} not reached. Defaulting to 0.5")
        return 0.5

    best_idx = valid_idx[0]
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    return thresholds[best_idx]


def main() -> None:
    """Evaluates the trained model on the test set with dynamic thresholding and alarm cooldown."""
    logging.info("Loading validation and test data...")
    try:
        model = joblib.load(config.MODEL_PATH)
        X_val = np.load(os.path.join(config.DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(config.DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.DATA_DIR, 'y_test.npy'))
        t_test = np.load(os.path.join(config.DATA_DIR, 't_test.npy'))
        series_test = np.load(os.path.join(config.DATA_DIR, 'series_test.npy'))
    except FileNotFoundError:
        logging.error("Missing files. Run prepare_data.py and train.py first.")
        return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    y_probs_val = model.predict_proba(X_val)[:, 1]
    optimal_thresh = find_optimal_threshold(y_val, y_probs_val, target_precision=config.TARGET_PRECISION)
    logging.info(f"Optimal threshold calculated from VALIDATION set: {optimal_thresh:.3f}")

    y_probs_test = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_probs_test)
    roc_auc = roc_auc_score(y_test, y_probs_test)
    logging.info(f"Test Set AUC Metrics - PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f}")

    y_pred_raw = (y_probs_test >= optimal_thresh).astype(int)
    y_pred_filtered = apply_alarm_cooldown(y_pred_raw, config.COOLDOWN_STEPS)

    model_report = classification_report(y_test, y_pred_raw, labels=[0, 1], target_names=["Normal", "Incident"])

    logging.info(f"\nMODEL EVALUATION ON TEST SET (Raw Predictions):\n{model_report}")

    with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Optimal Threshold (from Validation Set): {optimal_thresh:.3f}\n\n")
        f.write(model_report)

    plot_predictions(t_test, series_test, y_test, y_pred_filtered,
                     title=f"Predictive Maintenance: Dynamic Threshold ({optimal_thresh:.2f})",
                     save_path=os.path.join(config.RESULTS_DIR, 'predictions_plot.png'))
    plot_pr_curve(y_test, y_probs_test, optimal_thresh, save_path=os.path.join(config.RESULTS_DIR, 'pr_curve.png'))
    plot_feature_importances(model, config.FEATURE_NAMES,
                             save_path=os.path.join(config.RESULTS_DIR, 'feature_importances.png'))
    plot_confusion_matrix(y_test, y_pred_raw, title=f"Confusion Matrix (Thresh: {optimal_thresh:.2f})",
                          save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))

    logging.info(f"Evaluation complete")

if __name__ == "__main__":
    main()