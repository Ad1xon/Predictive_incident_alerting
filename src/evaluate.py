import numpy as np
import joblib
import os
import logging
import config
from visualization import plot_predictions, plot_feature_importances
from sklearn.metrics import classification_report, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def apply_alarm_cooldown(y_pred, cooldown_steps):
    filtered_pred = np.copy(y_pred)
    cooldown_counter = 0

    for i in range(1, len(filtered_pred)):
        if cooldown_counter > 0:
            filtered_pred[i] = 0
            cooldown_counter -= 1
        elif filtered_pred[i - 1] == 1 and filtered_pred[i] == 0:
            cooldown_counter = cooldown_steps

    return filtered_pred


def find_optimal_threshold(y_true, y_probs, target_precision=0.90):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(precisions >= target_precision)[0]

    if len(valid_idx) == 0:
        logging.warning(f"Target precision {target_precision} not reached. Defaulting to 0.5")
        return 0.5

    best_idx = valid_idx[0]
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    return thresholds[best_idx]


def main():
    logging.info("Loading test data...")
    try:
        model = joblib.load(config.MODEL_PATH)
        X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.DATA_DIR, 'y_test.npy'))
        t_test = np.load(os.path.join(config.DATA_DIR, 't_test.npy'))
        series_test = np.load(os.path.join(config.DATA_DIR, 'series_test.npy'))
    except FileNotFoundError:
        logging.error("Missing files. Run prepare_data.py and train.py first.")
        return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    y_probs = model.predict_proba(X_test)[:, 1]

    optimal_thresh = find_optimal_threshold(y_test, y_probs, target_precision=0.88)
    logging.info(f"Optimal threshold calculated: {optimal_thresh:.3f}")

    y_pred_raw = (y_probs >= optimal_thresh).astype(int)
    y_pred_filtered = apply_alarm_cooldown(y_pred_raw, config.COOLDOWN_STEPS)

    report = classification_report(y_test, y_pred_filtered, labels=[0, 1], target_names=["Normal", "Incident"])
    logging.info(f"\nCLASSIFICATION REPORT (Threshold: {optimal_thresh:.2f})\n{report}")

    with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Optimal Threshold: {optimal_thresh:.3f}\n\n")
        f.write(report)

    plot_predictions(t_test, series_test, y_test, y_pred_filtered, title=f"Predictive Maintenance: Dynamic Threshold ({optimal_thresh:.2f})", save_path=os.path.join(
        config.RESULTS_DIR, 'predictions_plot.png'))

    plot_feature_importances(model, config.FEATURE_NAMES, save_path=os.path.join(config.RESULTS_DIR, 'feature_importances.png'))

    logging.info(f"Evaluation complete")

if __name__ == "__main__":
    main()