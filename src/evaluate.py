import numpy as np
import joblib
import os
import logging
import config
from visualization import plot_predictions, plot_feature_importances
from sklearn.metrics import classification_report

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


def main():
    logging.info("Loading model and test datasets...")
    try:
        model = joblib.load(config.MODEL_PATH)
        X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.DATA_DIR, 'y_test.npy'))
        t_test = np.load(os.path.join(config.DATA_DIR, 't_test.npy'))
        series_test = np.load(os.path.join(config.DATA_DIR, 'series_test.npy'))
    except FileNotFoundError:
        logging.error("Missing files! Run prepare_data.py and train.py first.")
        return

    logging.info("Generating predictions and applying cooldown logic...")
    y_pred_raw = model.predict(X_test)
    y_pred_filtered = apply_alarm_cooldown(y_pred_raw, config.COOLDOWN_STEPS)

    report = classification_report(y_test, y_pred_filtered, labels=[0, 1], target_names=["Normal (0)", "Incident (1)"])
    logging.info(f"\nCLASSIFICATION REPORT (Test Set)\n{report}")

    logging.info("Generating charts...")
    plot_predictions(t_test, series_test, y_test, y_pred_filtered)
    plot_feature_importances(model, config.FEATURE_NAMES)

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()