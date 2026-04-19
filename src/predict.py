import numpy as np
import joblib
import json
import os
import logging
import config
from logging_config import setup_logging
from rule_based import RuleBasedBaseline  # noqa: F401
from create_sliding_window import create_multiscale_sliding_window
from evaluation.alerting import apply_alarm_cooldown, find_f1_optimal_threshold


def load_best_model():
    """Loads the best model (by F1) from the saved model comparison results."""
    comparison_path = os.path.join(config.RESULTS_DIR, 'model_comparison.txt')
    if not os.path.exists(comparison_path):
        raise FileNotFoundError("Run evaluate.py first to generate model comparison results.")

    model_files = {
        'Rule-Based': os.path.join(config.MODELS_DIR, 'rule_based.pkl'),
        'Logistic Regression': os.path.join(config.MODELS_DIR, 'logistic_regression.pkl'),
        'Random Forest': config.MODEL_PATH,
        'Gradient Boosting': os.path.join(config.MODELS_DIR, 'gradient_boosting.pkl'),
    }

    best_f1 = -1
    best_name = None
    with open(comparison_path, 'r') as f:
        for line in f:
            for name, path in model_files.items():
                if line.strip().startswith(name):
                    parts = line.strip().split('|')
                    f1_val = float(parts[5].strip())
                    if f1_val > best_f1:
                        best_f1 = f1_val
                        best_name = name

    model = joblib.load(model_files[best_name])
    logging.info(f"Loaded best model: {best_name} (F1={best_f1:.4f})")
    return model, best_name


def predict_on_series(model, series: np.ndarray, labels: np.ndarray = None) -> dict:
    """Runs the full inference pipeline: feature extraction -> prediction -> cooldown filtering."""
    if labels is None:
        labels = np.zeros(len(series), dtype=int)

    X, y_true = create_multiscale_sliding_window(
        series, labels,
        short_w=config.SHORT_WINDOW,
        long_w=config.LONG_WINDOW,
        horizon=config.HORIZON,
    )

    y_probs = model.predict_proba(X)[:, 1]

    X_val = np.load(os.path.join(config.DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(config.DATA_DIR, 'y_val.npy'))
    val_probs = model.predict_proba(X_val)[:, 1]
    f1_thresh = find_f1_optimal_threshold(y_val, val_probs)

    y_pred = (y_probs >= f1_thresh).astype(int)
    y_pred_filtered = apply_alarm_cooldown(y_pred, config.COOLDOWN_STEPS)

    alert_indices = np.where(y_pred_filtered == 1)[0]
    alert_times = alert_indices + config.LONG_WINDOW

    logging.info(f"Threshold: {f1_thresh:.3f}")
    logging.info(f"Raw predictions: {np.sum(y_pred)} positive out of {len(y_pred)}")
    logging.info(f"After cooldown: {np.sum(y_pred_filtered)} alerts")
    logging.info(f"Alert timestamps (raw series index): {alert_times.tolist()}")

    return {
        'threshold': f1_thresh,
        'y_probs': y_probs,
        'y_pred_raw': y_pred,
        'y_pred_filtered': y_pred_filtered,
        'alert_indices': alert_times,
        'num_alerts': int(np.sum(y_pred_filtered)),
    }


def main() -> None:
    """Demonstrates end-to-end inference: loads best model, generates new data, runs predictions."""
    setup_logging('predict')

    logging.info("=" * 60)
    logging.info("INFERENCE PIPELINE DEMONSTRATION")
    logging.info("=" * 60)

    model, model_name = load_best_model()
    logging.info(f"Using model: {model_name}")

    logging.info("\n--- Generating new unseen telemetry (seed offset for novelty) ---")
    np.random.seed(config.SEED + 999)

    from generate_synthetic_timeseries import generate_synthetic_timeseries
    t, series, labels, metadata = generate_synthetic_timeseries(length=3000, num_incidents=10, num_false_precursors=5)

    logging.info(f"Generated {len(metadata)} incidents: {[m['type'] for m in metadata]}")

    result = predict_on_series(model, series, labels)

    logging.info(f"\n--- PREDICTION SUMMARY ---")
    logging.info(f"Model: {model_name}")
    logging.info(f"Series length: {len(series)}")
    logging.info(f"Incidents in data: {len(metadata)}")
    logging.info(f"Alerts raised: {result['num_alerts']}")

    detected = 0
    for m in metadata:
        alert_in_range = any(m['start'] - config.HORIZON <= idx <= m['end'] for idx in result['alert_indices'])
        status = "DETECTED" if alert_in_range else "MISSED"
        if alert_in_range:
            detected += 1
        logging.info(f"  {m['type']:25s} [{m['start']}-{m['end']}] -> {status}")

    logging.info(f"\nDetection rate: {detected}/{len(metadata)} ({detected/len(metadata):.0%})")
    logging.info("Inference pipeline complete.")


if __name__ == "__main__":
    main()
