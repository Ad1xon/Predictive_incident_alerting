import numpy as np
import json
import joblib
import os
import logging
import config
from logging_config import setup_logging
from rule_based import RuleBasedBaseline  # noqa: F401 — required for pickle deserialization
from evaluation.alerting import apply_alarm_cooldown, find_optimal_threshold, find_f1_optimal_threshold
from evaluation.analysis import run_error_analysis, run_incident_type_analysis
from visualization import (
    plot_predictions, plot_feature_importances, plot_pr_curve,
    plot_confusion_matrix, plot_model_comparison, plot_error_analysis,
    plot_threshold_sensitivity,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_score, recall_score, f1_score,
)


def _evaluate_single_model(name: str, model, X_test, y_test, X_val, y_val) -> dict:
    """Evaluates a single model with both precision-targeted and F1-optimal thresholds."""
    y_probs_val = model.predict_proba(X_val)[:, 1]
    prec_thresh = find_optimal_threshold(y_val, y_probs_val, target_precision=config.TARGET_PRECISION)
    f1_thresh = find_f1_optimal_threshold(y_val, y_probs_val)

    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_f1 = (y_probs_test >= f1_thresh).astype(int)

    pr_auc = average_precision_score(y_test, y_probs_test)
    roc_auc = roc_auc_score(y_test, y_probs_test)
    prec = precision_score(y_test, y_pred_f1, zero_division=0)
    rec = recall_score(y_test, y_pred_f1, zero_division=0)
    f1 = f1_score(y_test, y_pred_f1, zero_division=0)

    logging.info(f"  {name:25s} | PR-AUC={pr_auc:.3f} | ROC-AUC={roc_auc:.3f} | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f} | F1-Thresh={f1_thresh:.3f} | Prec-Thresh={prec_thresh:.3f}")

    return {
        'PR-AUC': round(pr_auc, 4),
        'ROC-AUC': round(roc_auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'F1_Threshold': round(f1_thresh, 4),
        'Precision_Threshold': round(prec_thresh, 4),
        'y_pred': y_pred_f1,
        'y_probs': y_probs_test,
    }


def main() -> None:
    """Evaluates all trained models with F1-optimal thresholds, runs error analysis, and generates plots."""
    setup_logging('evaluate')

    logging.info("Loading validation and test data...")
    try:
        X_val = np.load(os.path.join(config.DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(config.DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(config.DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.DATA_DIR, 'y_test.npy'))
        t_test = np.load(os.path.join(config.DATA_DIR, 't_test.npy'))
        series_test = np.load(os.path.join(config.DATA_DIR, 'series_test.npy'))
    except FileNotFoundError:
        logging.error("Missing files. Run prepare_data.py and train.py first.")
        return

    try:
        with open(os.path.join(config.DATA_DIR, 'test_incident_metadata.json'), 'r') as f:
            test_metadata = json.load(f)
    except FileNotFoundError:
        test_metadata = []

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    model_files = {
        'Rule-Based': os.path.join(config.MODELS_DIR, 'rule_based.pkl'),
        'Logistic Regression': os.path.join(config.MODELS_DIR, 'logistic_regression.pkl'),
        'Random Forest': config.MODEL_PATH,
        'Gradient Boosting': os.path.join(config.MODELS_DIR, 'gradient_boosting.pkl'),
    }

    logging.info("\n" + "=" * 70)
    logging.info("MODEL COMPARISON ON TEST SET (F1-Optimal Threshold)")
    logging.info("=" * 70)

    comparison_results = {}
    for name, path in model_files.items():
        try:
            model = joblib.load(path)
            comparison_results[name] = _evaluate_single_model(name, model, X_test, y_test, X_val, y_val)
        except FileNotFoundError:
            logging.warning(f"  {name} model not found at {path}, skipping.")

    comparison_table_lines = []
    comparison_table_lines.append(f"{'Model':25s} | {'PR-AUC':>8s} | {'ROC-AUC':>8s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | {'F1-Thresh':>9s} | {'P90-Thresh':>10s}")
    comparison_table_lines.append("-" * 100)
    for name, res in comparison_results.items():
        comparison_table_lines.append(
            f"{name:25s} | {res['PR-AUC']:8.4f} | {res['ROC-AUC']:8.4f} | {res['Precision']:6.4f} | {res['Recall']:6.4f} | {res['F1']:6.4f} | {res['F1_Threshold']:9.4f} | {res['Precision_Threshold']:10.4f}"
        )
    comparison_text = "\n".join(comparison_table_lines)

    with open(os.path.join(config.RESULTS_DIR, 'model_comparison.txt'), 'w') as f:
        f.write(comparison_text)
    logging.info(f"\n{comparison_text}")

    plot_comparison_data = {name: {k: v for k, v in res.items() if k not in ('y_pred', 'y_probs', 'F1_Threshold', 'Precision_Threshold')} for name, res in comparison_results.items()}
    plot_model_comparison(plot_comparison_data, save_path=os.path.join(config.RESULTS_DIR, 'model_comparison.png'))

    best_model_name = max(comparison_results, key=lambda k: (comparison_results[k]['F1'], comparison_results[k]['PR-AUC']))
    best_result = comparison_results[best_model_name]
    best_model = joblib.load(model_files[best_model_name])
    logging.info(f"\nBest model (by F1): {best_model_name} (F1: {best_result['F1']:.4f}, PR-AUC: {best_result['PR-AUC']:.4f})")

    y_pred_raw = best_result['y_pred']
    y_probs_test = best_result['y_probs']
    optimal_thresh = best_result['F1_Threshold']
    y_pred_filtered = apply_alarm_cooldown(y_pred_raw, config.COOLDOWN_STEPS)

    model_report = classification_report(y_test, y_pred_raw, labels=[0, 1], target_names=["Normal", "Incident"])
    logging.info(f"\n{best_model_name} — CLASSIFICATION REPORT (Test Set):\n{model_report}")

    with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"F1-Optimal Threshold (from Validation Set): {optimal_thresh:.3f}\n")
        f.write(f"Precision-Targeted Threshold: {best_result['Precision_Threshold']:.3f}\n\n")
        f.write(model_report)

    logging.info("Running error analysis...")
    error_report = run_error_analysis(X_test, y_test, y_pred_raw, config.RESULTS_DIR)
    logging.info(f"\n{error_report}")

    logging.info("Running per-incident-type analysis...")
    type_report = run_incident_type_analysis(y_test, y_pred_raw, t_test, test_metadata, config.RESULTS_DIR)
    logging.info(f"\n{type_report}")

    logging.info("Generating plots...")
    plot_predictions(
        t_test, series_test, y_test, y_pred_filtered,
        title=f"{best_model_name}: F1-Optimal Threshold ({optimal_thresh:.2f})",
        save_path=os.path.join(config.RESULTS_DIR, 'predictions_plot.png'),
    )
    plot_pr_curve(y_test, y_probs_test, optimal_thresh, save_path=os.path.join(config.RESULTS_DIR, 'pr_curve.png'))
    plot_confusion_matrix(y_test, y_pred_raw, title=f"Confusion Matrix ({best_model_name})", save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    plot_error_analysis(X_test, y_test, y_pred_raw, config.FEATURE_NAMES, save_path=os.path.join(config.RESULTS_DIR, 'error_analysis.png'))
    plot_threshold_sensitivity(y_test, y_probs_test, optimal_thresh, save_path=os.path.join(config.RESULTS_DIR, 'threshold_sensitivity.png'))

    rf_model = joblib.load(config.MODEL_PATH)
    if hasattr(rf_model, 'feature_importances_'):
        plot_feature_importances(
            rf_model.feature_importances_, config.FEATURE_NAMES,
            title='Feature Importances',
            save_path=os.path.join(config.RESULTS_DIR, 'feature_importances.png'),
        )

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()