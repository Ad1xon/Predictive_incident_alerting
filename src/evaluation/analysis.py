import numpy as np
import os
from numpy.typing import NDArray
import config


def run_error_analysis(X_test: NDArray, y_test: NDArray, y_pred: NDArray, results_dir: str) -> str:
    """Writes feature-level FP/FN diagnostic report to disk and returns the report text."""
    tp_mask = (y_test == 1) & (y_pred == 1)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS — Feature Statistics by Prediction Category")
    lines.append("=" * 60)
    lines.append(f"\nTotal TP: {np.sum(tp_mask)} | FP: {np.sum(fp_mask)} | FN: {np.sum(fn_mask)}\n")

    for label, mask in [("TRUE POSITIVES", tp_mask), ("FALSE POSITIVES", fp_mask), ("FALSE NEGATIVES", fn_mask)]:
        lines.append(f"\n--- {label} ---")
        if np.sum(mask) == 0:
            lines.append("  No samples in this category.\n")
            continue
        subset = X_test[mask]
        for i, name in enumerate(config.FEATURE_NAMES):
            col = subset[:, i]
            lines.append(f"  {name:20s} | mean={np.mean(col):8.3f} | std={np.std(col):7.3f} | min={np.min(col):8.3f} | max={np.max(col):8.3f}")
        lines.append("")

    lines.append("\n--- KEY DIAGNOSTIC ---")
    if np.sum(fp_mask) > 0 and np.sum(tp_mask) > 0:
        fp_ratio = np.mean(X_test[fp_mask, config.FEATURE_NAMES.index('Mean_Ratio')])
        tp_ratio = np.mean(X_test[tp_mask, config.FEATURE_NAMES.index('Mean_Ratio')])
        lines.append(f"  FP avg Mean_Ratio: {fp_ratio:.3f} vs TP avg Mean_Ratio: {tp_ratio:.3f}")
        lines.append(f"  -> FPs triggered by {'false precursors or heavy-tail noise spikes' if fp_ratio < tp_ratio else 'genuine-looking anomaly patterns'}.")

    if np.sum(fn_mask) > 0:
        fn_short_trend = np.mean(X_test[fn_mask, config.FEATURE_NAMES.index('Short_Trend')])
        fn_long_trend = np.mean(X_test[fn_mask, config.FEATURE_NAMES.index('Long_Trend')])
        lines.append(f"  FN avg Short_Trend: {fn_short_trend:.4f} | FN avg Long_Trend: {fn_long_trend:.4f}")
        lines.append(f"  -> Missed incidents have {'subtle/slow onset (memory leaks)' if abs(fn_short_trend) < 0.3 else 'noisy precursors masked by heavy-tail background'}.")

    report_text = "\n".join(lines)

    with open(os.path.join(results_dir, 'error_analysis.txt'), 'w') as f:
        f.write(report_text)

    return report_text


def run_incident_type_analysis(y_test: NDArray, y_pred: NDArray, t_test: NDArray, test_metadata: list[dict], results_dir: str) -> str:
    """Computes per-incident-type detection rates using incident metadata."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("PER-INCIDENT-TYPE DETECTION ANALYSIS")
    lines.append("=" * 60)

    if not test_metadata:
        lines.append("  No incident metadata available for the test set region.")
        report = "\n".join(lines)
        with open(os.path.join(results_dir, 'incident_type_analysis.txt'), 'w') as f:
            f.write(report)
        return report

    type_stats = {}
    t_min = int(t_test[0])

    for meta in test_metadata:
        inc_type = meta['type']
        if inc_type not in type_stats:
            type_stats[inc_type] = {'total': 0, 'detected': 0}
        type_stats[inc_type]['total'] += 1

        inc_start = max(meta['start'] - t_min, 0)
        inc_end = min(meta['end'] - t_min, len(y_pred))

        if inc_start < len(y_pred) and inc_end > 0:
            if np.sum(y_pred[inc_start:inc_end]) > 0:
                type_stats[inc_type]['detected'] += 1

    for inc_type, stat in sorted(type_stats.items()):
        rate = stat['detected'] / stat['total'] if stat['total'] > 0 else 0
        lines.append(f"  {inc_type:25s} | Detected: {stat['detected']}/{stat['total']} ({rate:.0%})")

    report = "\n".join(lines)

    with open(os.path.join(results_dir, 'incident_type_analysis.txt'), 'w') as f:
        f.write(report)

    return report
