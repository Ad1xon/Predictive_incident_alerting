import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_predictions(
    t_test: NDArray[np.int_],
    series_test: NDArray[np.float64],
    y_test: NDArray[np.int_],
    y_pred_filtered: NDArray[np.int_],
    title: str = "Predictive Maintenance Alerts",
    save_path: str | None = None,
) -> None:
    """Plots the sensor signal with actual incident regions and model alarm markers overlaid."""
    plt.figure(figsize=(15, 5))
    plt.plot(t_test, series_test, label='Sensor Signal', color='blue', alpha=0.6)

    y_plot = np.copy(y_test)
    changes = np.diff(y_plot.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if y_plot[0] == 1:
        starts = np.insert(starts, 0, 0)
    if y_plot[-1] == 1:
        ends = np.append(ends, len(y_plot))
    for s, e in zip(starts, ends):
        if e - s < 3:
            y_plot[s:e] = 0

    plt.fill_between(t_test, series_test.min() - 5, series_test.max() + 5, where=(y_plot == 1), color='red', alpha=0.3, label='Actual Incident')

    pred_incidents_t = t_test[y_pred_filtered == 1]
    pred_incidents_val = series_test[y_pred_filtered == 1]

    if len(pred_incidents_t) > 0:
        plt.scatter(pred_incidents_t, pred_incidents_val, color='orange', edgecolor='black',
                    s=40, label='Model Alarm', zorder=5)

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Value')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_feature_importances(importances: NDArray[np.float64], feature_names: list[str], title: str = "Feature Importances", save_path: str | None = None) -> None:
    """Plots a ranked bar chart of model feature importances."""
    indices = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    bars = plt.bar(range(len(importances)), sorted_importances, color='teal', align='center')
    plt.xticks(range(len(importances)), sorted_names, rotation=25, ha='right')
    plt.ylabel('Relative Importance')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_pr_curve(y_true: NDArray[np.int_], y_probs: NDArray[np.float64], optimal_thresh: float, save_path: str | None = None) -> None:
    """Plots the Precision-Recall curve with the chosen operating threshold highlighted."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', label='PR Curve', linewidth=2)

    idx = np.argmin(np.abs(thresholds - optimal_thresh)) if len(thresholds) > 0 else 0
    plt.scatter(recalls[idx], precisions[idx], color='red', s=100, label=f'Threshold ({optimal_thresh:.2f})', zorder=5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: NDArray[np.int_], y_pred: NDArray[np.int_], title: str = "Confusion Matrix", save_path: str | None = None) -> None:
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Incident"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_model_comparison(results: dict[str, dict], save_path: str | None = None) -> None:
    """Plots a grouped bar chart comparing all models across PR-AUC, ROC-AUC, Precision, Recall, and F1."""
    model_names = list(results.keys())
    metrics = ['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1']
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_names):
        values = [results[model].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)], edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_error_analysis(X_test: NDArray[np.float64], y_test: NDArray[np.int_], y_pred: NDArray[np.int_], feature_names: list[str], save_path: str | None = None) -> None:
    """Plots boxplots of top feature distributions segmented by TP/FP/TN/FN prediction categories."""
    tp_mask = (y_test == 1) & (y_pred == 1)
    fp_mask = (y_test == 0) & (y_pred == 1)
    tn_mask = (y_test == 0) & (y_pred == 0)
    fn_mask = (y_test == 1) & (y_pred == 0)

    key_features = ['Mean_Ratio', 'Short_p99', 'Short_Trend', 'Short_Std']
    key_indices = [feature_names.index(f) for f in key_features if f in feature_names]
    key_labels = [feature_names[i] for i in key_indices]

    fig, axes = plt.subplots(1, len(key_indices), figsize=(4 * len(key_indices), 5))
    if len(key_indices) == 1:
        axes = [axes]

    categories = ['TP', 'FP', 'TN', 'FN']
    masks = [tp_mask, fp_mask, tn_mask, fn_mask]
    colors_map = ['#4CAF50', '#FF5722', '#90CAF9', '#FFC107']

    for ax_idx, feat_idx in enumerate(key_indices):
        data = []
        labels_used = []
        colors_used = []
        for cat, mask, color in zip(categories, masks, colors_map):
            vals = X_test[mask, feat_idx]
            if len(vals) > 0:
                data.append(vals)
                labels_used.append(cat)
                colors_used.append(color)

        bp = axes[ax_idx].boxplot(data, labels=labels_used, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_used):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[ax_idx].set_title(key_labels[ax_idx], fontsize=10)
        axes[ax_idx].grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle('Error Analysis: Feature Distributions by Prediction Category', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_threshold_sensitivity(y_true: NDArray[np.int_], y_probs: NDArray[np.float64], optimal_thresh: float, save_path: str | None = None) -> None:
    """Plots precision and recall as functions of the decision threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], color='blue', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], color='green', label='Recall', linewidth=2)
    plt.axvline(x=optimal_thresh, color='red', linestyle='--', label=f'Operating Threshold ({optimal_thresh:.2f})')

    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Sensitivity Analysis')
    plt.legend(loc='center left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()