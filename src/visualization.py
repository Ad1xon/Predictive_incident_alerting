import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def plot_predictions(t_test, series_test, y_test, y_pred_filtered, title="Predictive Maintenance Alerts", save_path=None):
    plt.figure(figsize=(15, 5))
    plt.plot(t_test, series_test, label='Sensor Signal', color='blue', alpha=0.6)

    plt.fill_between(t_test, series_test.min() - 5, series_test.max() + 5, where=(y_test == 1), color='red', alpha=0.3, label='Actual Incident (Target Horizon)')

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
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feature_importances(model, feature_names, save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    bars = plt.bar(range(len(importances)), sorted_importances, color='teal', align='center')
    plt.xticks(range(len(importances)), sorted_names, rotation=15)
    plt.ylabel('Relative Importance (Sum = 1.0)')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, optimal_thresh: float, save_path: str = None) -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', label='PR Curve', linewidth=2)

    idx = np.argmin(np.abs(thresholds - optimal_thresh)) if len(thresholds) > 0 else 0
    plt.scatter(recalls[idx], precisions[idx], color='red', s=100, label=f'Chosen Threshold ({optimal_thresh:.2f})',
                zorder=5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", save_path: str = None) -> None:
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Incident"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()