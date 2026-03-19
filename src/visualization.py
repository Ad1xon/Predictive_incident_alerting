import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(t_test, series_test, y_test, y_pred_filtered,title="Predictive Maintenance: Alerts vs Actual Incidents (Debounced)"):
    plt.figure(figsize=(15, 5))
    plt.plot(t_test, series_test, label='Sensor Signal', color='blue', alpha=0.6)

    plt.fill_between(t_test, series_test.min() - 5, series_test.max() + 5,
                     where=(y_test == 1), color='red', alpha=0.3, label='Actual Incident (Target Horizon)')

    pred_incidents_t = t_test[y_pred_filtered == 1]
    pred_incidents_val = series_test[y_pred_filtered == 1]

    if len(pred_incidents_t) > 0:
        plt.scatter(pred_incidents_t, pred_incidents_val, color='orange', edgecolor='black',
                    s=40, label='Model Alarm', zorder=5)

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names):
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
    plt.show()