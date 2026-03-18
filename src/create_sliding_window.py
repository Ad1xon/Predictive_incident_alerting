import numpy as np


def create_predictive_sliding_window(series, labels, window_size=20, horizon=15):
    """
    Transforms a 1D time series into feature matrix X and target vector y.
    Predicts if an anomaly will occur within the 'horizon' steps ahead.
    """
    X = []
    y = []

    for i in range(len(series) - window_size - horizon):
        window_data = series[i: i + window_size]

        # Feature Engineering: Extracting statistical properties
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        max_val = np.max(window_data)
        min_val = np.min(window_data)
        diff_val = window_data[-1] - window_data[0]

        features = [mean_val, std_val, max_val, min_val, diff_val]
        X.append(features)

        # Checking the horizon for any upcoming incidents
        future_labels = labels[i + window_size: i + window_size + horizon]
        if np.sum(future_labels) > 0:
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y)