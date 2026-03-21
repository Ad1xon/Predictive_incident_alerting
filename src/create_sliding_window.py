import numpy as np
from numpy.typing import NDArray

def calculate_trend(y: NDArray[np.float64]) -> float:
    """Calculates the linear slope (trend) of a 1-D array using a first-degree polynomial fit."""
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0

def create_multiscale_sliding_window(
    series: NDArray[np.float64],
    labels: NDArray[np.int_],
    short_w: int = 20,
    long_w: int = 100,
    horizon: int = 15,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Extracts multiscale sliding-window features and forward-looking incident labels from a time series."""
    X, y = [], []

    for i in range(len(series) - long_w - horizon):
        end_idx = i + long_w
        short_start_idx = end_idx - short_w

        short_window = series[short_start_idx : end_idx]
        long_window = series[i : end_idx]

        short_features = [
            np.mean(short_window),
            np.percentile(short_window, 90),
            np.percentile(short_window, 99),
            np.min(short_window),
            calculate_trend(short_window)
        ]

        long_features = [
            np.mean(long_window),
        ]

        mean_ratio = np.mean(short_window) / (np.mean(long_window) + 1e-5)

        combined_features = short_features + long_features + [mean_ratio]
        X.append(combined_features)

        future_labels = labels[end_idx : end_idx + horizon]
        y.append(1 if np.sum(future_labels) > 0 else 0)

    return np.array(X), np.array(y)