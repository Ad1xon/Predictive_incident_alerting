import numpy as np

def create_multiscale_sliding_window(series, labels, short_w=20, long_w=100, horizon=15):
    """
    Transforms a 1D time series into a multiscale feature matrix X.
    Extracts features from both a short-term window and a long-term window
    ending at the exact same timestep to capture local and global patterns.
    """
    X = []
    y = []

    for i in range(len(series) - long_w - horizon):
        end_idx = i + long_w
        short_start_idx = end_idx - short_w

        short_window = series[short_start_idx : end_idx]
        long_window = series[i : end_idx]

        # Short-term features (local anomalies/spikes)
        short_features = [
            np.mean(short_window), np.std(short_window),
            np.max(short_window), np.min(short_window),
            short_window[-1] - short_window[0]
        ]

        # Long-term features (global degradation trends)
        long_features = [
            np.mean(long_window), np.std(long_window),
            np.max(long_window), np.min(long_window),
            long_window[-1] - long_window[0]
        ]

        combined_features = short_features + long_features
        X.append(combined_features)

        future_labels = labels[end_idx : end_idx + horizon]
        if np.sum(future_labels) > 0:
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y)