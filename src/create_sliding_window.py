import numpy as np

def create_multiscale_sliding_window(series, labels, short_w=20, long_w=100, horizon=15):
    """
    Transforms a 1D time series into a multiscale feature matrix X.
    Extracts features from both a short-term window and a long-term window
    ending at the exact same timestep to capture local and global patterns.
    """
    X, y = [], []

    for i in range(len(series) - long_w - horizon):
        end_idx = i + long_w
        short_start_idx = end_idx - short_w

        short_window = series[short_start_idx : end_idx]
        long_window = series[i : end_idx]

        short_std = np.std(short_window)
        long_std = np.std(long_window)

        short_features = [
            np.mean(short_window), short_std,
            np.max(short_window), np.min(short_window),
            short_window[-1] - short_window[0]
        ]

        long_features = [
            np.mean(long_window), long_std,
            np.max(long_window), np.min(long_window),
            long_window[-1] - long_window[0]
        ]

        variance_ratio = short_std / (long_std + 1e-5)

        combined_features = short_features + long_features + [variance_ratio]
        X.append(combined_features)

        future_labels = labels[end_idx : end_idx + horizon]
        y.append(1 if np.sum(future_labels) > 0 else 0)

    return np.array(X), np.array(y)