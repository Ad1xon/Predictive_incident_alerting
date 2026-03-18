import os
import numpy as np
from generate_synthetic_timeseries import generate_synthetic_timeseries
from create_sliding_window import create_predictive_sliding_window


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Generating synthetic data...")
    t, series, labels = generate_synthetic_timeseries(length=3000, num_incidents=15)

    W, H = 20, 10
    print(f"Applying sliding window (W={W}, H={H})...")
    X, y = create_predictive_sliding_window(series, labels, window_size=W, horizon=H)

    split_index = int(len(X) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    test_start_index = split_index + W
    t_test = t[test_start_index: test_start_index + len(y_test)]
    series_test = series[test_start_index: test_start_index + len(y_test)]

    print("Saving datasets to /data/ directory...")
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    np.save('data/t_test.npy', t_test)
    np.save('data/series_test.npy', series_test)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()