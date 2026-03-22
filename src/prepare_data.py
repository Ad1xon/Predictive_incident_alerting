import os
import numpy as np
import logging
import config
from generate_synthetic_timeseries import generate_synthetic_timeseries
from create_sliding_window import create_multiscale_sliding_window
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    """Generates synthetic data, applies multiscale sliding windows, and saves train/val/test splits to disk."""
    logging.info("Creating directories...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    logging.info("Generating synthetic data...")
    t, series, labels = generate_synthetic_timeseries(length=config.SERIES_LENGTH, num_incidents=config.NUM_INCIDENTS)

    logging.info(f"Applying multi-scale sliding window (Short={config.SHORT_WINDOW}, Long={config.LONG_WINDOW}, H={config.HORIZON})...")
    X, y = create_multiscale_sliding_window(
        series, labels,
        short_w=config.SHORT_WINDOW,
        long_w=config.LONG_WINDOW,
        horizon=config.HORIZON
    )

    logging.info(f"Full Dataset Class distribution: {Counter(y)}")

    # 3-way split: 70% Train, 15% Val, 15% Test
    split_idx_1 = int(len(X) * 0.70)
    split_idx_2 = int(len(X) * 0.85)

    X_train, X_val, X_test = X[:split_idx_1], X[split_idx_1:split_idx_2], X[split_idx_2:]
    y_train, y_val, y_test = y[:split_idx_1], y[split_idx_1:split_idx_2], y[split_idx_2:]

    test_start_index = split_idx_2 + config.LONG_WINDOW
    t_test = t[test_start_index: test_start_index + len(y_test)]
    series_test = series[test_start_index: test_start_index + len(y_test)]

    logging.info("Saving datasets...")
    np.save(os.path.join(config.DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(config.DATA_DIR, 't_test.npy'), t_test)
    np.save(os.path.join(config.DATA_DIR, 'series_test.npy'), series_test)

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()