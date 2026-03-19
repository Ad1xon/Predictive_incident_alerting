import os
import numpy as np
import logging
import config
from generate_synthetic_timeseries import generate_synthetic_timeseries
from create_sliding_window import create_multiscale_sliding_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
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

    split_index = int(len(X) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    test_start_index = split_index + config.LONG_WINDOW
    t_test = t[test_start_index: test_start_index + len(y_test)]
    series_test = series[test_start_index: test_start_index + len(y_test)]

    logging.info("Saving datasets...")
    np.save(os.path.join(config.DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(config.DATA_DIR, 't_test.npy'), t_test)
    np.save(os.path.join(config.DATA_DIR, 'series_test.npy'), series_test)

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()