import os
import json
import numpy as np
import logging
import config
from logging_config import setup_logging
from generate_synthetic_timeseries import generate_synthetic_timeseries
from create_sliding_window import create_multiscale_sliding_window
from collections import Counter


def main() -> None:
    """Generates synthetic data with embargo-gapped chronological train/val/test splits."""
    setup_logging('prepare_data')

    logging.info("Creating directories...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    logging.info("Generating synthetic data (4 incident types, heavy-tailed noise, drift, false precursors)...")
    t, series, labels, incident_metadata = generate_synthetic_timeseries(
        length=config.SERIES_LENGTH,
        num_incidents=config.NUM_INCIDENTS,
        num_false_precursors=config.NUM_FALSE_PRECURSORS,
    )

    logging.info(f"Applying multi-scale sliding window (Short={config.SHORT_WINDOW}, Long={config.LONG_WINDOW}, H={config.HORIZON})...")
    X, y = create_multiscale_sliding_window(
        series, labels,
        short_w=config.SHORT_WINDOW,
        long_w=config.LONG_WINDOW,
        horizon=config.HORIZON
    )

    logging.info(f"Full Dataset — Class distribution: {Counter(y)}")

    gap = config.EMBARGO_GAP
    total = len(X)
    train_end = int(total * 0.70)
    val_start = train_end + gap
    val_end = int(total * 0.85)
    test_start = val_end + gap

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]

    X_test = X[test_start:]
    y_test = y[test_start:]

    logging.info(f"Embargo gap: {gap} samples between each split")
    logging.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logging.info(f"Train classes: {Counter(y_train)}")
    logging.info(f"Val classes:   {Counter(y_val)}")
    logging.info(f"Test classes:  {Counter(y_test)}")

    test_series_start = test_start + config.LONG_WINDOW
    t_test = t[test_series_start: test_series_start + len(y_test)]
    series_test = series[test_series_start: test_series_start + len(y_test)]

    test_meta = [m for m in incident_metadata if m["start"] >= test_series_start and m["end"] <= test_series_start + len(y_test)]

    logging.info("Saving datasets...")
    np.save(os.path.join(config.DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(config.DATA_DIR, 't_test.npy'), t_test)
    np.save(os.path.join(config.DATA_DIR, 'series_test.npy'), series_test)

    with open(os.path.join(config.DATA_DIR, 'incident_metadata.json'), 'w') as f:
        json.dump(incident_metadata, f, indent=2)

    with open(os.path.join(config.DATA_DIR, 'test_incident_metadata.json'), 'w') as f:
        json.dump(test_meta, f, indent=2)

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()