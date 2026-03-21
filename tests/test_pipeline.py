import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.generate_synthetic_timeseries import generate_synthetic_timeseries
from src.create_sliding_window import create_multiscale_sliding_window, calculate_trend
from src.evaluate import find_optimal_threshold, apply_alarm_cooldown


def test_generate_synthetic_timeseries_label_distribution() -> None:
    """Verifies that the generator outputs the correct length and actually injects anomalies."""
    t, series, labels = generate_synthetic_timeseries(length=1000, num_incidents=2)

    assert len(t) == 1000
    assert len(series) == 1000
    assert len(labels) == 1000
    assert np.sum(labels) > 0
    assert set(np.unique(labels)).issubset({0, 1})


def test_calculate_trend() -> None:
    """Validates the linear slope calculation for a simple ascending array."""
    y_up = np.array([1, 2, 3, 4, 5])

    assert calculate_trend(y_up) > 0


def test_create_multiscale_sliding_window_output_shape() -> None:
    """Ensures the feature matrix dimensions match the expected feature count and series length."""
    series = np.ones(200)
    labels = np.zeros(200)
    short_w, long_w, horizon = 10, 50, 5

    X, y = create_multiscale_sliding_window(series, labels, short_w, long_w, horizon)

    expected_rows = len(series) - long_w - horizon
    expected_features = 7  # 5 short + 1 long + 1 ratio

    assert X.shape == (expected_rows, expected_features)
    assert y.shape == (expected_rows,)


def test_create_multiscale_sliding_window_label_correctness() -> None:
    """Validates that the target 'y' correctly identifies anomalies in the future horizon."""
    series = np.zeros(200)
    labels = np.zeros(200)

    labels[100] = 1

    short_w, long_w, horizon = 10, 50, 5
    X, y = create_multiscale_sliding_window(series, labels, short_w, long_w, horizon)

    assert np.sum(y) == 5


def test_apply_alarm_cooldown() -> None:
    """Validates that subsequent alerts are suppressed immediately upon trigger."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1])
    filtered = apply_alarm_cooldown(y_pred, cooldown_steps=2)

    expected = np.array([0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(filtered, expected)


def test_find_optimal_threshold_edge_cases() -> None:
    """Validates threshold fallback behavior when precision targets are unreachable."""
    y_true = np.array([0, 0, 1, 1])

    y_probs_poor = np.array([0.9, 0.8, 0.1, 0.2])
    thresh_poor = find_optimal_threshold(y_true, y_probs_poor, target_precision=1.0)

    assert thresh_poor == 0.9

    y_true_neg = np.array([0, 0, 0, 0])
    y_probs_neg = np.array([0.1, 0.2, 0.3, 0.4])
    thresh_neg = find_optimal_threshold(y_true_neg, y_probs_neg, target_precision=0.90)

    assert thresh_neg == 0.4


def test_end_to_end_smoke_test() -> None:
    """Simulates the entire pipeline: Generate -> Window -> Train -> Evaluate."""
    t, series, labels = generate_synthetic_timeseries(length=500, num_incidents=2)
    X, y = create_multiscale_sliding_window(series, labels, short_w=10, long_w=30, horizon=5)

    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    thresh = find_optimal_threshold(y, probs, target_precision=0.5)

    assert thresh is not None
    assert 0.0 <= thresh <= 1.0