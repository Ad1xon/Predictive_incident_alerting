import os
import sys
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.generate_synthetic_timeseries import generate_synthetic_timeseries
from src.create_sliding_window import create_multiscale_sliding_window, calculate_trend
from src.evaluation.alerting import apply_alarm_cooldown, find_optimal_threshold, find_f1_optimal_threshold
from src.rule_based import RuleBasedBaseline


def test_generate_synthetic_timeseries_output_shape() -> None:
    """Verifies generator outputs correct length and injects anomalies with metadata."""
    t, series, labels, metadata = generate_synthetic_timeseries(length=1000, num_incidents=3)

    assert len(t) == 1000
    assert len(series) == 1000
    assert len(labels) == 1000
    assert np.sum(labels) > 0
    assert set(np.unique(labels)).issubset({0, 1})
    assert len(metadata) > 0
    assert all('type' in m and 'start' in m and 'end' in m for m in metadata)


def test_generate_incident_types() -> None:
    """Verifies that multiple incident types are generated across a large enough series."""
    t, series, labels, metadata = generate_synthetic_timeseries(length=5000, num_incidents=40)

    generated_types = {m['type'] for m in metadata}
    assert len(generated_types) >= 2


def test_generate_false_precursors() -> None:
    """Verifies that false precursors do not introduce incident labels."""
    t, series, labels, metadata = generate_synthetic_timeseries(length=2000, num_incidents=3, num_false_precursors=10)

    incident_count_from_meta = sum(m['end'] - m['start'] for m in metadata)
    assert np.sum(labels) <= incident_count_from_meta


def test_calculate_trend() -> None:
    """Validates the linear slope for a simple ascending array."""
    y_up = np.array([1, 2, 3, 4, 5])
    assert calculate_trend(y_up) > 0


def test_create_multiscale_sliding_window_output_shape() -> None:
    """Ensures 12-feature matrix dimensions are correct."""
    series = np.random.normal(50, 5, 300)
    labels = np.zeros(300, dtype=int)
    short_w, long_w, horizon = 10, 50, 5

    X, y = create_multiscale_sliding_window(series, labels, short_w, long_w, horizon)

    expected_rows = len(series) - long_w - horizon
    expected_features = 12

    assert X.shape == (expected_rows, expected_features)
    assert y.shape == (expected_rows,)


def test_create_multiscale_sliding_window_label_correctness() -> None:
    """Validates forward-looking label flags the correct horizon."""
    series = np.zeros(200)
    labels = np.zeros(200, dtype=int)
    labels[100] = 1

    short_w, long_w, horizon = 10, 50, 5
    X, y = create_multiscale_sliding_window(series, labels, short_w, long_w, horizon)

    assert np.sum(y) == 5


def test_apply_alarm_cooldown() -> None:
    """Validates cooldown suppression logic."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1])
    filtered = apply_alarm_cooldown(y_pred, cooldown_steps=2)

    expected = np.array([0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(filtered, expected)


def test_find_optimal_threshold_edge_cases() -> None:
    """Validates threshold fallback when precision targets are unreachable."""
    y_true = np.array([0, 0, 1, 1])

    y_probs_poor = np.array([0.9, 0.8, 0.1, 0.2])
    thresh_poor = find_optimal_threshold(y_true, y_probs_poor, target_precision=1.0)
    assert thresh_poor == 0.9

    y_true_neg = np.array([0, 0, 0, 0])
    y_probs_neg = np.array([0.1, 0.2, 0.3, 0.4])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thresh_neg = find_optimal_threshold(y_true_neg, y_probs_neg, target_precision=0.90)

    assert thresh_neg == 0.4


def test_find_f1_optimal_threshold() -> None:
    """Validates F1-optimal threshold returns value between 0 and 1."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6])
    thresh = find_f1_optimal_threshold(y_true, y_probs)

    assert 0.0 <= thresh <= 1.0


def test_rule_based_baseline() -> None:
    """Validates rule-based baseline flags high Mean_Ratio or Short_p99 samples."""
    import src.config as cfg
    n_features = len(cfg.FEATURE_NAMES)
    X_normal = np.zeros((5, n_features))
    X_normal[:, cfg.FEATURE_NAMES.index('Mean_Ratio')] = 1.0
    X_normal[:, cfg.FEATURE_NAMES.index('Short_p99')] = 50.0

    X_anomaly = np.zeros((3, n_features))
    X_anomaly[:, cfg.FEATURE_NAMES.index('Mean_Ratio')] = 1.5
    X_anomaly[:, cfg.FEATURE_NAMES.index('Short_p99')] = 90.0

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    model = RuleBasedBaseline()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds[:5] == 0)
    assert np.all(preds[5:] == 1)


def test_end_to_end_smoke_test() -> None:
    """Runs full pipeline: Generate -> Window -> Train -> Predict on separate val set."""
    t, series, labels, metadata = generate_synthetic_timeseries(length=800, num_incidents=3)
    X, y = create_multiscale_sliding_window(series, labels, short_w=10, long_w=30, horizon=5)

    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    thresh = find_f1_optimal_threshold(y_val, probs)

    assert thresh is not None
    assert 0.0 <= thresh <= 1.0