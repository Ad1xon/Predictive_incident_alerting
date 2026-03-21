import numpy as np
from src.create_sliding_window import calculate_trend
from src.evaluate import apply_alarm_cooldown


def test_calculate_trend():
    y_up = np.array([1, 2, 3, 4, 5])
    assert calculate_trend(y_up) > 0

    y_down = np.array([5, 4, 3, 2, 1])
    assert calculate_trend(y_down) < 0

    y_flat = np.array([3, 3, 3, 3, 3])
    assert calculate_trend(y_flat) == 0


def test_apply_alarm_cooldown():
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1])

    filtered = apply_alarm_cooldown(y_pred, cooldown_steps=2)

    expected = np.array([0, 1, 0, 0, 0, 1, 0])

    np.testing.assert_array_equal(filtered, expected)