import numpy as np
import config


class RuleBasedBaseline:
    """Threshold-based anomaly detector using Mean_Ratio and Short_p99."""

    def __init__(self, mean_ratio_thresh: float = 1.15, p99_thresh: float = 75.0):
        self.mean_ratio_thresh = mean_ratio_thresh
        self.p99_thresh = p99_thresh
        self.mean_ratio_idx = config.FEATURE_NAMES.index('Mean_Ratio')
        self.p99_idx = config.FEATURE_NAMES.index('Short_p99')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RuleBasedBaseline':
        """No-op fit to maintain scikit-learn compatible interface."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts 1 if Mean_Ratio OR Short_p99 exceeds configured thresholds."""
        ratio_flag = X[:, self.mean_ratio_idx] > self.mean_ratio_thresh
        p99_flag = X[:, self.p99_idx] > self.p99_thresh
        return (ratio_flag | p99_flag).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns pseudo-probabilities based on normalized feature distance from thresholds."""
        ratio_score = np.clip((X[:, self.mean_ratio_idx] - 1.0) / 0.3, 0, 1)
        p99_score = np.clip((X[:, self.p99_idx] - 60.0) / 30.0, 0, 1)
        prob_pos = np.maximum(ratio_score, p99_score)
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])
