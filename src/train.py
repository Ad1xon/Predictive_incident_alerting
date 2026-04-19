import numpy as np
import joblib
import json
import os
import logging
import config
from logging_config import setup_logging
from rule_based import RuleBasedBaseline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score


def main() -> None:
    """Trains 4 models (Rule-Based, LR, RF, Calibrated GBT), compares CV PR-AUC, and saves all artifacts."""
    setup_logging('train')

    logging.info("Loading datasets...")
    try:
        X_train = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.DATA_DIR, 'y_train.npy'))
    except FileNotFoundError:
        logging.error("Data not found! Run prepare_data.py first.")
        return

    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    cv_results = {}

    logging.info("=== Model 1/4: Rule-Based Baseline ===")
    rule_model = RuleBasedBaseline()
    rule_model.fit(X_train, y_train)
    rule_probs = rule_model.predict_proba(X_train)[:, 1]
    rule_prauc = average_precision_score(y_train, rule_probs)
    cv_results['Rule-Based'] = rule_prauc
    logging.info(f"Rule-Based train PR-AUC: {rule_prauc:.4f}")
    joblib.dump(rule_model, os.path.join(config.MODELS_DIR, 'rule_based.pkl'))

    logging.info("=== Model 2/4: Logistic Regression (Scaled) ===")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=config.SEED)),
    ])
    lr_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=tscv, scoring='average_precision')
    cv_results['Logistic Regression'] = float(np.mean(lr_scores))
    logging.info(f"LR CV PR-AUC: {np.mean(lr_scores):.4f} (+/- {np.std(lr_scores):.4f})")
    lr_pipeline.fit(X_train, y_train)
    joblib.dump(lr_pipeline, os.path.join(config.MODELS_DIR, 'logistic_regression.pkl'))

    logging.info("=== Model 3/4: Random Forest (GridSearchCV) ===")
    rf_base = RandomForestClassifier(random_state=config.SEED, class_weight='balanced')
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
    }
    grid_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid_rf,
        cv=tscv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1,
    )
    grid_rf.fit(X_train, y_train)
    cv_results['Random Forest'] = float(grid_rf.best_score_)
    logging.info(f"RF Best CV PR-AUC: {grid_rf.best_score_:.4f} | Params: {grid_rf.best_params_}")
    joblib.dump(grid_rf.best_estimator_, config.MODEL_PATH)

    logging.info("=== Model 4/4: Gradient Boosting (Calibrated) ===")
    gbt_base = GradientBoostingClassifier(random_state=config.SEED)
    param_grid_gbt = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    grid_gbt = GridSearchCV(
        estimator=gbt_base,
        param_grid=param_grid_gbt,
        cv=tscv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1,
    )
    grid_gbt.fit(X_train, y_train)
    logging.info(f"GBT raw CV PR-AUC: {grid_gbt.best_score_:.4f} | Params: {grid_gbt.best_params_}")

    logging.info("Calibrating GBT probabilities with isotonic regression...")
    calibrated_gbt = CalibratedClassifierCV(grid_gbt.best_estimator_, method='isotonic', cv=3)
    calibrated_gbt.fit(X_train, y_train)
    cal_probs = calibrated_gbt.predict_proba(X_train)[:, 1]
    cal_prauc = average_precision_score(y_train, cal_probs)
    cv_results['Gradient Boosting'] = float(grid_gbt.best_score_)
    logging.info(f"GBT calibrated train PR-AUC: {cal_prauc:.4f}")
    joblib.dump(calibrated_gbt, os.path.join(config.MODELS_DIR, 'gradient_boosting.pkl'))

    with open(os.path.join(config.MODELS_DIR, 'cv_comparison.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    logging.info(f"\n{'='*50}")
    logging.info("Cross-Validation PR-AUC Comparison:")
    for name, score in sorted(cv_results.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {name:25s} -> {score:.4f}")
    logging.info(f"{'='*50}")
    logging.info("Model training complete. All models saved.")


if __name__ == "__main__":
    main()