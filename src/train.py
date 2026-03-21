import numpy as np
import joblib
import os
import logging
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """Loads training data, performs hyperparameter tuning with TimeSeriesSplit, and saves the best Random Forest model."""
    logging.info("Loading datasets...")
    try:
        X_train = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.DATA_DIR, 'y_train.npy'))
    except FileNotFoundError:
        logging.error("Data not found! Run prepare_data.py first.")
        return

    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)

    logging.info("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=tscv, scoring='average_precision')
    logging.info(f"LR Baseline PR-AUC: {np.mean(lr_scores):.4f}")

    logging.info("Initializing Random Forest model...")
    base_model = RandomForestClassifier(random_state=config.SEED, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    logging.info("Starting GridSearchCV (Optimizing for PR-AUC)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")

    logging.info("Saving optimized model...")
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, config.MODEL_PATH)
    logging.info("Model training complete.")

if __name__ == "__main__":
    main()