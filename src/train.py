import numpy as np
import joblib
import os
import logging
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Loading datasets...")
    try:
        X_train = np.load(os.path.join(config.DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.DATA_DIR, 'y_train.npy'))
    except FileNotFoundError:
        logging.error("Data not found! Run prepare_data.py first.")
        return

    logging.info("Initializing Random Forest model and parameter grid...")
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    logging.info("Starting GridSearchCV (Hyperparameter tuning)...")
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