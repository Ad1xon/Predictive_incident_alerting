import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def main():
    print("Loading datasets from /data/...")
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    print("Initializing base Random Forest model and parameter grid...")
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")

    print("Saving the optimized model to /models/...")
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/predictive_model.pkl')
    print("Model training and tuning complete.")

if __name__ == "__main__":
    main()