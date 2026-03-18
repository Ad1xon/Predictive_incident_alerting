from create_sliding_window import create_predictive_sliding_window
from generate_synthetic_timeseries import generate_synthetic_timeseries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main():
    print("Generating synthetic data...")
    t, series, labels = generate_synthetic_timeseries(length=3000, num_incidents=15)

    W = 20
    H = 10
    print(f"Applying sliding window (W={W}, H={H})...")
    X, y = create_predictive_sliding_window(series, labels, window_size=W, horizon=H)

    # Chronological train/test split (no shuffling for time series!)
    split_index = int(len(X) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("Training the base Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    print("\nCLASSIFICATION REPORT (Test Set)")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Normal (0)", "Incident (1)"]))


if __name__ == "__main__":
    main()