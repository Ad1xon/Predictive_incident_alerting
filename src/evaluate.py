import numpy as np
import joblib
from sklearn.metrics import classification_report


def main():
    print("Loading model and test datasets...")
    model = joblib.load('models/predictive_model.pkl')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Normal", "Incident"]))


if __name__ == "__main__":
    main()