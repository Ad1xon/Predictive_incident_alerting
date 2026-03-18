import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading datasets from /data/...")
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Saving the trained model to /models/...")
    joblib.dump(model, 'models/predictive_model.pkl')

if __name__ == "__main__":
    main()