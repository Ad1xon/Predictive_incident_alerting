import os

SERIES_LENGTH = 3000
NUM_INCIDENTS = 15

WINDOW_SIZE = 20
HORIZON = 10

# Post-processing
COOLDOWN_STEPS = 25

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
MODEL_PATH = os.path.join(MODELS_DIR, 'predictive_model.pkl')

FEATURE_NAMES = [
    'Mean', 'Standard Deviation (Std)', 'Maximum', 'Minimum', 'Difference (Trend)'
]