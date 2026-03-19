import os

SERIES_LENGTH = 3000
NUM_INCIDENTS = 15

SHORT_WINDOW = 20
LONG_WINDOW = 100
HORIZON = 10

# Post-processing
COOLDOWN_STEPS = 25

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
MODEL_PATH = os.path.join(MODELS_DIR, 'predictive_model.pkl')

FEATURE_NAMES = [
    'Short_Mean', 'Short_Std', 'Short_Max', 'Short_Min', 'Short_Trend',
    'Long_Mean', 'Long_Std', 'Long_Max', 'Long_Min', 'Long_Trend'
]