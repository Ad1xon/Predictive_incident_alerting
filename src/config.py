import os

SEED = 101
SERIES_LENGTH = 3000
NUM_INCIDENTS = 15

SHORT_WINDOW = 20
LONG_WINDOW = 100
HORIZON = 10

COOLDOWN_STEPS = 25

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

MODEL_PATH = os.path.join(MODELS_DIR, 'predictive_model.pkl')

FEATURE_NAMES = [
    'Short_Mean', 'Short_p90', 'Short_p99', 'Short_Min', 'Short_Trend',
    'Long_Mean', 'Mean_Ratio'
]