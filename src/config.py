import os

SEED = 167
SERIES_LENGTH = 15_000
NUM_INCIDENTS = 50
NUM_FALSE_PRECURSORS = 20

DIURNAL_PERIOD = 1440
MAX_CAPACITY = 100
INCIDENT_BUFFER = 40
EDGE_BUFFER = 150
MIN_INC_DURATION = 30
MAX_INC_DURATION = 90
PRECURSOR_LEN = 20

NOISE_STD = 6.0
HEAVY_TAIL_DF = 4
DRIFT_STD = 0.15

INCIDENT_TYPES = ["traffic_spike", "resource_saturation", "memory_leak", "cascading_failure"]

SHORT_WINDOW = 20
LONG_WINDOW = 100
HORIZON = 10

EMBARGO_GAP = LONG_WINDOW + HORIZON

COOLDOWN_STEPS = 25

TARGET_PRECISION = 0.90
CV_FOLDS = 5

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

MODEL_PATH = os.path.join(MODELS_DIR, 'predictive_model.pkl')

FEATURE_NAMES = [
    'Short_Mean', 'Short_p90', 'Short_p99', 'Short_Min', 'Short_Trend',
    'Short_Std', 'Short_Kurtosis',
    'Long_Mean', 'Long_Std', 'Long_Trend',
    'Mean_Ratio', 'Std_Ratio'
]