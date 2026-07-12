from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent

# Raw CirCor files: .txt and .wav
DATA_DIR = Path(
    "/Users/marammarzouki/CaCOM-HeartWISE/data/raw/training_data"
)

# Existing patient-level split
SPLITS_FILE = Path(
    "/Users/marammarzouki/CaCOM-HeartWISE/data/processed/patient_splits.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputsGRF"
FRAME_FEATURE_DIR = OUTPUT_DIR / "frame_features"
GRANGER_DIR = OUTPUT_DIR / "granger_edges"

LABELS_FILE = OUTPUT_DIR / "labels.csv"
ACOUSTIC_FEATURES_FILE = OUTPUT_DIR / "acoustic_patient_features.csv"
GRANGER_FEATURES_FILE = OUTPUT_DIR / "granger_patient_features.csv"

# Reuse outputs when available
FORCE_RECOMPUTE = False

# Audio
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 8

# Fast mode is recommended on a 16 GB laptop.
FAST_MODE = True

# FAST_MODE uses these 9 time series.
FAST_FEATURE_NAMES = [
    "rms",
    "centroid",
    "bandwidth",
    "zcr",
    "entropy",
    "energy_20_100",
    "energy_100_250",
    "energy_250_500",
    "energy_500_1000",
]

# Full mode additionally includes MFCC 1-8.
FULL_FEATURE_NAMES = FAST_FEATURE_NAMES + [
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
]

FEATURE_NAMES = FAST_FEATURE_NAMES if FAST_MODE else FULL_FEATURE_NAMES

# Granger
MAX_LAG = 2 if FAST_MODE else 3
P_VALUE_THRESHOLD = 0.05

# Parallel extraction / Granger jobs.
# On a 16 GB laptop, leave 1-2 cores free.
CPU_COUNT = os.cpu_count() or 4
PREPROCESS_N_JOBS = max(1, min(6, CPU_COUNT - 1))

# Random Forest tuning
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER_SEARCH = 16

# Parallelize candidate fits, not each individual forest.
SEARCH_N_JOBS = -1
FOREST_N_JOBS = 1
PRE_DISPATCH = max(2, CPU_COUNT)

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01
