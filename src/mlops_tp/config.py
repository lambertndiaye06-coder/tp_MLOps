"""
config.py — Paramètres globaux du projet
"""

import os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(BASE_DIR, "crx.data")                        # chemin vers le fichier de données
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
FEATURE_SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
RUN_INFO_PATH = os.path.join(ARTIFACTS_DIR, "run_info.json")

# ── Données ───────────────────────────────────────────────────────────────────
TARGET_COLUMN = "A16"
POSITIVE_CLASS = "+"
NEGATIVE_CLASS = "-"

COLUMN_NAMES = [f"A{i}" for i in range(1, 17)]   # A1 … A16

# ── Entraînement ──────────────────────────────────────────────────────────────
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ── Modèle (XGBoost) ──────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}
