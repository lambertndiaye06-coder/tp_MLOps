import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import subprocess
import sys

# ─────────────────────────────────────────
# Fixtures : données synthétiques
# ─────────────────────────────────────────
@pytest.fixture
def sample_data():
    """Crée un petit dataset synthétique qui imite crx.data"""
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        "A1":  np.random.choice(["b", "a"], n),
        "A2":  np.random.uniform(20, 60, n),
        "A3":  np.random.uniform(0, 10, n),
        "A4":  np.random.choice(["u", "y", "l"], n),
        "A5":  np.random.choice(["g", "p", "gg"], n),
        "A6":  np.random.choice(["c", "d", "cc"], n),
        "A7":  np.random.choice(["v", "h", "ff"], n),
        "A8":  np.random.uniform(0, 20, n),
        "A9":  np.random.choice(["t", "f"], n),
        "A10": np.random.choice(["t", "f"], n),
        "A11": np.random.randint(0, 20, n),
        "A12": np.random.choice(["t", "f"], n),
        "A13": np.random.choice(["g", "p", "s"], n),
        "A14": np.random.uniform(0, 2000, n),
        "A15": np.random.randint(0, 10000, n),
        "A16": np.random.choice(["+", "-"], n)
    })

    # Introduire quelques NaN comme dans le vrai dataset
    for col in ["A1", "A2", "A4", "A5", "A6", "A7", "A14"]:
        idx = np.random.choice(df.index, 3, replace=False)
        df.loc[idx, col] = np.nan

    return df


@pytest.fixture
def trained_pipeline(sample_data, tmp_path):
    """Entraîne le pipeline complet sur les données synthétiques"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = sample_data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({"+": 1, "-": 0})

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    full_pipeline.fit(X_train, y_train)

    # Sauvegarder dans un dossier temporaire
    model_path = tmp_path / "model.joblib"
    joblib.dump(full_pipeline, model_path)

    return full_pipeline, X_test, y_test, model_path


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────
def test_model_file_is_created(trained_pipeline):
    """Vérifie que model.joblib est bien généré"""
    _, _, _, model_path = trained_pipeline
    assert model_path.exists(), "Le fichier model.joblib n'a pas été créé"


def test_model_can_be_loaded(trained_pipeline):
    """Vérifie que le modèle sauvegardé peut être rechargé"""
    _, _, _, model_path = trained_pipeline
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None


def test_pipeline_predicts(trained_pipeline, sample_data):
    """Vérifie que le pipeline fait des prédictions sans erreur"""
    pipeline, X_test, y_test, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_predictions_are_binary(trained_pipeline):
    """Vérifie que les prédictions sont bien 0 ou 1"""
    pipeline, X_test, _, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    assert set(y_pred).issubset({0, 1}), "Les prédictions contiennent des valeurs inattendues"


def test_predict_proba_shape(trained_pipeline):
    """Vérifie que predict_proba retourne bien 2 colonnes (classe 0 et 1)"""
    pipeline, X_test, _, _ = trained_pipeline
    proba = pipeline.predict_proba(X_test)
    assert proba.shape[1] == 2


def test_accuracy_above_threshold(trained_pipeline):
    """Vérifie que l'accuracy est au dessus d'un seuil minimum"""
    from sklearn.metrics import accuracy_score
    pipeline, X_test, y_test, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.5, f"Accuracy trop faible : {acc}"


def test_pipeline_handles_missing_values(trained_pipeline, sample_data):
    """Vérifie que le pipeline gère les valeurs manquantes"""
    pipeline, _, _, _ = trained_pipeline
    X_with_nan = sample_data.iloc[:5, :-1].copy()
    X_with_nan.iloc[0, 0] = np.nan  # introduire un NaN
    y_pred = pipeline.predict(X_with_nan)
    assert len(y_pred) == 5