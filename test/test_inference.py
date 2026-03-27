import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# Fixtures
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

    return df


@pytest.fixture
def trained_pipeline(sample_data):
    """Entraîne le pipeline complet sur les données synthétiques"""
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

    return full_pipeline, X_test, y_test


# ─────────────────────────────────────────
# Tests predict
# ─────────────────────────────────────────
def test_predict_returns_known_classes(trained_pipeline):
    """Vérifie que predict renvoie uniquement des classes connues (0 ou 1)"""
    pipeline, X_test, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    assert set(y_pred).issubset({0, 1}), f"Classes inconnues détectées : {set(y_pred)}"


def test_predict_single_sample(trained_pipeline, sample_data):
    """Vérifie que predict fonctionne sur un seul échantillon"""
    pipeline, _, _ = trained_pipeline
    single_sample = sample_data.iloc[[0], :-1]
    y_pred = pipeline.predict(single_sample)

    assert len(y_pred) == 1, "predict doit retourner 1 valeur pour 1 échantillon"
    assert y_pred[0] in {0, 1}, f"Classe inconnue : {y_pred[0]}"


def test_predict_output_is_integer(trained_pipeline):
    """Vérifie que les prédictions sont bien des entiers"""
    pipeline, X_test, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    assert all(isinstance(p, (int, np.integer)) for p in y_pred), \
        "Les prédictions doivent être des entiers"


def test_predict_length_matches_input(trained_pipeline):
    """Vérifie que le nombre de prédictions correspond au nombre d'entrées"""
    pipeline, X_test, _ = trained_pipeline
    y_pred = pipeline.predict(X_test)
    assert len(y_pred) == len(X_test), \
        f"Attendu {len(X_test)} prédictions, obtenu {len(y_pred)}"


# ─────────────────────────────────────────
# Tests predict_proba
# ─────────────────────────────────────────
def test_predict_proba_between_0_and_1(trained_pipeline):
    """Vérifie que toutes les probabilités sont entre 0 et 1"""
    pipeline, X_test, _ = trained_pipeline
    proba = pipeline.predict_proba(X_test)
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        "Les probabilités doivent être comprises entre 0 et 1"


def test_predict_proba_sums_to_1(trained_pipeline):
    """Vérifie que les probabilités par ligne somment à 1"""
    pipeline, X_test, _ = trained_pipeline
    proba = pipeline.predict_proba(X_test)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0), \
        "Les probabilités de chaque ligne doivent sommer à 1"


def test_predict_proba_shape(trained_pipeline):
    """Vérifie que predict_proba retourne 2 colonnes (classe 0 et classe 1)"""
    pipeline, X_test, _ = trained_pipeline
    proba = pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2), \
        f"Shape attendu ({len(X_test)}, 2), obtenu {proba.shape}"


def test_predict_proba_single_sample(trained_pipeline, sample_data):
    """Vérifie que predict_proba fonctionne sur un seul échantillon"""
    pipeline, _, _ = trained_pipeline
    single_sample = sample_data.iloc[[0], :-1]
    proba = pipeline.predict_proba(single_sample)

    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)