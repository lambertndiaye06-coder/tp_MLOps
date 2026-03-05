"""
inference.py — Chargement du modèle entraîné et prédiction sur de nouvelles données
"""

import json
import joblib
import pandas as pd

from config import (
    FEATURE_SCHEMA_PATH,
    MODEL_PATH,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
)


# ── Chargement du modèle ──────────────────────────────────────────────────────

def load_model():
    """Charge le modèle sérialisé depuis artifacts/."""
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis {MODEL_PATH}")
    return model


def load_feature_schema() -> list:
    """Retourne la liste ordonnée des features attendues par le modèle."""
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)
    return schema["features"]


# ── Prétraitement (même pipeline que train.py) ────────────────────────────────

def preprocess_input(raw: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Applique le même prétraitement qu'à l'entraînement sur un DataFrame brut,
    puis aligne les colonnes sur le schéma attendu.
    """
    X = raw.copy()

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    X = pd.get_dummies(X, drop_first=True)

    # Aligner les colonnes : ajouter les manquantes (valeur 0), supprimer les inconnues
    X = X.reindex(columns=feature_names, fill_value=0)

    return X


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict(raw: pd.DataFrame) -> pd.Series:
    """
    Prend un DataFrame brut (sans colonne cible),
    retourne les prédictions sous forme de labels ('+' ou '-').
    """
    model = load_model()
    feature_names = load_feature_schema()

    X = preprocess_input(raw, feature_names)
    preds_binary = model.predict(X)

    return pd.Series(preds_binary).map({1: POSITIVE_CLASS, 0: NEGATIVE_CLASS})


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    """Exemple : réutilise les premières lignes du dataset pour tester l'inférence."""
    from config import COLUMN_NAMES, DATA_PATH, TARGET_COLUMN

    df = pd.read_csv(DATA_PATH, header=None, na_values="?")
    df.columns = COLUMN_NAMES

    # On retire la colonne cible pour simuler des données "nouvelles"
    X_raw = df.drop(columns=[TARGET_COLUMN]).head(10)

    predictions = predict(X_raw)
    print("Prédictions :")
    print(predictions.to_string())


if __name__ == "__main__":
    main()
