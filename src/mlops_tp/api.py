from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
from pathlib import Path
import json
from pydantic import BaseModel, field_validator
from config import MODEL_PATH

# ─────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────

# Description affichée dans /docs
app = FastAPI(
    title="API de prédiction de crédit",
    description="API ML pour prédire l'approbation d'un crédit bancaire",
    version="1.0.0"
)

# Charger le pipeline sauvegardé
#pipeline = joblib.load("/home/ndiaylam/tp_MLOps/artifacts/model.joblib")


pipeline = joblib.load(MODEL_PATH)


# Charger le run_info pour les métadonnées
from config import RUN_INFO_PATH
with open(RUN_INFO_PATH, "r") as f:
    run_info = json.load(f)


# ─────────────────────────────────────────
# Schéma des données
# ─────────────────────────────────────────
class CreditData(BaseModel):
    A1:  Optional[str]   = None
    A2:  Optional[float] = None
    A3:  float
    A4:  Optional[str]   = None
    A5:  Optional[str]   = None
    A6:  Optional[str]   = None
    A7:  Optional[str]   = None
    A8:  float
    A9:  str
    A10: str
    A11: int
    A12: str
    A13: str
    A14: Optional[float] = None
    A15: int
    @field_validator("A3", "A8", "A15")
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError("La valeur doit être positive")
        return v

    @field_validator("A1")
    def a1_valid_values(cls, v):
        if v not in {"a", "b", None}:
            raise ValueError("A1 doit être 'a' ou 'b'")
        return v

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/health", summary="Vérifier l'état de l'API")
def health():
    """
    Vérifie que l'API est vivante et que le modèle est bien chargé.
    - **status** : ok si tout fonctionne
    - **model_loaded** : confirme que le pipeline est chargé
    """
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }


@app.get("/metadata", summary="Métadonnées du modèle")
def metadata():
    """
    Retourne les informations sur le modèle et les données attendues.
    - **version** : version de l'API
    - **type_tache** : classification binaire
    - **modele** : algorithme utilisé
    - **variables** : liste des features attendues
    - **dataset** : informations sur le dataset d'entraînement
    """
    return {
        "version": "1.0.0",
        "type_tache": "classification binaire",
        "modele": "RandomForestClassifier",
        "classes": {"0": "refusé", "1": "accordé"},
        "variables_attendues": list(CreditData.model_fields.keys()),
        "dataset": run_info
    }


@app.post("/predict", summary="Prédire l'approbation d'un crédit")
def predict(data: CreditData):
    """
    Reçoit les données d'un client et retourne la prédiction.
    - **prediction** : 0 (refusé) ou 1 (accordé)
    - **label** : version lisible de la prédiction
    - **probabilité** : confiance du modèle entre 0 et 1
    """
    df = pd.DataFrame([data.dict()])

    prediction = pipeline.predict(df)[0]
    proba      = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "accordé" if prediction == 1 else "refusé",
        "probabilité": round(float(proba), 3)
    }