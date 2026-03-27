from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
import joblib
import json
from config import MODEL_PATH, RUN_INFO_PATH

# ─────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────

app = FastAPI(
    title="API de prédiction de crédit",
    description="API ML pour prédire l'approbation d'un crédit bancaire",
    version="1.0.0"
)

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None
    print(f"⚠️ Modèle non chargé : {e}")

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

    @field_validator("A1", mode="before")
    def a1_valid_values(cls, v):
        if v is not None and v not in {"a", "b"}:
            raise ValueError("A1 doit être 'a' ou 'b'")
        return v


# ─────────────────────────────────────────
# Fonction de prédiction commune
# ─────────────────────────────────────────

def run_predict(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15):
    """Logique métier partagée entre la route GET et POST."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    df = pd.DataFrame([{
        "A1": A1, "A2": A2, "A3": A3, "A4": A4,
        "A5": A5, "A6": A6, "A7": A7, "A8": A8,
        "A9": A9, "A10": A10, "A11": A11,
        "A12": A12, "A13": A13, "A14": A14, "A15": A15
    }])

    prediction = pipeline.predict(df)[0]
    proba      = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "accordé" if prediction == 1 else "refusé",
        "probabilité": round(float(proba), 3)
    }


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if pipeline is not None else "degraded",
        "model_loaded": pipeline is not None
    }


@app.get("/metadata")
def metadata():
    return {
        "version": "1.0.0",
        "type_tache": "classification binaire",
        "modele": "RandomForestClassifier",
        "classes": {"0": "refusé", "1": "accordé"},
        "variables_attendues": list(CreditData.model_fields.keys()),
        "dataset": run_info
    }


# ── POST /predict — usage programmatique (Streamlit, scripts, prod) ──────────
@app.post("/predict", summary="Prédire via JSON body")
def predict_post(data: CreditData):
    """Envoyer les données en JSON body — usage recommandé en production."""
    return run_predict(
        data.A1, data.A2, data.A3, data.A4, data.A5,
        data.A6, data.A7, data.A8, data.A9, data.A10,
        data.A11, data.A12, data.A13, data.A14, data.A15
    )


# ── GET /predict — test rapide via navigateur ou URL ─────────────────────────
@app.get("/predict", summary="Prédire via paramètres URL")
def predict_get(
    A1:  Optional[str]   = None,
    A2:  Optional[float] = None,
    A3:  float           = 0.0,
    A4:  Optional[str]   = None,
    A5:  Optional[str]   = None,
    A6:  Optional[str]   = None,
    A7:  Optional[str]   = None,
    A8:  float           = 0.0,
    A9:  str             = "t",
    A10: str             = "t",
    A11: int             = 0,
    A12: str             = "f",
    A13: str             = "g",
    A14: Optional[float] = None,
    A15: int             = 0
):
    """Passer les données en paramètres URL — pratique pour tester dans le navigateur."""
    return run_predict(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)
