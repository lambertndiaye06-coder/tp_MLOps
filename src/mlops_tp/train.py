import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from sklearn.impute import SimpleImputer
from config import DATA_PATH, ARTIFACTS_DIR


# ─────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────

# Lire le fichier CSV sans en-tête, les "?" sont considérés comme des valeurs manquantes
#df = pd.read_csv("~/tp_MLOps/data/credit_approval/crx.data", header=None, na_values="?")
df = pd.read_csv(DATA_PATH, header=None, na_values="?")

# Nommer les colonnes A1 à A16
df.columns = [f"A{i}" for i in range(1, 17)]

# Séparer les features (X) et la cible (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({"+": 1, "-": 0})  # Encoder "+" en 1 et "-" en 0

# Identifier les colonnes numériques et catégorielles pour le pipeline
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

# ─────────────────────────────────────────
# 2. Traçabilité : Run info
# ─────────────────────────────────────────

# Créer le dossier artifacts s'il n'existe pas
#artifacts_dir = Path("/home/ndiaylam/tp_MLOps/artifacts")
artifacts_dir = Path(ARTIFACTS_DIR)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Consigner les informations du run pour la traçabilité
run_info = {
    "dataset": "credit_approval/crx.data",
    "shape": {"lignes": df.shape[0], "colonnes": df.shape[1]},
    "cible": "A16",
    "split": {"train": 0.70, "test": 0.15, "validation": 0.15},
    "random_state": 42
}

with open(artifacts_dir / "run_info.json", "w") as f:
    json.dump(run_info, f, indent=4)

# ─────────────────────────────────────────
# 3. Split train / test / validation
# ─────────────────────────────────────────

# Étape 1 : 70% train, 30% temporaire
# stratify=y conserve la proportion des classes dans chaque split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Étape 2 : diviser les 30% en 15% test et 15% validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ─────────────────────────────────────────
# 4. Construction du Pipeline
# ─────────────────────────────────────────

# Pipeline pour les colonnes numériques :
# - SimpleImputer remplace les NaN par la médiane
# - StandardScaler normalise les valeurs (moyenne=0, écart-type=1)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline pour les colonnes catégorielles :
# - SimpleImputer remplace les NaN par la valeur la plus fréquente
# - OneHotEncoder encode les catégories en colonnes binaires
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

# ColumnTransformer applique chaque pipeline aux bonnes colonnes
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ─────────────────────────────────────────
# 5. Courbes de loss (log loss en fonction du n_estimators)
# ─────────────────────────────────────────

train_losses = []
test_losses  = []
n_estimators_range = range(1, 201, 10)  # de 1 à 200 arbres, par pas de 10

for n in n_estimators_range:
    # Créer un pipeline complet avec n arbres
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=n, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # predict_proba retourne les probabilités pour chaque classe
    # nécessaire pour calculer le log loss
    y_train_proba = pipeline.predict_proba(X_train)
    y_test_proba  = pipeline.predict_proba(X_test)

    train_losses.append(log_loss(y_train, y_train_proba))
    test_losses.append(log_loss(y_test, y_test_proba))

# Sauvegarder les métriques dans un fichier JSON pour la traçabilité
metrics = {
    "n_estimators": list(n_estimators_range),
    "train_loss": train_losses,
    "test_loss": test_losses
}
with open(artifacts_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Tracer et sauvegarder la courbe de loss
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, train_losses, label="Train Loss (log loss)", marker="o")
plt.plot(n_estimators_range, test_losses,  label="Test Loss (log loss)",  marker="o")
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("Log Loss")
plt.title("Courbe de Loss - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(artifacts_dir / "loss_curve.png", dpi=150)  # sauvegarde avant show()
plt.show()

# ─────────────────────────────────────────
# 6. Entraînement final
# ─────────────────────────────────────────

# On entraîne le pipeline final avec 200 arbres (meilleur compromis vu sur la courbe)
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

full_pipeline.fit(X_train, y_train)

# Évaluation sur le jeu de test
y_pred = full_pipeline.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────
# 7. Sauvegarde du pipeline complet
# ─────────────────────────────────────────

# joblib sérialise le pipeline entier (preprocessing + modèle)
# ce fichier suffit pour faire de l'inférence en production
joblib.dump(full_pipeline, artifacts_dir / "model.joblib")
print("Modèle sauvegardé sous 'model.joblib'")