import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from config import DATA_PATH, ARTIFACTS_DIR
import mlflow
import mlflow.sklearn

# ─────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────

df = pd.read_csv(DATA_PATH, header=None, na_values="?")
df.columns = [f"A{i}" for i in range(1, 17)]

X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({"+": 1, "-": 0})

num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

# ─────────────────────────────────────────
# 2. Traçabilité : dossier artifacts
# ─────────────────────────────────────────

artifacts_dir = Path(ARTIFACTS_DIR)
artifacts_dir.mkdir(parents=True, exist_ok=True)

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

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ─────────────────────────────────────────
# 4. Construction du Pipeline
# ─────────────────────────────────────────

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

# ─────────────────────────────────────────
# 5. Lancement du run MLflow
# ─────────────────────────────────────────

mlflow.set_experiment("credit_approval")

with mlflow.start_run(run_name="random_forest_200"):

    # --- Paramètres loggés dès le début ---
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators_final": 200,
        "random_state": 42,
        "test_size": 0.15,
        "val_size": 0.15,
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
    }
    mlflow.log_params(params)

    # ─────────────────────────────────────
    # Courbes de loss (n_estimators de 1 à 200)
    # ─────────────────────────────────────

    train_losses = []
    test_losses  = []
    n_estimators_range = range(1, 201, 10)

    for n in n_estimators_range:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=n, random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        y_train_proba = pipeline.predict_proba(X_train)
        y_test_proba  = pipeline.predict_proba(X_test)

        train_losses.append(log_loss(y_train, y_train_proba))
        test_losses.append(log_loss(y_test,  y_test_proba))

        # Log du log_loss à chaque step pour voir la courbe dans l'UI MLflow
        mlflow.log_metric("train_log_loss", log_loss(y_train, y_train_proba), step=n)
        mlflow.log_metric("test_log_loss",  log_loss(y_test,  y_test_proba),  step=n)

    # Sauvegarde JSON (inchangée)
    metrics_json = {
        "n_estimators": list(n_estimators_range),
        "train_loss": train_losses,
        "test_loss": test_losses
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)

    # Courbe PNG → loggée comme artifact MLflow
    plt.figure(figsize=(10, 5))
    plt.plot(n_estimators_range, train_losses, label="Train Loss (log loss)", marker="o")
    plt.plot(n_estimators_range, test_losses,  label="Test Loss (log loss)",  marker="o")
    plt.xlabel("Nombre d'arbres (n_estimators)")
    plt.ylabel("Log Loss")
    plt.title("Courbe de Loss - Random Forest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "loss_curve.png", dpi=150)
    plt.show()

    mlflow.log_artifact(str(artifacts_dir / "loss_curve.png"))   # ← visible dans l'UI
    mlflow.log_artifact(str(artifacts_dir / "run_info.json"))    # ← traçabilité dataset

    # ─────────────────────────────────────
    # Entraînement final (200 arbres)
    # ─────────────────────────────────────

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix

    # ─────────────────────────────────────────
    # Artefacts : matrice de confusion + courbe ROC
    # ─────────────────────────────────────────

    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    # -- Matrice de confusion --
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Refus (-)", "Accord (+)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    mlflow.log_artifact(str(artifacts_dir / "confusion_matrix.png"))

    # -- Courbe ROC --
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("auc", auc)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "roc_curve.png", dpi=150)
    plt.close()
    mlflow.log_artifact(str(artifacts_dir / "roc_curve.png"))

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    # Métriques finales
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print("Accuracy :", acc)
    print(classification_report(y_test, y_pred))

    # ─────────────────────────────────────
    # Sauvegarde du modèle (joblib + MLflow)
    # ─────────────────────────────────────

    joblib.dump(full_pipeline, artifacts_dir / "model2.joblib")
    print("Modèle sauvegardé sous 'model2.joblib'")

    # Log du pipeline sklearn dans MLflow (permet de le recharger via mlflow.sklearn.load_model)
    mlflow.sklearn.log_model(full_pipeline, "model2")

print("\nRun terminé. Lance 'mlflow ui' pour visualiser les résultats.")