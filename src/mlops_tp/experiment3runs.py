import pandas as pd
import json
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from config import DATA_PATH, ARTIFACTS_DIR
import mlflow
import mlflow.sklearn

# ─────────────────────────────────────────
# Chargement des données (commun à tous les runs)
# ─────────────────────────────────────────

df = pd.read_csv(DATA_PATH, header=None, na_values="?")
df.columns = [f"A{i}" for i in range(1, 17)]

X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({"+": 1, "-": 0})

num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

artifacts_dir = Path(ARTIFACTS_DIR)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# Définition des 3 configurations à tester
# ─────────────────────────────────────────

configs = [
    {
        "run_name":        "run1_rf_200_median_70",
        "model":           RandomForestClassifier(n_estimators=200, random_state=42),
        "model_type":      "RandomForest",
        "n_estimators":    200,
        "imputer_num":     "median",
        "test_size":       0.15,
    },
    {
        "run_name":        "run2_rf_50_mean_70",
        "model":           RandomForestClassifier(n_estimators=50, random_state=42),
        "model_type":      "RandomForest",
        "n_estimators":    50,
        "imputer_num":     "mean",       # ← imputation par moyenne au lieu de médiane
        "test_size":       0.15,
    },
    {
        "run_name":        "run3_rf_200_median_80",
        "model":           RandomForestClassifier(n_estimators=200, random_state=42),
        "model_type":      "RandomForest",
        "n_estimators":    200,
        "imputer_num":     "median",
        "test_size":       0.10,         # ← split plus petit : 80% train / 10% test / 10% val
    },
]

# ─────────────────────────────────────────
# Fonction utilitaire : un run complet
# ─────────────────────────────────────────

def run_experiment(cfg):

    # -- Split selon la config --
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg["test_size"] * 2, random_state=42, stratify=y
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # -- Pipelines de preprocessing --
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["imputer_num"])),
        ("scaler",  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # -- Pipeline complet --
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", cfg["model"])
    ])

    # ── MLflow run ──────────────────────────────────────────
    mlflow.set_experiment("credit_approval")

    with mlflow.start_run(run_name=cfg["run_name"]):

        # Paramètres
        mlflow.log_params({
            "model_type":    cfg["model_type"],
            "n_estimators":  cfg["n_estimators"],
            "imputer_num":   cfg["imputer_num"],
            "test_size":     cfg["test_size"],
            "random_state":  42,
        })

        # Entraînement
        full_pipeline.fit(X_train, y_train)

        # Prédictions
        y_pred  = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc",      auc)

        print(f"\n{cfg['run_name']} → accuracy={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")

        # -- Artefact : matrice de confusion --
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Refus (-)", "Accord (+)"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Matrice de confusion — {cfg['run_name']}")
        plt.tight_layout()
        cm_path = artifacts_dir / f"confusion_matrix_{cfg['run_name']}.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(cm_path))

        # -- Artefact : courbe ROC --
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aléatoire")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title(f"Courbe ROC — {cfg['run_name']}")
        ax.legend()
        plt.tight_layout()
        roc_path = artifacts_dir / f"roc_curve_{cfg['run_name']}.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(roc_path))

        # -- Modèle --
        mlflow.sklearn.log_model(full_pipeline, "model")
        joblib.dump(full_pipeline, artifacts_dir / f"model_{cfg['run_name']}.joblib")

# ─────────────────────────────────────────
# Lancement des 3 runs
# ─────────────────────────────────────────

for cfg in configs:
    run_experiment(cfg)

print("\nTous les runs sont terminés. Lance 'mlflow ui' pour comparer.")