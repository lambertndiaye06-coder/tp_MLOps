# TP MLOps — Prédiction d'octroi de crédit

Projet MLOps complet de classification binaire pour prédire l'octroi d'un crédit bancaire à partir du dataset UCI Credit Approval (`crx.data`).

---

## Structure du projet

```
tp_MLOps/
├── Dockerfile
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── credit_approval/
│       └── crx.data
├── src/
│   └── mlops_tp/
│       ├── __init__.py
│       ├── config.py          # Paramètres globaux (chemins, hyperparamètres)
│       ├── train.py           # Entraînement du modèle
│       ├── inference.py       # Prédiction sur nouvelles données
│       ├── api.py             # API REST (FastAPI)
│       ├── app.py             # Interface web (Streamlit)
│       ├── schemas.py         # Schémas Pydantic
│       ├── crx.data           # Données (copie locale pour Docker)
│       └── artifacts/         # Générés après entraînement
│           ├── model.joblib
│           ├── metrics.json
│           ├── loss_curve.png
│           └── run_info.json
└── test/
    ├── test_api.py
    └── test_inference.py
```

---

## Dataset

- **Source** : [UCI Machine Learning Repository — Credit Approval](https://archive.ics.uci.edu/ml/datasets/credit+approval)
- **Tâche** : Classification binaire (`+` = crédit accordé, `-` = crédit refusé)
- **Taille** : 690 lignes, 16 colonnes (A1 à A16)
- **Features** : mélange de variables numériques et catégorielles, avec valeurs manquantes
- **Classes** : 307 acceptés (44,5%), 383 refusés (55,5%)

---

## Installation

### En local

```bash
# Cloner le projet
git clone <url-du-repo>
cd tp_MLOps

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Avec Docker

```bash
docker build -t mlops_tp .
```

---

## Utilisation

### 1. Entraînement

```bash
# En local
cd src/mlops_tp
python train.py

# Avec Docker
docker run \
  -v $(pwd)/src/mlops_tp/crx.data:/app/src/mlops_tp/crx.data \
  -v $(pwd)/src/mlops_tp/artifacts:/app/src/mlops_tp/artifacts \
  mlops_tp python train.py
```

Le script génère dans `artifacts/` :
- `model.joblib` — pipeline complet (preprocessing + RandomForest)
- `metrics.json` — log loss train/test pour n_estimators de 1 à 200
- `loss_curve.png` — courbe de loss
- `run_info.json` — métadonnées du run (split, random_state, dataset…)

### 2. API REST (FastAPI)

```bash
# En local
cd src/mlops_tp
uvicorn api:app --reload

# Avec Docker
docker run -p 8000:8000 \
  -v $(pwd)/src/mlops_tp/artifacts:/app/src/mlops_tp/artifacts \
  mlops_tp
```

L'API est accessible sur `http://localhost:8000`.

#### Endpoints

| Méthode | Route       | Description                        |
|---------|-------------|------------------------------------|
| GET     | `/health`   | Vérifie que l'API est opérationnelle |
| GET     | `/metadata` | Infos sur le modèle et les features |
| POST    | `/predict`  | Prédiction pour un client           |

#### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "A1": "b", "A2": 30.83, "A3": 0.0,
    "A4": "u", "A5": "g", "A6": "w", "A7": "v",
    "A8": 1.25, "A9": "t", "A10": "t",
    "A11": 1, "A12": "f", "A13": "g",
    "A14": 202.0, "A15": 0
  }'
```

#### Exemple de réponse

```json
{
  "prediction": 1,
  "label": "accordé",
  "probabilité": 0.873
}
```

La documentation interactive Swagger est disponible sur `http://localhost:8000/docs`.

### 3. Interface Streamlit

```bash
cd src/mlops_tp
streamlit run app.py
```

L'interface est accessible sur `http://localhost:8501`. Elle permet de remplir un formulaire et d'obtenir une prédiction en temps réel.

---

## Modèle

Le pipeline sklearn inclut :

- **Prétraitement numérique** : `SimpleImputer(median)` + `StandardScaler`
- **Prétraitement catégoriel** : `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first')`
- **Modèle** : `RandomForestClassifier(n_estimators=200, random_state=42)`

**Performances sur le jeu de test (15%) :**

| Métrique  | Classe 0 (refusé) | Classe 1 (accordé) | Moyenne |
|-----------|-------------------|--------------------|---------|
| Precision | ~0.85             | ~0.85              | ~0.85   |
| Recall    | ~0.85             | ~0.85              | ~0.85   |
| F1-score  | ~0.85             | ~0.85              | ~0.85   |

---

## Tests

```bash
cd test
pytest .
```

---

## Variables du dataset

| Variable | Type        | Description                  |
|----------|-------------|------------------------------|
| A1       | Catégorielle| a, b                         |
| A2       | Numérique   | Âge                          |
| A3       | Numérique   | Dette                        |
| A4       | Catégorielle| u, y, l                      |
| A5       | Catégorielle| g, p, gg                     |
| A6       | Catégorielle| c, d, cc, i, j, k, m, r, q…  |
| A7       | Catégorielle| v, h, bb, j, n, z, dd…       |
| A8       | Numérique   | Ancienneté emploi            |
| A9       | Catégorielle| t, f                         |
| A10      | Catégorielle| t, f                         |
| A11      | Numérique   | Nombre de comptes            |
| A12      | Catégorielle| t, f                         |
| A13      | Catégorielle| g, p, s                      |
| A14      | Numérique   | Code postal                  |
| A15      | Numérique   | Revenu                       |
| A16      | Cible       | + (accordé) / - (refusé)     |

> Les noms des colonnes sont anonymisés dans le dataset original pour des raisons de confidentialité.
