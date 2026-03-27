import pandas as pd
#Lire le fichier CSV sans en-tête, les "?" sont considérés comme des valeurs manquantes
from ydata_profiling import ProfileReport

# Charger tes données
df = pd.read_csv("~/tp_MLOps/data/credit_approval/crx.data", header=None, na_values="?")

# Générer le rapport
profile = ProfileReport(df, title="Mon Rapport")

# Afficher dans Jupyter Notebook
#profile.to_notebook_iframe()

# OU sauvegarder en HTML
profile.to_file("rapport.html")

# Rapport minimal (plus rapide sur grands datasets)
profile = ProfileReport(df, minimal=True)

# Rapport complet avec détection des corrélations
profile = ProfileReport(
    df,
    title="Analyse",
    explorative=True,        # mode exploratoire
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True}
    }
)

# Pour grands datasets (>100k lignes)
profile = ProfileReport(df, minimal=True, samples={"head": 10, "tail": 10})
# Ouvre le rapport dans le navigateur
profile.to_file("rapport.html")
import webbrowser
webbrowser.open("rapport.html")