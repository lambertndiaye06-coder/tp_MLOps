import streamlit as st
import pandas as pd
import joblib
from config import MODEL_PATH

# Charger le modèle
pipeline = joblib.load(MODEL_PATH)

st.title("Prédiction d'octroi de crédit")
st.write("Remplissez les informations pour obtenir une prédiction.")

# Formulaire de saisie
col1, col2 = st.columns(2)

with col1:
    A1 = st.selectbox("A1", ["a", "b"])
    A2 = st.number_input("A2 (âge)", min_value=0.0)
    A3 = st.number_input("A3", min_value=0.0)
    A4 = st.selectbox("A4", ["u", "y", "l"])
    A5 = st.selectbox("A5", ["g", "p", "gg"])
    A6 = st.selectbox("A6", ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"])
    A7 = st.selectbox("A7", ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"])
    A8 = st.number_input("A8", min_value=0.0)

with col2:
    A9  = st.selectbox("A9",  ["t", "f"])
    A10 = st.selectbox("A10", ["t", "f"])
    A11 = st.number_input("A11", min_value=0, step=1)
    A12 = st.selectbox("A12", ["t", "f"])
    A13 = st.selectbox("A13", ["g", "p", "s"])
    A14 = st.number_input("A14", min_value=0.0)
    A15 = st.number_input("A15", min_value=0, step=1)

# Prédiction
if st.button("Prédire"):
    input_data = pd.DataFrame([{
        "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
        "A6": A6, "A7": A7, "A8": A8, "A9": A9, "A10": A10,
        "A11": A11, "A12": A12, "A13": A13, "A14": A14, "A15": A15
    }])

    prediction = pipeline.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Crédit accordé")
    else:
        st.error("❌ Crédit refusé")