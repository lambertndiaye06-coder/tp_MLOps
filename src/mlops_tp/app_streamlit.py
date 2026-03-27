import streamlit as st
import requests

st.set_page_config(page_title="Prédiction de crédit", page_icon="🏦")
st.title("🏦 Prédiction d'approbation de crédit")
st.markdown("Remplissez les informations du client pour obtenir une prédiction.")

# ── URL de l'API ──────────────────────────────────────────────────────────────
API_URL = "http://localhost:8001"  # changer par l'URL publique en prod

# ── Formulaire ────────────────────────────────────────────────────────────────
with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        A1  = st.selectbox("A1 — Sexe",          options=["a", "b", None])
        A2  = st.number_input("A2 — Âge",         min_value=0.0, value=30.0)
        A3  = st.number_input("A3 — Dette",        min_value=0.0, value=0.0)
        A4  = st.text_input("A4",  value="u")
        A5  = st.text_input("A5",  value="g")
        A6  = st.text_input("A6",  value="w")
        A7  = st.text_input("A7",  value="v")
        A8  = st.number_input("A8 — Ancienneté",   min_value=0.0, value=1.25)

    with col2:
        A9  = st.selectbox("A9",  ["t", "f"])
        A10 = st.selectbox("A10", ["t", "f"])
        A11 = st.number_input("A11", min_value=0, value=1, step=1)
        A12 = st.selectbox("A12", ["t", "f"])
        A13 = st.text_input("A13", value="g")
        A14 = st.number_input("A14", min_value=0.0, value=202.0)
        A15 = st.number_input("A15", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("🔍 Prédire")

# ── Appel à l'API ─────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "A1": A1, "A2": A2, "A3": A3, "A4": A4,
        "A5": A5, "A6": A6, "A7": A7, "A8": A8,
        "A9": A9, "A10": A10, "A11": int(A11),
        "A12": A12, "A13": A13, "A14": A14, "A15": int(A15)
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result   = response.json()

        if result["prediction"] == 1:
            st.success(f"✅ Crédit **accordé** — confiance : {result['probabilité']:.1%}")
        else:
            st.error(f"❌ Crédit **refusé** — confiance : {1 - result['probabilité']:.1%}")

    except Exception as e:
        st.warning("⚠️ Impossible de contacter l'API. Vérifiez qu'elle est bien lancée.")
        st.code(str(e))