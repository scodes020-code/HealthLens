# =========================================================
# Symptom-Only Health Checker — run with: streamlit run app.py
# =========================================================
# Needs this file in the same folder (built by build_model.py):
#   - disease_knowledge_base.csv

import streamlit as st
import pandas as pd
import ast
import urllib.parse

st.set_page_config(page_title="Health Symptom Checker", page_icon="🩺")

# -----------------------------------------------------------
# Load data once
# -----------------------------------------------------------
@st.cache_data
def load_knowledge_base():
    df = pd.read_csv('disease_knowledge_base.csv')
    df['symptoms'] = df['symptoms'].apply(ast.literal_eval)
    return df

kb_df = load_knowledge_base()
all_symptoms = sorted(set().union(*kb_df['symptoms']))

# -----------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------
def predict_health_issue(user_symptoms, top_n=5):
    user_symptoms_set = set(s.strip().lower() for s in user_symptoms)
    results = []
    for _, row in kb_df.iterrows():
        disease_symptoms = set(row['symptoms'])
        if not disease_symptoms:
            continue
        overlap = user_symptoms_set & disease_symptoms
        if len(overlap) == 0:
            continue
        score = len(overlap) / len(disease_symptoms | user_symptoms_set)
        results.append({
            'disease': row['disease'].title(),
            'score': round(score * 100, 1),
            'matched': list(overlap),
            'description': row['description'],
            'doctor': row['doctor']
        })
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("🩺 Health Symptom Checker")
st.warning(
    "⚠️ This tool provides informational suggestions only and is **not a medical diagnosis**. "
    "Always consult a licensed physician for any health concerns."
)

name = st.text_input("Your name")
symptoms = st.multiselect("Select your symptoms", options=all_symptoms)

if st.button("Check my health"):
    if not symptoms:
        st.error("Please select at least one symptom.")
    else:
        results = predict_health_issue(symptoms)

        st.subheader(f"Results for {name if name else 'you'}")

        if not results:
            st.info("No close matches found for the symptoms you selected.")
        else:
            for r in results:
                with st.expander(f"{r['disease']} — {r['score']}% symptom match"):
                    st.write(f"**Matched symptoms:** {', '.join(r['matched'])}")
                    if r['description']:
                        st.write(r['description'])
                    if r['doctor']:
                        st.write(f"**Suggested doctor type:** {r['doctor']}")

                    query = urllib.parse.quote(f"{r['disease']} symptoms causes treatment")
                    st.link_button("🔎 Search more about this", f"https://www.google.com/search?q={query}")

        st.markdown("---")
        st.caption(
            "Reminder: this is not a diagnosis. If symptoms are severe or you're concerned, "
            "please see a doctor or seek emergency care."
        )
