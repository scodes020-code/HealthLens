
# =========================================================
# Health Symptom Checker — run with: streamlit run app.py
# =========================================================
# Needs these files in the same folder:
#   - disease_knowledge_base.csv   (built by build_model.py)
#   - bmi_risk_model.pkl           (built by build_model.py)
#
# Feedback addressed in this version:
#   1. Weighted/specificity-based scoring, not plain overlap
#   2. General self-care suggestions for low-risk matches (never dosages)
#   3. In-app feedback box -> emailed straight to your Gmail (+ feedback.csv backup)
#   4. Lay-language symptom synonyms + fuzzy matching
#   5. "Browse by illness" tab (illness-first, not just symptom-first)
#   6. Age / height / weight / lifestyle intake, BMI risk model wired in
#   7. Combines 1+6 so you get two distinct signals instead of one flat score

import streamlit as st
import pandas as pd
import ast
import csv
import os
import difflib
import urllib.parse
from datetime import datetime
import pickle
import smtplib
import ssl
from email.message import EmailMessage

st.set_page_config(page_title="Health Symptom Checker", page_icon="🩺", layout="wide")

# -----------------------------------------------------------
# Lay-language -> knowledge-base symptom synonyms (feedback #4)
# Extend this dict any time a user reports a term that doesn't match.
# -----------------------------------------------------------
SYNONYMS = {
    "throwing up": "vomiting", "puking": "vomiting", "sick to stomach": "nausea",
    "stomach ache": "stomach pain", "belly ache": "belly pain", "tummy pain": "stomach pain",
    "runny tummy": "diarrhoea", "loose stool": "diarrhoea", "the runs": "diarrhoea",
    "cant sleep": "restlessness", "can't sleep": "restlessness",
    "tired all the time": "fatigue", "no energy": "lethargy", "exhausted": "fatigue",
    "high temperature": "high fever", "feverish": "mild fever", "hot body": "high fever",
    "cold body": "chills", "shivering": "chills",
    "yellow eyes": "yellowing of eyes", "yellow skin": "yellowish skin",
    "sore throat": "throat irritation", "scratchy throat": "throat irritation",
    "blocked nose": "congestion", "stuffy nose": "congestion",
    "runny nose": "runny nose", "cant smell": "loss of smell", "can't smell": "loss of smell",
    "dizzy": "dizziness", "room spinning": "spinning movements", "feel faint": "dizziness",
    "chest tightness": "chest pain", "hard to breathe": "breathlessness",
    "short of breath": "breathlessness", "out of breath": "breathlessness",
    "itchy skin": "itching", "skin rash": "skin rash", "red bumps": "red spots over body",
    "peeing a lot": "polyuria", "peeing often": "polyuria", "burning pee": "burning micturition",
    "burning when i pee": "burning micturition", "dark pee": "dark urine",
    "joint ache": "joint pain", "achy joints": "joint pain", "achy muscles": "muscle pain",
    "weight dropping": "weight loss", "losing weight": "weight loss",
    "gaining weight": "weight gain", "puffy face": "puffy face and eyes",
    "swollen legs": "swollen legs", "cant focus": "lack of concentration",
    "can't focus": "lack of concentration", "mood swings": "mood swings",
    "anxious": "anxiety", "sad": "depression", "irritable": "irritability",
    "back ache": "back pain", "neck ache": "neck pain", "headache": "headache",
    "blurry vision": "blurred and distorted vision", "eye pain": "pain behind the eyes",
    "cough with mucus": "mucoid sputum", "coughing blood": "blood in sputum",
    "sneezing a lot": "continuous sneezing", "cracked lips": "drying and tingling lips",
    "slurred talk": "slurred speech", "weak arm": "weakness of one body side",
    "weak leg": "weakness in limbs", "nail changes": "brittle nails",
}


# -----------------------------------------------------------
# Load data once
# -----------------------------------------------------------
@st.cache_data
def load_knowledge_base():
    df = pd.read_csv('disease_knowledge_base.csv')
    df['symptoms'] = df['symptoms'].apply(ast.literal_eval)
    return df

@st.cache_resource
def load_bmi_model():
    if os.path.exists('bmi_risk_model.pkl'):
        with open('bmi_risk_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

kb_df = load_knowledge_base()
bmi_model = load_bmi_model()
all_symptoms = sorted(set().union(*kb_df['symptoms']))
all_diseases = sorted(kb_df['disease'].unique())

# Specificity weight per symptom (feedback #1): a symptom that appears in
# fewer diseases is more diagnostic, so it should count for more than a
# generic symptom like "fatigue" that shows up almost everywhere.
@st.cache_data
def symptom_weights(_kb_df):
    counts = {}
    for syms in _kb_df['symptoms']:
        for s in syms:
            counts[s] = counts.get(s, 0) + 1
    n_diseases = len(_kb_df)
    # +1 smoothing so a symptom in every disease doesn't hit weight 0
    return {s: max(0.15, 1 - (c / n_diseases)) for s, c in counts.items()}

WEIGHTS = symptom_weights(kb_df)


def normalize_symptom(raw):
    """Map a free-typed lay term to a known symptom via synonyms, then fuzzy match."""
    s = raw.strip().lower()
    if s in SYNONYMS:
        s = SYNONYMS[s]
    if s in all_symptoms:
        return s
    close = difflib.get_close_matches(s, all_symptoms, n=1, cutoff=0.75)
    return close[0] if close else None


# -----------------------------------------------------------
# Prediction logic — weighted match instead of plain overlap
# -----------------------------------------------------------
def predict_health_issue(user_symptoms, top_n=5):
    user_set = set(user_symptoms)
    results = []
    for _, row in kb_df.iterrows():
        disease_symptoms = set(row['symptoms'])
        if not disease_symptoms:
            continue
        overlap = user_set & disease_symptoms
        if not overlap:
            continue
        matched_weight = sum(WEIGHTS.get(s, 0.3) for s in overlap)
        union_weight = sum(WEIGHTS.get(s, 0.3) for s in (disease_symptoms | user_set))
        score = matched_weight / union_weight if union_weight else 0
        results.append({
            'disease': row['disease'].title(),
            'score': round(score * 100, 1),
            'matched': sorted(overlap),
            'unmatched_from_disease': sorted(disease_symptoms - user_set)[:6],
            'description': row['description'],
            'doctor': row['doctor'],
            'cures': row.get('cures', ''),
            'risk_level': str(row.get('risk_level', '')).lower(),
        })
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]


def save_feedback_to_csv(name, message):
    """Always keep a local backup, even if email sending fails or isn't configured."""
    file_exists = os.path.exists('feedback.csv')
    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'name', 'message'])
        writer.writerow([datetime.now().isoformat(timespec='seconds'), name, message])


def send_feedback_email(name, message):
    """
    Emails feedback straight to your Gmail using an App Password.
    Reads credentials from Streamlit secrets (.streamlit/secrets.toml or
    the "Secrets" panel in Streamlit Community Cloud) — never hardcode
    your password in this file.

    Required secrets:
        GMAIL_ADDRESS     = "scodes020@gmail.com"
        GMAIL_APP_PASSWORD = "ksjd geiq kzsb ssnw"   # 16-char App Password
        FEEDBACK_TO        = "scodes020@gmail.com" # where it lands (can be same as above)
    """
    try:
        gmail_address = st.secrets["GMAIL_ADDRESS"]
        gmail_app_password = st.secrets["GMAIL_APP_PASSWORD"]
        feedback_to = st.secrets.get("FEEDBACK_TO", gmail_address)
    except Exception:
        return False, "Email not configured yet (missing secrets)."

    msg = EmailMessage()
    msg["Subject"] = f"Health Symptom Checker feedback from {name or 'Anonymous'}"
    msg["From"] = gmail_address
    msg["To"] = feedback_to
    msg.set_content(
        f"From: {name or 'Anonymous'}\n"
        f"Time: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"{message}"
    )

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_address, gmail_app_password)
            server.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


def save_feedback(name, message):
    """Try email first; CSV backup always happens regardless of email outcome."""
    save_feedback_to_csv(name, message)
    return send_feedback_email(name, message)


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("🩺 Health Symptom Checker")
st.warning(
    "⚠️ This tool provides informational suggestions only and is **not a medical diagnosis**. "
    "Always consult a licensed physician for any health concerns."
)

tab_symptoms, tab_browse, tab_feedback = st.tabs(
    ["🔍 Check my symptoms", "📖 Browse by illness", "💬 Feedback"]
)

# ---------------- TAB 1: symptom-first checker ----------------
with tab_symptoms:
    name = st.text_input("Your name")

    with st.expander("Optional: add your details for a more personalized risk read (age, BMI, lifestyle)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])
        with col2:
            height_cm = st.number_input("Height (cm)", min_value=0.0, value=0.0)
            weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0)
        with col3:
            smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
            alcohol = st.selectbox("Alcohol", ["None", "Moderate", "Heavy"])
            activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])

        bmi_value = None
        if height_cm > 0 and weight_kg > 0:
            bmi_value = weight_kg / ((height_cm / 100) ** 2)
            st.metric("Your BMI", f"{bmi_value:.1f}")

    st.subheader("What are you feeling?")
    st.caption(
        "Type in your own words if a symptom isn't in the list below — "
        "e.g. 'throwing up' or 'stomach ache' — we'll match it for you."
    )

    free_text = st.text_input("Describe a symptom in your own words (press Enter to add)")
    if free_text:
        matched = normalize_symptom(free_text)
        if matched:
            st.success(f"Matched to: **{matched}**")
        else:
            st.info("Couldn't confidently match that term — please also pick from the list below.")

    picked = st.multiselect("Or select your symptoms directly", options=all_symptoms)

    symptoms = set(picked)
    if free_text:
        m = normalize_symptom(free_text)
        if m:
            symptoms.add(m)

    if st.button("Check my health", type="primary"):
        if not symptoms:
            st.error("Please add at least one symptom, either typed or selected.")
        else:
            results = predict_health_issue(list(symptoms))
            st.subheader(f"Results for {name if name else 'you'}")

            if not results:
                st.info("No close matches found for the symptoms you selected.")
            else:
                for r in results:
                    with st.expander(f"{r['disease']} — {r['score']}% likelihood match"):
                        st.write(f"**Matched symptoms:** {', '.join(r['matched'])}")
                        if r['unmatched_from_disease']:
                            st.caption(
                                "Other symptoms commonly seen with this: "
                                + ', '.join(r['unmatched_from_disease'])
                            )
                        if r['description']:
                            st.write(r['description'])
                        if r['doctor']:
                            st.write(f"**Suggested doctor type:** {r['doctor']}")

                        # Feedback #2: general self-care only, gated to low risk, never dosages
                        if r['cures'] and 'low' in r['risk_level']:
                            st.info(
                                f"**If symptoms are mild:** {r['cures']}. "
                                "These are general comfort measures, not a prescription. "
                                "**See a doctor if symptoms persist beyond a few days, worsen, "
                                "or you're unsure.**"
                            )
                        elif r['risk_level'] and 'low' not in r['risk_level']:
                            st.warning("This condition is generally associated with higher risk — please see a doctor rather than self-treating.")

                        query = urllib.parse.quote(f"{r['disease']} symptoms causes treatment")
                        st.link_button("🔎 Search more about this", f"https://www.google.com/search?q={query}")

                # Feedback #6/#7: separate lifestyle-risk signal, kept distinct from the symptom match
                if bmi_model is not None and 'bmi_value' in dir() and bmi_value:
                    risk_prob = bmi_model.predict_proba([[bmi_value]])[0][1]
                    st.markdown("---")
                    st.subheader("General health risk (based on BMI)")
                    st.write(
                        f"Based on your BMI of {bmi_value:.1f}, our model estimates a "
                        f"**{risk_prob*100:.0f}% elevated general disease risk** score. "
                        "This is a separate, general signal — not tied to the specific symptoms above. "
                        "Smoking, alcohol, and activity level aren't factored into this number yet."
                    )

            st.markdown("---")
            st.caption(
                "Reminder: this is not a diagnosis. If symptoms are severe or you're concerned, "
                "please see a doctor or seek emergency care."
            )

# ---------------- TAB 2: illness-first browser (feedback #5) ----------------
with tab_browse:
    st.subheader("Look up an illness directly")
    st.caption("Not sure how to describe your symptoms? Search for an illness you suspect and see what it typically involves.")

    chosen_disease = st.selectbox(
        "Search or select an illness",
        options=[""] + [d.title() for d in all_diseases],
    )
    if chosen_disease:
        row = kb_df[kb_df['disease'] == chosen_disease.lower()].iloc[0]
        st.write(f"### {chosen_disease}")
        if row['description']:
            st.write(row['description'])
        st.write("**Common symptoms:**")
        st.write(", ".join(sorted(row['symptoms'])) if row['symptoms'] else "Not listed in our data.")
        if row.get('doctor'):
            st.write(f"**Suggested doctor type:** {row['doctor']}")
        if row.get('cures') and 'low' in str(row.get('risk_level', '')).lower():
            st.info(f"**If mild:** {row['cures']}. Not a prescription — see a doctor if symptoms persist or worsen.")

        query = urllib.parse.quote(f"{chosen_disease} symptoms causes treatment")
        st.link_button("🔎 Search more about this", f"https://www.google.com/search?q={query}")

# ---------------- TAB 3: feedback box (feedback #3) ----------------
with tab_feedback:
    st.subheader("Send feedback or report an issue")
    fb_name = st.text_input("Your name (optional)", key="fb_name")
    fb_message = st.text_area("What's on your mind?")
    if st.button("Submit feedback"):
        if not fb_message.strip():
            st.error("Please write a message before submitting.")
        else:
            sent, error = save_feedback(fb_name, fb_message.strip())
            if sent:
                st.success("Thanks — your feedback was sent!")
            else:
                # Still saved to feedback.csv even though email didn't go out
                st.success("Thanks — your feedback was recorded!")
                if error and "not configured" not in error:
                    st.caption(f"(Note for app owner: email send failed — {error})")
