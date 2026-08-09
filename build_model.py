
import pandas as pd
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

d1 = pd.read_csv('dataset.csv')
d2 = pd.read_csv('disease_data.csv')
desc = pd.read_csv('Disease_Description.csv')
test_csv = pd.read_csv('Testing.csv')  

d1['disease'] = d1['disease'].str.strip().str.lower()
d1['symptom_list'] = d1['symptoms'].apply(lambda x: set(s.strip().lower() for s in str(x).split(',')))

d2['Disease'] = d2['Disease'].str.strip().str.lower()
d2['symptom_list'] = d2['Symptom'].apply(lambda x: set(s.strip().lower() for s in ast.literal_eval(x)))


d2['Disease'] = d2['Disease'].replace({'fibromyalgi': 'fibromyalgia'})

desc['Disease'] = desc['Disease'].str.strip().str.lower()


symptom_cols = [c for c in test_csv.columns if c != 'prognosis']
test_csv['disease'] = test_csv['prognosis'].str.strip().str.lower()
test_csv['symptom_list'] = test_csv.apply(
    lambda row: set(col.replace('_', ' ').strip() for col in symptom_cols if row[col] == 1), axis=1
)

kb = {}

for _, row in d1.iterrows():
    kb[row['disease']] = {
        'symptoms': row['symptom_list'],
        'cures': row.get('cures', ''),
        'doctor': row.get('doctor', ''),
        'risk_level': row.get('risk level', ''),
        'description': ''
    }

for _, row in d2.iterrows():
    name = row['Disease']
    if name in kb:
        kb[name]['symptoms'] = kb[name]['symptoms'].union(row['symptom_list'])
    else:
        kb[name] = {'symptoms': row['symptom_list'], 'cures': '', 'doctor': '', 'risk_level': '', 'description': ''}

for _, row in test_csv.iterrows():
    name = row['disease']
    if name in kb:
        kb[name]['symptoms'] = kb[name]['symptoms'].union(row['symptom_list'])
    else:
        kb[name] = {'symptoms': row['symptom_list'], 'cures': '', 'doctor': '', 'risk_level': '', 'description': ''}

for _, row in desc.iterrows():
    name = row['Disease']
    if name in kb:
        kb[name]['description'] = row['Description']
    else:
        kb[name] = {'symptoms': set(), 'cures': '', 'doctor': '', 'risk_level': '', 'description': row['Description']}

kb_df = pd.DataFrame([
    {'disease': k, 'symptoms': [s for s in v['symptoms'] if s.strip()], 'cures': v['cures'],
     'doctor': v['doctor'], 'risk_level': v['risk_level'], 'description': v['description']}
    for k, v in kb.items()
])
kb_df.to_csv('disease_knowledge_base.csv', index=False)
print(f"Knowledge base built: {len(kb_df)} diseases saved to disease_knowledge_base.csv")



risk_df = pd.read_csv('synthetic_disease_risk_dataset_csv.csv')
risk_df = risk_df.dropna(subset=['Disease_Risk'])

X = risk_df[['BMI']]
y = risk_df['Disease_Risk'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

bmi_model = LogisticRegression()
bmi_model.fit(X_train, y_train)

y_pred = bmi_model.predict(X_test)
print("\nBMI risk model performance:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Elevated Risk']))

joblib.dump(bmi_model, 'bmi_risk_model.pkl')
print("BMI risk model saved to bmi_risk_model.pkl")

print("\nDone. Download disease_knowledge_base.csv and bmi_risk_model.pkl")
print("from the Colab file browser — you'll need both for the Streamlit app.")
