# =========================================================
# TEST SCRIPT — checks how accurate the SYMPTOM matching is
# =========================================================
# Run this AFTER build_model.py has already created disease_knowledge_base.csv
#
# What this does: takes the 42 real labeled cases in Testing.csv, and for each one,
# only gives the matcher HALF of that disease's known symptoms (to simulate a real
# person who doesn't list every single symptom), then checks if it still guesses
# the right disease.
#
# Needs these files in the same folder:
#   - disease_knowledge_base.csv   (created by build_model.py)
#   - Testing.csv

import pandas as pd
import ast
import random

random.seed(42)

kb_df = pd.read_csv('disease_knowledge_base.csv')
kb_df['symptoms'] = kb_df['symptoms'].apply(ast.literal_eval)

def predict_diseases(user_symptoms, kb_df, top_n=5):
    user_symptoms = set(s.strip().lower() for s in user_symptoms)
    results = []
    for _, row in kb_df.iterrows():
        disease_symptoms = set(row['symptoms'])
        if not disease_symptoms:
            continue
        overlap = user_symptoms & disease_symptoms
        if len(overlap) == 0:
            continue
        score = len(overlap) / len(disease_symptoms | user_symptoms)
        results.append({'disease': row['disease'], 'score': score})
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

test_df = pd.read_csv('Testing.csv')
symptom_cols = [c for c in test_df.columns if c != 'prognosis']

correct_top1 = 0
correct_top5 = 0
total = len(test_df)

print("Testing each case (showing first 10):\n")

for i, row in test_df.iterrows():
    actual = row['prognosis'].strip().lower()
    full_symptoms = [col.replace('_', ' ') for col in symptom_cols if row[col] == 1]

    # Simulate a realistic user who only reports about half their symptoms
    if len(full_symptoms) > 1:
        reported = random.sample(full_symptoms, k=max(1, len(full_symptoms)//2))
    else:
        reported = full_symptoms

    preds = predict_diseases(reported, kb_df, top_n=5)
    pred_names = [p['disease'] for p in preds]

    top1_hit = pred_names and pred_names[0] == actual
    top5_hit = actual in pred_names

    if top1_hit:
        correct_top1 += 1
    if top5_hit:
        correct_top5 += 1

    if i < 10:
        status = "✅" if top1_hit else ("〰️ in top 5" if top5_hit else "❌")
        print(f"{status} Actual: {actual} | Reported symptoms: {reported} | Top guess: {pred_names[0] if pred_names else 'none'}")

print(f"\n--- FINAL RESULTS on {total} test cases ---")
print(f"Top-1 accuracy (correct on first guess): {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
print(f"Top-5 accuracy (correct answer in top 5 shown to user): {correct_top5}/{total} = {correct_top5/total*100:.1f}%")
