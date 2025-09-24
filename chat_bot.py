import re
import pandas as pd
import streamlit as st
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# -----------------------------
# Load datasets
# -----------------------------
training = pd.read_csv("Data/Training.csv")
testing = pd.read_csv("Data/Testing.csv")
cols = training.columns[:-1]
x = training[cols]
y = training["prognosis"]

# Label encoding for target
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Train SVM for comparison
model = SVC()
model.fit(x_train, y_train)

# Global dicts
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# -----------------------------
# Helper functions
# -----------------------------
def getDescription():
    global description_list
    with open("MasterData/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip header if exists
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0].strip()] = row[1].strip()

def getSeverityDict():
    global severityDictionary
    with open("MasterData/symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip header if exists
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    severityDictionary[row[0].strip()] = int(row[1])
                except ValueError:
                    continue

def getprecautionDict():
    global precautionDictionary
    with open("MasterData/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip header if exists
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0].strip()] = [row[1], row[2], row[3], row[4]]

def calc_condition(exp, days):
    severity_sum = sum(severityDictionary.get(item, 0) for item in exp)
    if (severity_sum * days) / (len(exp) + 1) > 13:
        return "⚠️ You should take consultation from a doctor."
    else:
        return "🙂 It might not be that bad but you should take precautions."

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 AI Health Assistant")
st.write("Enter your symptoms below and get disease predictions with precautions.")

# Load dictionaries
getSeverityDict()
getDescription()
getprecautionDict()

# Collect symptoms from user
symptoms_list = cols.tolist()
selected_symptoms = st.multiselect("Select your symptoms:", symptoms_list)

days = st.number_input("For how many days have you had these symptoms?", min_value=1, max_value=30, value=1)

if st.button("Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Build input vector
        input_data = [0] * len(symptoms_list)
        for s in selected_symptoms:
            if s in symptoms_list:
                input_data[symptoms_list.index(s)] = 1
        disease = clf.predict([input_data])[0]
        disease_name = le.inverse_transform([disease])[0]

        st.subheader(f"🧾 Prediction: {disease_name}")

        # Show description
        desc = description_list.get(disease_name, "No description available.")
        st.write(f"**Description:** {desc}")

        # Show precautions
        precs = precautionDictionary.get(disease_name, ["No precautions available."])
        st.write("**Precautions:**")
        for i, p in enumerate(precs, 1):
            st.write(f"{i}. {p}")

        # Show severity analysis
        advice = calc_condition(selected_symptoms, days)
        st.info(advice)
