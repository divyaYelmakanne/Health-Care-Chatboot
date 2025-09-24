import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
import warnings
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------- Load Data -------------------
training = pd.read_csv("Data/Training.csv")
testing = pd.read_csv("Data/Testing.csv")

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Label Encoding
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = le.transform(testing['prognosis'])

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# ------------------- Dictionaries -------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
reduced_data = training.groupby(training['prognosis']).max()

def getSeverityDict():
    with open("MasterData/symptom_severity.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            severityDictionary[row[0]] = int(row[1])

def getDescription():
    with open("MasterData/symptom_Description.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            description_list[row[0]] = row[1]

def getprecautionDict():
    with open("MasterData/symptom_precaution.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

getSeverityDict()
getDescription()
getprecautionDict()

# ------------------- Helper Functions -------------------
def sec_predict(symptoms_exp):
    df = pd.read_csv("Data/Training.csv")
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])

def calc_condition(exp, days):
    total = sum(severityDictionary[item] for item in exp)
    if (total * days) / (len(exp) + 1) > 13:
        return "⚠️ You should take consultation from a doctor."
    else:
        return "🙂 It might not be that bad, but you should take precautions."

# ------------------- Streamlit UI -------------------
st.title("🩺 Healthcare Chatbot")
st.write("Predict possible diseases based on your symptoms.")

name = st.text_input("Enter your Name")
if name:
    st.success(f"Hello, {name}! Please select your symptoms below.")

symptoms_selected = st.multiselect("Select the symptoms you are experiencing:", options=cols)

days = st.number_input("From how many days are you experiencing symptoms?", min_value=1, max_value=30, value=1)

if st.button("Predict Disease"):
    if len(symptoms_selected) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        second_prediction = sec_predict(symptoms_selected)
        present_disease = le.inverse_transform(clf.predict([x.loc[0]*0 + [sym in symptoms_selected for sym in cols]]))

        # Condition Severity
        condition_msg = calc_condition(symptoms_selected, days)
        st.info(condition_msg)

        # Disease Prediction
        if present_disease[0] == second_prediction[0]:
            st.subheader(f"✅ You may have: **{present_disease[0]}**")
            st.write(description_list.get(present_disease[0], "No description available."))
        else:
            st.subheader(f"Possible Diseases: **{present_disease[0]}** or **{second_prediction[0]}**")
            st.write(description_list.get(present_disease[0], ""))
            st.write(description_list.get(second_prediction[0], ""))

        # Precautions
        prec = precautionDictionary.get(present_disease[0], [])
        if prec:
            st.subheader("🩹 Recommended Precautions:")
            for i, p in enumerate(prec):
                st.write(f"{i+1}. {p}")
