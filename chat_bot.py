import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv

# --- Data Loading and Model Training ---
@st.cache_data
def load_data():
    """Loads and preprocesses data for the model."""
    try:
        training = pd.read_csv('Data/Training.csv')
        testing = pd.read_csv('Data/Testing.csv')
    except FileNotFoundError:
        st.error("Data files not found. Please check that 'Data/Training.csv' and 'Data/Testing.csv' exist.")
        st.stop()
        
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.33, random_state=42)
    
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    return clf, le, cols, y, training

clf, le, cols, y_encoded, training_df = load_data()

# --- Utility Functions ---
@st.cache_data
def get_master_data(file_path):
    """Loads a master data CSV file into a dictionary."""
    data_dict = {}
    try:
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row:
                    data_dict[row[0]] = row[1:] if len(row) > 1 else row[0]
    except FileNotFoundError:
        st.error(f"Master data file not found: {file_path}")
        st.stop()
    return data_dict

severityDictionary = get_master_data('MasterData/symptom_severity.csv')
description_list = get_master_data('MasterData/symptom_Description.csv')
precautionDictionary = get_master_data('MasterData/symptom_precaution.csv')

def get_symptom_info(symptoms_list):
    """
    Calculates a 'condition score' based on symptoms and days.
    (Simplified for Streamlit UI)
    """
    try:
        sum_severity = sum(int(severityDictionary.get(item, ['0'])[0]) for item in symptoms_list)
        return sum_severity
    except (ValueError, IndexError):
        return 0

# --- Streamlit UI and App Logic ---
st.header("👨‍⚕️ HealthCare ChatBot")
st.markdown("Please select your symptoms to get a likely diagnosis and precautions.")

# Get all unique symptoms from the training data
all_symptoms = cols.to_list()

# --- Main app flow ---
# Use a multiselect box for user to choose symptoms
selected_symptoms = st.multiselect(
    "Select the symptoms you are experiencing:",
    options=all_symptoms,
    key="symptoms"
)

# Number of days slider
num_days = st.slider("Number of days you've had these symptoms:", 1, 30, 3)

# Prediction button
predict_button = st.button("Get Diagnosis")

if predict_button and selected_symptoms:
    # Create an input vector for the model
    input_vector = np.zeros(len(all_symptoms))
    symptom_to_index = {symptom: i for i, symptom in enumerate(all_symptoms)}
    
    for symptom in selected_symptoms:
        if symptom in symptom_to_index:
            input_vector[symptom_to_index[symptom]] = 1
            
    # Make a prediction
    prediction_encoded = clf.predict([input_vector])
    predicted_disease = le.inverse_transform(prediction_encoded)[0]

    # Display the result
    st.subheader("Diagnosis Result")
    st.success(f"You may have: **{predicted_disease}**")

    # Display description and precautions
    with st.expander("Details about the disease"):
        st.markdown(f"**Description:**")
        if predicted_disease in description_list:
            st.write(description_list[predicted_disease][0])
        else:
            st.write("Description not available.")
            
        st.markdown(f"**Precautions to take:**")
        if predicted_disease in precautionDictionary:
            precautions = precautionDictionary[predicted_disease]
            for i, p in enumerate(precautions):
                if p:
                    st.write(f"• {p}")
        else:
            st.write("Precautions not available.")
            
    # Display simplified condition note
    condition_score = get_symptom_info(selected_symptoms)
    if condition_score > 13:
        st.warning("Based on the severity of your symptoms, you should take consultation from a doctor.")
    else:
        st.info("It might not be that bad, but you should take precautions.")

elif predict_button and not selected_symptoms:
    st.warning("Please select at least one symptom.")
