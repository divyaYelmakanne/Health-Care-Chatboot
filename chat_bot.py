# chatbot.py (Streamlit Version)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Streamlit UI Setup
# -------------------------
# This MUST be the first Streamlit command.
st.set_page_config(page_title="Career Guidance Chatbot", layout="centered")

st.title("💼 Career Guidance Chatbot")

# -------------------------
# Load & Preprocess Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CareerMap- Mapping Tech Roles With Personality & Skills.csv")

    # Encode target (career roles)
    le = LabelEncoder()
    df["Role"] = le.fit_transform(df["Role"])

    # Encode non-numeric features (if any)
    for col in df.columns:
        if df[col].dtype == "object" and col != "Role":
            df[col] = LabelEncoder().fit_transform(df[col])

    return df, le

@st.cache_resource
def train_model(df):
    X = df.drop("Role", axis=1)
    y = df["Role"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, list(X.columns)

# Load data and train model
df, le = load_data()
model, questions = train_model(df)


# -------------------------
# Session State Initialization
# -------------------------
# Initialize all session state variables at the beginning to avoid KeyErrors
if "answers" not in st.session_state:
    st.session_state.answers = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_input" not in st.session_state:
    st.session_state.show_input = True
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -------------------------
# Chatbot Logic
# -------------------------
def chatbot_response(user_input):
    if user_input:
        try:
            st.session_state.answers.append(int(user_input))
        except ValueError:
            st.session_state.chat_history.append(f"**Bot:** ⚠️ Please enter a number between 1 and 7.")
            return

    # Check for prediction
    if len(st.session_state.answers) == len(questions):
        st.session_state.show_input = False # Hide input box after prediction
        user_features = pd.DataFrame([st.session_state.answers], columns=questions)
        prediction = model.predict(user_features)[0]
        career = le.inverse_transform([prediction])[0]
        st.session_state.chat_history.append(f"**Bot:** ✅ Based on your skills, I suggest you explore a career as: **{career}**")
        return

    # Ask next question
    if st.session_state.current_q < len(questions):
        bot_msg = f"Rate your skill level in **{questions[st.session_state.current_q]}** (1-7):"
        st.session_state.current_q += 1
        st.session_state.chat_history.append(f"**Bot:** {bot_msg}")

def handle_input():
    if st.session_state.user_input:
        user_input = st.session_state.user_input
        st.session_state.chat_history.append(f"**You:** {user_input}")
        chatbot_response(user_input)
        st.session_state.user_input = ""

# -------------------------
# Display Chat History & Input
# -------------------------
for chat in st.session_state.chat_history:
    st.markdown(chat)

# Start conversation if it's the first run
if st.session_state.current_q == 0 and not st.session_state.chat_history:
    st.session_state.chat_history.append(f"**Bot:** Hi! I'll ask you about your skills.")
    st.session_state.current_q += 1
    st.session_state.chat_history.append(f"**Bot:** Rate your skill level in **{questions[st.session_state.current_q-1]}** (1-7):")

# Input box section at the bottom
if st.session_state.show_input:
    st.text_input("Your answer (1-7):", key="user_input", on_change=handle_input)
