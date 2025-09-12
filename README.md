#🩺 Healthcare Chatbot


##📖 Description

This is a healthcare chatbot built using machine learning (scikit-learn).
The chatbot interacts with users to gather symptoms 🤒, asks follow-up questions ❓, and predicts potential diseases 🧬.
It also provides detailed disease descriptions 📄 and suggests precautions ✅.

The chatbot uses a Decision Tree Classifier 🌳 as the primary model, with SVM 🤖 as an alternative for secondary predictions.
It incorporates symptom severity scores ⚖️ to assess condition seriousness and recommends consulting a doctor 🏥 if necessary.


##✨ Features

💬 Interactive symptom input & follow-up questioning
🔮 Disease prediction using ML models (Decision Tree + SVM)
📄 Detailed disease descriptions
✅ Precaution recommendations
⚖️ Severity assessment based on symptoms & duration
🔊 Text-to-speech output for accessibility
🔎 Pattern matching for symptom input (handles variations)


##⚙️ Installation

1. 📥 Clone the repository:
```
git clone <repository-url>
cd healthcare-chatbot-master
```
2. 📦 Install dependencies:
```
pip install -r requirements.txt
```


##🚀 Usage

1. ▶️ Run the chatbot:
```
python chat_bot.py
```
2. 🧑‍⚕️ Follow the prompts:

✍️ Enter your name
🤒 Input the symptom you’re experiencing
⏳ Specify the number of days you’ve had the symptom
➕ Answer follow-up questions about additional symptoms
🧬 Receive the diagnosis, 📄 description, and ✅ precautions
🔊 The chatbot will speak out results using text-to-speech.


##📊 Dataset

This project uses the following datasets:

📂 Data/Training.csv → Training data with symptoms & diseases
📂 Data/Testing.csv → Testing data for evaluation
📂 Data/dataset.csv → Combined dataset
📂 MasterData/symptom_Description.csv → Disease descriptions
📂 MasterData/symptom_precaution.csv → Precautions for each disease
📂 MasterData/Symptom_severity.csv → Symptom severity scores


##🛠️ How It Works

1. Model Training
🌳 Decision Tree trained on symptoms → diseases

2. User Interaction
👤 User enters symptoms
🔄 Chatbot asks follow-up questions (pattern matching included)

3. Prediction
🎯 Primary prediction → Decision Tree
🤖 Secondary prediction → SVM
⚖️ Severity calculated using symptoms + duration

4. Output
🧬 Predicted disease(s)
📄 Description of disease
✅ Precautions to take
🏥 Recommendation to see doctor if severity is high


##📦 Dependencies
🧮 scikit-learn → ML models
🐼 pandas → Data handling
🔊 pyttsx3 → Text-to-speech


##🤝 Contributing

⚠️ This project is no longer maintained.
If you’d like to expand or improve it → fork the repo and make your changes.

##📜 License

📄 Licensed under the MIT License – see LICENSE file for details.
