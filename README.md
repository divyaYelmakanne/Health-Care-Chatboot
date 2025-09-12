# Healthcare Chatbot

## Description
This is a healthcare chatbot built using machine learning techniques with scikit-learn. The chatbot interacts with users to gather symptoms, asks follow-up questions, and predicts potential diseases based on the input. It provides detailed descriptions of the predicted diseases and suggests precautions to take.

The chatbot uses a Decision Tree Classifier as the primary model, with SVM as an alternative for secondary predictions. It incorporates symptom severity scores to assess the condition's seriousness and recommends consulting a doctor if necessary.

## Features
- Interactive symptom input and follow-up questioning
- Disease prediction using machine learning models
- Detailed disease descriptions
- Precaution recommendations
- Severity assessment based on symptoms and duration
- Text-to-speech output for accessibility
- Pattern matching for symptom input to handle variations

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd healthcare-chatbot-master
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the chatbot:
   ```
   python chat_bot.py
   ```

2. Follow the prompts:
   - Enter your name
   - Input the symptom you are experiencing
   - Specify the number of days you've had the symptom
   - Answer follow-up questions about additional symptoms
   - Receive the diagnosis, description, and precautions

The chatbot will use text-to-speech to read out the results.

## Dataset
The project uses the following datasets:
- `Data/Training.csv`: Training data with symptoms and corresponding diseases
- `Data/Testing.csv`: Testing data for model evaluation
- `Data/dataset.csv`: Combined dataset
- `MasterData/symptom_Description.csv`: Descriptions for each disease
- `MasterData/symptom_precaution.csv`: Precautions for each disease
- `MasterData/Symptom_severity.csv`: Severity scores for symptoms

## How It Works
1. **Model Training**: The Decision Tree Classifier is trained on the training data using symptoms as features and diseases as labels.

2. **User Interaction**:
   - User inputs initial symptom
   - Chatbot traverses the decision tree to gather more symptoms
   - Uses pattern matching to handle input variations

3. **Prediction**:
   - Primary prediction using Decision Tree
   - Secondary prediction using SVM for confirmation
   - Calculates condition severity based on symptom severity scores and duration

4. **Output**:
   - Predicted disease(s)
   - Disease description
   - Precautions to take
   - Recommendation to consult a doctor if severity is high

## Dependencies
- scikit-learn: For machine learning models
- pandas: For data manipulation
- pyttsx3: For text-to-speech functionality

## Contributing
This project is no longer maintained. For any expansions or improvements, please fork the repository and make changes to your own version.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
