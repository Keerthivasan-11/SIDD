import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st
from tensorflow.keras.models import load_model

# Set a custom title and description with HTML
st.set_page_config(page_title="Dominant Dosha Predictor", layout="wide")
st.title("💫 Dominant Dosha Prediction 💫")
st.markdown("""
    Welcome to the **Dominant Dosha Prediction** tool! 🙏
    This tool helps you find out your dominant dosha based on a few simple questions.
    **Answer the following survey questions**, and hit the **Predict** button to get your result! ✨
""")

file_url = 'https://raw.githubusercontent.com/Keerthivasan-11/SIDD/main/IMPORTANT%20VARIABLES.csv'

# Read the CSV file from the raw URL
data = pd.read_csv(file_url, encoding='latin')

# Separate features and target
X = data.drop(columns=['Dominant_Dosha'])  # Replace 'Dominant_Dosha' with your actual target column name
y = data['Dominant_Dosha']

# One-hot encode categorical features
for column in X.select_dtypes(include=['object']).columns:
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = onehot_encoder.fit_transform(X[[column]])
    encoded_columns = [f"{column}_{cat}" for cat in onehot_encoder.categories_[0]]
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=X.index)
    X = X.drop(columns=[column])
    X = pd.concat([X, encoded_df], axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Get the number of features from the processed training data
number_of_features = X.shape[1]

# Sidebar for Survey Questions
st.sidebar.header("Survey Questions")
survey_questions = {
    "BODY FRAME": ["Thin and unusually tall or short", "Medium body", "Large body"],
    "TEETH ALIGNMENT": ["Crooked, uneven, buck teeth", "Med., even teeth, gums bleed easily", "Large, even, gleaming teeth"],
    "WORK": ["Creative, spontaneous artist", "Organized thinker/leader", "Prefers to work under routine"],
    "WORK NATURE": [
        "Does many projects at once, prefers multitasking",
        "Prefers organization and order in projects",
        "Prefers simplicity, dislikes complications",
    ],
    "WEATHER": ["Prefers warm climate", "Prefers cool, well-ventilated places", "Tolerates most climates"],
    "SLEEP CYCLE": ["Light sleeper, wakes frequently", "Usually sleeps well", "Sound, heavy sleeper (8+ hours)"],
    "TOES": ["Long tapering fingers/toes", "Medium fingers/toes", "Large fingers/toes"],
    "BODY HAIR": [
        "Body hair is scanty or overly abundant, coarse, curly",
        "Light body hair, finely textured",
        "Moderate amount of body hair",
    ],
    "DAILY LIFE": ["Dislikes routine", "Enjoys planning/organizing", "Works well with routine"],
    "SKIN TEXTURE": [
        "Dry skin, prone to calluses",
        "Oily skin, prone to pimples and rashes",
        "Thicker skin, usually beautiful",
    ],
}

user_responses = []
for question, options in survey_questions.items():
    choice = st.sidebar.radio(question, options)
    user_responses.append(options.index(choice) + 1)

# Add a beautiful button with an icon
button_style = """
    <style>
    .stButton>button {
        background-color: #FF5722;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #FF7043;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Button to trigger the prediction
if st.button("✨ Predict Dominant Dosha ✨"):
    # Preprocess user responses (convert to one-hot encoding)
    user_input_encoded = []
    for i, question in enumerate(survey_questions.keys()):
        response_vector = [0] * 3  # Assuming 3 options for each question
        response_vector[user_responses[i] - 1] = 1  # Set the chosen option to 1
        user_input_encoded.extend(response_vector)

    # Convert to numpy array and reshape
    user_input = np.array(user_input_encoded).reshape(1, -1)

    # Ensure the input matches the expected number of features (pad if necessary)
    if user_input.shape[1] < number_of_features:
        padded_input = np.zeros((1, number_of_features))
        padded_input[:, :user_input.shape[1]] = user_input
        user_input = padded_input

    # Reshape the input to match CNN's expected input shape (1, features, 1)
    user_input = np.expand_dims(user_input, axis=2)

    # Load the trained model
    model = load_model('dominant_dosha_cnn_model.h5')

    # Make predictions
    predictions = model.predict(user_input)
    predicted_class = np.argmax(predictions, axis=1)

    # Decode the predicted class
    class_labels = label_encoder.classes_
    predicted_dosha = class_labels[predicted_class[0]]

    # Output the result with a fancy heading and icon
    st.markdown(f"### 🧘‍♂️ **Your Dominant Dosha is: {predicted_dosha}** 🧘‍♀️")
    st.balloons()  # Add confetti balloons to celebrate!
