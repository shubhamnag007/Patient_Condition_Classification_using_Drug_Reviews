import streamlit as st
import pickle
import re
import string

# Load the trained model using pickle
with open('model_s.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer using pickle
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    # convert text to lowercase
    text = text.lower()
    
    # remove numbers
    text = re.sub(r'\d+', '', text)
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove whitespace
    text = text.strip()
    
    return text
# Create a streamlit app
st.title('Patient Condition Classification using Drug Reviews')

# Create a text input box for user input
user_input = st.text_input('Enter the drug review')

# Create a button to submit the input text
if st.button('Submit'):
    # Preprocess the user input
    user_input_processed = preprocess_text(user_input)
    user_input_vectorized = vectorizer.transform([user_input_processed])
    
    # Make prediction using the loaded model
    prediction = model.predict(user_input_vectorized)[0]
    
    # Display the prediction
    st.write('Prediction:', prediction)
