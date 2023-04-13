import streamlit as st
import pickle
import re
import string
import pandas as pd

# Load the trained model and vectorizer using pickle
with open('model_s.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the dataframe from a pickle file
with open('df_app.pkl', 'rb') as f:
    df = pd.read_pickle(f)

# Set the title
st.title("Review based Medicinde recomendation")



# Define a function to preprocess user input
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove whitespace
    text = text.strip()

    return text



def display_dataframe(option):
    if option == 'Depression':
        df_filtered = df[df['condition'] == 'Depression'].tail(5).iloc[::-1].reset_index(drop=True).drop(columns='cumilative')
        df_filtered.index += 1
        st.table(df_filtered)
    elif option == 'Type 2 Diabetes':
        df_filtered = df[df['condition'] == 'Diabetes, Type 2'].tail(5).iloc[::-1].reset_index(drop=True).drop(columns='cumilative')
        df_filtered.index += 1
        st.table(df_filtered)
    else:
        df_filtered = df[df['condition'] == 'High Blood Pressure'].tail(5).iloc[::-1].reset_index(drop=True).drop(columns='cumilative')
        df_filtered.index += 1
        st.table(df_filtered)


# Create a text input box for user input
user_input = st.text_input('Enter a text')

# Create a button to submit the input text
if st.button('Submit'):
    # Preprocess the user input
    if user_input:
        user_input_processed = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_processed])

        # Make prediction using the loaded model
        prediction = model.predict(user_input_vectorized)[0]

        # Display the prediction
        st.write('Prediction:', prediction)

        # Display the filtered dataframe based on the prediction
        display_dataframe(prediction)
    else:
        st.warning('Please enter a text input')
