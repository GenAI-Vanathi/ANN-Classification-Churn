import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
##from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle

# Load the ANN trained model 
model = tf.keras.models.load_model('model_tf2')

# Load Encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.write("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#Preparing Input data

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])

# One-hot Encode 'Geogrphy'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one hot encoded column with input data
input_full = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_full)

# Make Prediction
prediction = model.predict(input_scaled)
churn = prediction[0][0]

if churn > 0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')

st.write(f'Churn probability: {prediction[0][0]:.2f}')
