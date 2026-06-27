import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_training_data():
    return pd.read_csv( "Airline Passenger Satisfaction.csv")

df = load_training_data()
df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'], inplace=True)
delayCol = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
for col in delayCol:
    df[col] = df[col].fillna(0)
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()


@st.cache_resource
def load_my_model():
    return joblib.load("satisfaction_model.pkl")

model = load_my_model()


st.set_page_config(page_title="Satisfaction Predictor", layout="centered")
st.title("🔮 Machine Learning Prediction App")
st.write("Enter the required details below to get an instant prediction from the model.")

feature_names = df.columns.tolist()
user_inputs = {}

columns = st.columns(3)

for i, feature in enumerate(feature_names):
    with columns[i % 3]:
        if feature in categorical_features:
            user_inputs[feature] = st.selectbox(f"Select {feature}:", df[feature].unique())
        else:
            min_value = int(df[feature].min())
            max_value = int(df[feature].max())
            user_inputs[feature] = st.slider(f"Select {feature}:", min_value, max_value, (min_value + max_value) // 2)

st.subheader("📋 Input Features")

df_user = pd.DataFrame([user_inputs])

df_user['Entertainment'] = df_user['Inflight wifi service'] + df_user['Food and drink'] + df_user['Seat comfort'] + df_user['Inflight entertainment']
df_user['Online_service'] = df_user['Ease of Online booking'] + df_user['Online boarding']
df_user['Service_quality'] = df_user['On-board service'] + df_user['Leg room service'] + df_user['Baggage handling'] + df_user['Checkin service'] + df_user['Inflight service'] + df_user['Cleanliness']
df_user['Total_service'] = df_user['Online_service'] + df_user['Service_quality']
df_user['Timing_crisis'] = df_user['Departure/Arrival time convenient']*(-1) - df_user['Departure Delay in Minutes'] - df_user['Arrival Delay in Minutes']

st.markdown("---")

expected_order = joblib.load('feature_columns.pkl')
df_user = df_user[expected_order]

if st.button("🚀 Run Prediction", type="primary"):
    with st.spinner("Calculating..."):
        prediction = model.predict(df_user)
        st.success(f"🔮 **Predicted Value:** {prediction[0]}")