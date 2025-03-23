import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset from the CSV file
df = pd.read_csv('health_data.csv')

# Feature and target variables
X = df[['Age', 'Weight', 'Disease']]  # Features (Age, Weight, Disease)
y = df['Diet']  # Target (Diet)

# Encode the 'Disease' column into numeric values
encoder = LabelEncoder()
X['Disease'] = encoder.fit_transform(X['Disease'])

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app title
st.title("Diet Plan Recommender for Health Conditions")

st.write("This app suggests a personalized diet plan based on the health conditions you input.")

# Input fields for user input
age = st.number_input("Enter your age:", min_value=18, max_value=120, value=30)
weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
disease = st.selectbox("Select your disease:", ["Heart Disease", "Lung Disease", "Diabetes"])

# Convert the disease input to numeric form using the same encoder
disease_encoded = encoder.transform([disease])[0]

# Prepare the input data for prediction
input_data = pd.DataFrame([[age, weight, disease_encoded]], columns=['Age', 'Weight', 'Disease'])

# Predict the diet plan using the trained model
predicted_diet = model.predict(input_data)[0]

# Display the predicted diet plan
st.write(f"Recommended Diet Plan: {predicted_diet}")

# Additional details about each disease and diet recommendations
if disease == "Heart Disease":
    st.subheader("Heart Disease Diet Plan")
    st.write("Heart disease patients are recommended to follow a low sodium diet, focus on fruits, vegetables, whole grains, lean proteins, and limit salt intake.")
elif disease == "Lung Disease":
    st.subheader("Lung Disease Diet Plan")
    st.write("Lung disease patients should focus on a high-protein diet to help maintain muscle strength. Foods rich in antioxidants, like berries and green vegetables, are also beneficial.")
else:
    st.subheader("Diabetes Diet Plan")
    st.write("Diabetes patients should follow a low-carb diet, focusing on whole grains, lean proteins, and avoiding sugar-rich foods. Monitor blood sugar levels regularly.")
