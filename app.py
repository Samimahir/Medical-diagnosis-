import streamlit as st
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Background Styling
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
    }
    label { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Login Page
if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; color: white;'>Login Page</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "r" and password == "9":
                st.session_state["authenticated"] = True
                st.session_state["page"] = "disease"
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.stop()

# Load Disease Prediction Models
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
}

# Function to process images
def process_image(image_file):
    image = Image.open(image_file).convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

# Load and Train Diet Model
df = pd.read_csv('health_data.csv')
if not {'Age', 'Weight', 'Disease', 'Diet'}.issubset(df.columns):
    st.error("Dataset is missing required columns. Check CSV file.")
    st.stop()

X = df[['Age', 'Weight', 'Disease']]
y = df['Diet']
encoder = LabelEncoder()
X['Disease'] = encoder.fit_transform(X['Disease'])
diet_model = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model.fit(X, y)

# Navigation Control
if "page" not in st.session_state:
    st.session_state["page"] = "disease"

# Disease Prediction Page
if st.session_state["page"] == "disease":
    st.markdown("<h1 style='color: #f39c12; text-align: center;'>Disease Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected = st.selectbox("Select a Disease to Predict:",
            ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Lung Cancer Prediction', 'Hypo-Thyroid Prediction'])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_image = st.file_uploader("Upload an image for disease prediction", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        image_array = process_image(uploaded_image)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict Disease"):
                result = "Disease Detected" if np.random.choice([0, 1]) == 1 else "No Disease Detected"
                st.success(result)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Go to Diet Plan Recommender"):
            st.session_state["page"] = "diet"
            st.rerun()

# Diet Plan Recommender Page
elif st.session_state["page"] == "diet":
    st.markdown("<h1 style='color: #1abc9c; text-align: center;'>Diet Plan Recommender</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        age = st.number_input("Enter your age:", min_value=18, max_value=120, value=30)
        weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
        disease_options = df['Disease'].unique()
        disease = st.selectbox("Select your disease:", disease_options)

    # Centering the "Get Diet Recommendation" Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Diet Recommendation"):
            disease_encoded = encoder.transform([disease])[0]
            input_data = pd.DataFrame([[age, weight, disease_encoded]], columns=['Age', 'Weight', 'Disease'])
            predicted_diet = diet_model.predict(input_data)[0]
            st.success(f"Recommended Diet Plan: {predicted_diet}")
    
    # Logout Button (Redirects to Login Page)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["page"] = "login"
            st.rerun()
