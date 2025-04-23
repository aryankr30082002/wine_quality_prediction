import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = load_data()

# ✅ Use only 4 selected features
selected_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']
model = RandomForestClassifier()
model.fit(df[selected_features], df['target'])

# Sidebar inputs
st.sidebar.title("Wine Feature Inputs")
alcohol = st.sidebar.slider("Alcohol", float(df['alcohol'].min()), float(df['alcohol'].max()))
malic_acid = st.sidebar.slider("Malic Acid", float(df['malic_acid'].min()), float(df['malic_acid'].max()))
ash = st.sidebar.slider("Ash", float(df['ash'].min()), float(df['ash'].max()))
alcalinity = st.sidebar.slider("Alcalinity of ash", float(df['alcalinity_of_ash'].min()), float(df['alcalinity_of_ash'].max()))

# Make input array
input_data = [[alcohol, malic_acid, ash, alcalinity]]

# Prediction
prediction = model.predict(input_data)

# Map numeric class to custom labels
class_map = {
    0: "Bad 🍷",
    1: "Average 🍷🍷",
    2: "Good 🍷🍷🍷"
}
predicted_class = class_map[prediction[0]]

# Show output
st.write("### 🍷 Wine Quality Prediction")
st.write(f"The predicted wine quality is: **{predicted_class}**")
