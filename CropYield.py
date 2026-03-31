import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="CropYield", layout="wide")

st.title("🌾 CropYield")
st.markdown("Smart Crop Yield Prediction System")

# --------------------------
# Creating Dataset 
# --------------------------
@st.cache_data
def load_data():
    data = {
        "Rainfall": [120, 100, 140, 90, 110, 130, 150, 80],
        "Temperature": [30, 28, 32, 25, 29, 31, 33, 24],
        "Fertilizer": [200, 180, 220, 160, 190, 210, 230, 150],
        "Soil_Quality": [7, 6, 8, 5, 6.5, 7.5, 8.5, 5],
        "Yield": [3.5, 3.0, 4.0, 2.5, 3.2, 3.8, 4.2, 2.3]
    }
    return pd.DataFrame(data)

data = load_data()

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Input Parameters")

rainfall = st.sidebar.slider("Rainfall", 50, 200, 120)
temperature = st.sidebar.slider("Temperature", 20, 40, 30)
fertilizer = st.sidebar.slider("Fertilizer", 100, 300, 200)
soil_quality = st.sidebar.slider("Soil Quality", 1.0, 10.0, 7.0)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest"]
)

# --------------------------
# Prepare Data
# --------------------------
X = data[['Rainfall', 'Temperature', 'Fertilizer', 'Soil_Quality']]
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Model Selection
# --------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

# --------------------------
# Prediction
# --------------------------
input_data = np.array([[rainfall, temperature, fertilizer, soil_quality]])
prediction = model.predict(input_data)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# --------------------------
# Layout
# --------------------------
col1, col2 = st.columns(2)

# Prediction Section
with col1:
    st.subheader("📊 Prediction Result")
    st.success(f"Estimated Yield: {prediction[0]:.2f} tons/hectare")

    st.metric(label="Model Accuracy (R² Score)", value=f"{accuracy:.2f}")

    st.subheader("📋 Input Summary")
    st.write({
        "Rainfall": rainfall,
        "Temperature": temperature,
        "Fertilizer": fertilizer,
        "Soil Quality": soil_quality
    })

# Visualization Section
with col2:
    st.subheader("📈 Visualization")

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="Rainfall", y="Yield", ax=ax)
    st.pyplot(fig)

# --------------------------
# Dataset
# --------------------------

st.subheader("📂 Dataset")

with st.expander("View Dataset"):
    st.dataframe(data)

# --------------------------
# Correlation Heatmap
# --------------------------
st.subheader("🔥 Feature Relationships")

fig2, ax2 = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)