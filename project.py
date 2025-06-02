import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("smart_buildings_energy.csv")  # Ensure correct filename
    return df

df = load_data()

# Train Model
def train_model(df, model_type, n_estimators=100, max_depth=10):
    features = ["temperature", "humidity", "occupancy", "smart_devices"]
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train, y_test = train["energy_consumption"], test["energy_consumption"]
    
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2_score(y_test, y_pred),  # Added R² Score
    }
    
    return model, scaler, metrics, X_test, y_test, y_pred

# Streamlit UI
st.title("🏢 Smart Building Energy Efficiency Prediction")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("⚙️ Model Selection")
model_type = st.selectbox("Choose Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
n_estimators = st.slider("Number of Trees (for RF & GB)", 50, 500, 100, step=50) if model_type != "Linear Regression" else 0
max_depth = st.slider("Max Depth", 5, 50, 20, step=5)

# Train and evaluate
model, scaler, metrics, X_test, y_test, y_pred = train_model(df, model_type, n_estimators, max_depth)

st.subheader("📈 Model Performance Metrics")
st.write(metrics)  # Now includes R² Score

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_test[:, 0], y_test, label="Actual", color="blue")
ax.scatter(X_test[:, 0], y_pred, label="Predicted", color="red", alpha=0.7)
ax.set_xlabel("Temperature")
ax.set_ylabel("Energy Consumption")
ax.set_title("Actual vs Predicted Energy Consumption")
ax.legend()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Future Predictions
st.subheader("🔮 Future Predictions")
future_entries = st.number_input("Enter number of future entries:", min_value=1, max_value=100, value=5)
future_data = {
    "temperature": np.random.uniform(18, 30, future_entries),
    "humidity": np.random.uniform(30, 70, future_entries),
    "occupancy": np.random.randint(0, 100, future_entries),
    "smart_devices": np.random.randint(1, 10, future_entries),
}

future_df = pd.DataFrame(future_data)
future_scaled = scaler.transform(future_df)
future_preds = model.predict(future_scaled)

future_df["Predicted Energy Consumption"] = future_preds
st.dataframe(future_df)

# Save Predictions
st.download_button("📥 Download Predictions (CSV)", future_df.to_csv(index=False), file_name="future_predictions.csv", mime="text/csv")
st.download_button("📥 Download Predictions (JSON)", future_df.to_json(), file_name="future_predictions.json", mime="application/json")