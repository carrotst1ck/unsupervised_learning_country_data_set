import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Country Dataset Clustering", layout="wide")

data = pd.read_csv("Country_Dataset.csv")
numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
X = data[numeric_columns].copy()

try:
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    st.error("Сначала запусти train.py")
    st.stop()

X_scaled = scaler.transform(X)
labels = model.predict(X_scaled)
data["Cluster"] = labels

sil_score = silhouette_score(X_scaled, labels)

st.title("Country Dataset Clustering")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Information")
    st.write(f"Number of clusters: {model.n_clusters}")
    st.write(f"Silhouette score: {sil_score:.4f}")
    st.write("Numeric columns used:")
    st.write(feature_columns)

with col2:
    st.subheader("10 Samples from the Dataset")
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("Figure")

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    data["income"],
    data["life_expec"],
    c=data["Cluster"]
)
ax.set_xlabel("income")
ax.set_ylabel("life_expec")
ax.set_title("Clusters by income and life expectancy")
st.pyplot(fig)

st.subheader("Input Data for Cluster Prediction")

user_values = []
input_cols = st.columns(3)

for i, col in enumerate(feature_columns):
    with input_cols[i % 3]:
        val = st.number_input(col, value=float(X[col].mean()))
        user_values.append(val)

if st.button("Predict Cluster"):
    input_data = pd.DataFrame([user_values], columns=feature_columns)
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    st.success(f"This input belongs to cluster: {cluster}")
