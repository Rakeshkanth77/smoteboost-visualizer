import streamlit as st
from data_handler.data_handler import DataHandler
from imblearn.datasets import fetch_datasets
import matplotlib.pyplot as plt

st.title("Welcome to SmoteBoost Visualizer")

st.sidebar.header("Selection")
all_datasets = list(fetch_datasets().keys())
dataset = st.sidebar.selectbox("Choose Dataset", all_datasets)
smote_param = st.sidebar.slider("SMOTE k_neighbors", 1, 10, 5)
smote_ratio = st.sidebar.slider("SMOTE Sampling Ratio", 0.1, 1.0, 0.5)
smote_strategy = st.sidebar.selectbox("SMOTE Strategy", ["minority", "not majority", "all", "auto"])


@st.cache_data
def load_data(dataset):
    X, y, class_distribution, feature_names = DataHandler.load_data(dataset)

    return X, y, class_distribution, feature_names

X, y, class_distribution, feature_names = load_data(dataset)
st.subheader("Dataset Overview")
st.write(f"Selected Dataset: **{dataset}**")

# Plot: Class Distribution Bar Plot
# Plot: Class Distribution Bar Plot (Compact)
fig1, ax1 = plt.subplots(figsize=(4, 2.5))
classes = list(class_distribution.keys())
counts = list(class_distribution.values())
ax1.bar(classes, counts, color='skyblue', edgecolor='black')
ax1.set_xlabel("Class", fontsize=8)
ax1.set_ylabel("Count", fontsize=8)
ax1.set_title("Class Distribution", fontsize=9, fontweight='bold')
plt.tight_layout()
st.pyplot(fig1)




