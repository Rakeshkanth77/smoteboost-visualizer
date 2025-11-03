from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
from data_handler.data_handler import DataHandler
from imblearn.datasets import fetch_datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_trainer.model_trainer import ModelTrainer

st.title("Welcome to SmoteBoost Visualizer")

import streamlit as st

st.title("SMOTEBoost: Imbalanced Data Handling Demo")  # OBJ-1/2

st.sidebar.header("Dataset Selection")  # F-1
dataset = st.sidebar.selectbox("Choose Dataset", ["Imbalanced Binary (make_classification)"])  # Simple start

st.write(f"Selected: {dataset}")
st.info("Educational demo: Apply SMOTE to balance classes and see model impact.")  # F-7



from data_handler import DataHandler

if st.sidebar.button("Load Dataset"):  # Trigger
    X, y = DataHandler.load_data(dataset)
    X_train, X_test, y_train, y_test = DataHandler.split_data(X, y)
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.write(f"Loaded: Train classes {y_train.value_counts().to_dict()}")

import matplotlib.pyplot as plt

if 'y_train' in st.session_state:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Bar: Class dist (F-2)
    st.session_state.y_train.value_counts().plot(kind='bar', ax=ax[0])
    ax[0].set_title("Train Class Distribution")
    # Scatter: Features (F-2)
    scatter = ax[1].scatter(st.session_state.X_train['Feature1'], st.session_state.X_train['Feature2'],
                            c=st.session_state.y_train, cmap='viridis')
    ax[1].set_title("Feature Scatter")
    plt.colorbar(scatter)
    st.pyplot(fig)

from smote_processor import SMOTEProcessor

# Sidebar params
apply_smote = st.sidebar.checkbox("Apply SMOTE")  # Toggle
if apply_smote:
    ratio = st.sidebar.slider("Sampling Ratio", 0.5, 2.0, 1.0)  # F-3 adjustable
    k = st.sidebar.slider("K Neighbors", 1, 10, 5)

if st.sidebar.button("Process Data") and 'X_train' in st.session_state:
    X_train = st.session_state.X_train.copy()
    y_train = st.session_state.y_train.copy()
    if apply_smote:
        X_train, y_train = SMOTEProcessor.apply_smote(X_train, y_train, ratio, k)
    st.session_state.X_train_bal = X_train
    st.session_state.y_train_bal = y_train
    st.write(f"After SMOTE: {y_train.value_counts().to_dict()}")

from model_trainer import ModelTrainer

if st.button("Train & Evaluate") and 'X_train' in st.session_state:
    # Before SMOTE
    model_before = ModelTrainer.train(st.session_state.X_train, st.session_state.y_train)
    metrics_before, cm_before = ModelTrainer.evaluate(model_before, st.session_state.X_test, st.session_state.y_test)
    st.session_state.metrics_before = metrics_before

    # After SMOTE (if applied)
    if 'X_train_bal' in st.session_state:
        model_after = ModelTrainer.train(st.session_state.X_train_bal, st.session_state.y_train_bal)
        metrics_after, cm_after = ModelTrainer.evaluate(model_after, st.session_state.X_test, st.session_state.y_test)
        st.session_state.metrics_after = metrics_after

from visualizer import Visualizer

if 'metrics_before' in st.session_state:
    col1, col2 = st.columns(2)  # Side-by-side (F-6)
    with col1:
        st.subheader("Before SMOTE")
        st.dataframe(st.session_state.metrics_before)
        Visualizer.plot_confusion_matrix(st.session_state.cm_before)  # Assume stored
    with col2:
        if 'metrics_after' in st.session_state:
            st.subheader("After SMOTE")
            st.dataframe(st.session_state.metrics_after)
            Visualizer.plot_confusion_matrix(st.session_state.cm_after)
    Visualizer.plot_roc(model_before, st.session_state.X_test, st.session_state.y_test, "Before ROC")

st.sidebar.info("SMOTE balances minority class by synthetic samples. Adjust params to see impact.")
# Add more in sections
