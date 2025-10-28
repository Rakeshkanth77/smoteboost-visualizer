import streamlit as st
import pandas as pd
from data_handler.data_handler import DataHandler
from imblearn.datasets import fetch_datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.title("Welcome to SmoteBoost Visualizer")

st.sidebar.header("Selection")
all_datasets = list(fetch_datasets().keys())
dataset = st.sidebar.selectbox("Choose Dataset", all_datasets)

@st.cache_data
def load_data(dataset):
    X, y, class_distribution, feature_names = DataHandler.load_data(dataset)
    return X, y, class_distribution, feature_names


# Only load and display if button is clicked
if st.sidebar.button("Load Data"):
    X, y, class_distribution, feature_names = load_data(dataset)
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['dataset'] = dataset
    st.success(f"✓ Dataset Loaded: {dataset}")


if 'X' in st.session_state:
    X = st.session_state['X']
    y = st.session_state['y']
    dataset = st.session_state['dataset']
    class_distribution = y.value_counts().to_dict()
    feature_names = X.columns.tolist()
    
    st.subheader("Dataset Overview")
    st.write(f"Selected Dataset: **{dataset}**")

    # Plot: Class Distribution Bar Plot
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    ax1.bar(classes, counts, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Class", fontsize=8)
    ax1.set_ylabel("Count", fontsize=8)
    ax1.set_title("Class Distribution", fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

    # Plot: Feature Scatter Plot
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    scatter = ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolor='black', s=30)
    ax2.set_xlabel(feature_names[0], fontsize=8)
    ax2.set_ylabel(feature_names[1], fontsize=8)
    ax2.set_title("Feature Scatter Plot", fontsize=9, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label="Class")
    plt.tight_layout()
    st.pyplot(fig2)

    def split_data(X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    X_train, X_test, y_train, y_test = split_data(X, y)
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.success(f"✓ Data Split: {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")


smote_param = st.sidebar.slider("SMOTE k_neighbors", 1, 10, 5)
smote_strategy = st.sidebar.selectbox("SMOTE Strategy", ["minority", "not majority", "all", "auto"])

if "X_train" in st.session_state:
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']

    if st.sidebar.button("Apply SMOTE"):
        from smote_processor.smote_processor import SMOTEProcessor

        X_resampled, y_resampled = SMOTEProcessor.apply_smote(
            X_train, y_train, k_neighbors=smote_param,
            sampling_strategy=smote_strategy, 
        )

        st.session_state['X_resampled'] = X_resampled
        st.session_state['y_resampled'] = y_resampled
        st.success(f"✓ SMOTE Applied: Resampled to {X_resampled.shape[0]} samples")

    if 'y_resampled' in st.session_state:
        st.subheader("Class Distribution: Before vs After SMOTE")
        
        col1, col2 = st.columns(2)
        
        # Before SMOTE
        with col1:
            fig_before, ax_before = plt.subplots(figsize=(4, 2.5))
            y_train_counts = y_train.value_counts().sort_index()
            ax_before.bar(y_train_counts.index, y_train_counts.values, color='lightcoral', edgecolor='black')
            ax_before.set_xlabel("Class", fontsize=8)
            ax_before.set_ylabel("Count", fontsize=8)
            ax_before.set_title("Before SMOTE", fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_before)
        
        # After SMOTE
        with col2:
            fig_after, ax_after = plt.subplots(figsize=(4, 2.5))
            y_resampled_counts = st.session_state['y_resampled'].value_counts().sort_index()
            ax_after.bar(y_resampled_counts.index, y_resampled_counts.values, color='lightgreen', edgecolor='black')
            ax_after.set_xlabel("Class", fontsize=8)
            ax_after.set_ylabel("Count", fontsize=8)
            ax_after.set_title("After SMOTE", fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_after)


