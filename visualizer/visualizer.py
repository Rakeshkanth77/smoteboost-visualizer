import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import streamlit as st

class Visualizer:  # HLD: Visualization Engine
    @staticmethod
    def plot_distribution(y_train, title="Class Distribution"):
        fig, ax = plt.subplots()
        y_train.value_counts().plot(kind='bar', ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    @staticmethod
    def plot_roc(model, X_test, y_test, title="ROC Curve"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_title(title)
        ax.legend()
        st.pyplot(fig)

    @staticmethod
    def plot_confusion_matrix(cm, title="Confusion Matrix"):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        st.pyplot(plt.gcf())