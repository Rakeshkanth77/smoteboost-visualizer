import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class DataHandler:  # HLD: Data Handler
    @staticmethod
    def load_data(dataset_name):
        """Load imbalanced dataset (F-1). Returns X, y as DataFrames."""
        if dataset_name == "Imbalanced Binary (make_classification)":
            X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                      n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42)  # Imbalanced
        else:
            raise ValueError("Invalid dataset")
        X = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        y = pd.Series(y)
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2):
        """Split 80/20 (LLD)."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)