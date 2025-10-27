from imblearn.datasets import fetch_datasets
import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler:
    @staticmethod
    def load_data(dataset_name):
        datasets = fetch_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not found in imblearn datasets.")
        data = datasets[dataset_name]
        n_features = data.data.shape[1]
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        X = pd.DataFrame(data.data, columns=feature_names)
        y = pd.Series(data.target)

        return X, y, y.value_counts().to_dict(), X.columns.tolist()

    @staticmethod
    def split_data(X,y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
        

datasets = fetch_datasets()
for name in list(datasets.keys())[:3]:
    data = datasets[name]
    print(f"\nDataset: {name}")
    print(f"  Shape: {data.data.shape}")
    print(f"  Has feature_names: {hasattr(data, 'feature_names')}")
    if hasattr(data, 'feature_names'):
        print(f"  Features: {data.feature_names}")
    else:
        print(f"  Features: Generic names (Feature_0, Feature_1, ...)")
        
