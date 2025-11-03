from imblearn.over_sampling import SMOTE
import pandas as pd

class SMOTEProcessor:  # HLD: SMOTE Processor
    @staticmethod
    def apply_smote(X_train, y_train, sampling_strategy=1.0, k_neighbors=5):
        """Apply SMOTE (F-3). Returns balanced X_train, y_train."""
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal)