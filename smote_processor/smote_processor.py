from imblearn.over_sampling import SMOTE


class SMOTEProcessor:
    @staticmethod
    def apply_smote(X_train, y_train, k_neighbors, sampling_strategy):
        smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    