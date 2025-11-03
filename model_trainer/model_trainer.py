from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelTrainer:  # HLD: Model Trainer
    @staticmethod
    def train(X_train, y_train):
        """Train classifier (F-4)."""
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate(model, X_test, y_test):
        """Compute metrics (F-5). Returns dict."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        return metrics, confusion_matrix(y_test, y_pred)