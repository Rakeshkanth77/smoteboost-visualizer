from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    @staticmethod
    def train(X_train, y_train, model):
        model.fit(X_train, y_train)
        return model
    
    @staticmethod
    def evaluate(X_test, y_test, model):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }

        if y_pred_proba is None:
             metrics["roc_auc"] = roc_auc_score(y_test, y_pred)

        return metrics, y_pred