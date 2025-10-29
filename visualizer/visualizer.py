
class Visualizer:
    @staticmethod
    def plot_distribution(y_train, y_test):
        import matplotlib.pyplot as plt
        from collections import Counter

        counter = Counter(y_train)
        classes = list(counter.keys())
        counts = list(counter.values())

        plt.figure(figsize=(4, 2.5))
        plt.bar(classes, counts, color='skyblue', edgecolor='black')
        plt.xlabel("Class", fontsize=8)
        plt.ylabel("Count", fontsize=8)
        plt.title("Class Distribution", fontsize=9, fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
        

    @staticmethod
    def plot_roc_curve(modelm, x_test, y_test):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        y_scores = modelm.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(4, 2.5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=8)
        plt.ylabel('True Positive Rate', fontsize=8)
        plt.title('Receiver Operating Characteristic', fontsize=9, fontweight='bold')
        plt.legend(loc="lower right", fontsize=7)
        plt.tight_layout()
        return plt.gcf()