import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
y_bin = label_binarize(y, classes=np.unique(y))  # Needed for ROC
n_classes = y_bin.shape[1]

# Models
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=200),
    "Naive Bayes": GaussianNB()
}

# Classification Reports and F1 Scores
print("=== Classification Reports (no OneVsRest) ===")
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average='macro')
    print(f"\n{name}:\nF1 Score (macro): {f1:.2f}")
    print(classification_report(y, y_pred, target_names=iris.target_names))

# ROC Curve Comparison
plt.figure(figsize=(10, 6))
for name, model in models.items():
    model.fit(X, y)
    y_score = model.predict_proba(X)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (Class {i}) AUC = {roc_auc:.2f}')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (All Samples Used)')
plt.legend(loc='lower right')
plt.grid()
plt.show()
