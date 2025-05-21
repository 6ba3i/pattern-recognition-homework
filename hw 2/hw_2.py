import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 2: Binary classification — Setosa vs Others
# Setosa = 1, others = 0
y_binary = (y == 0).astype(int)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Step 4: Train Naïve Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict probabilities for ROC
y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (Setosa)

# Step 6: Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Step 7: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Naïve Bayes ROC Curve (Setosa vs Others)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Optional Step 8: Cross-validation accuracy
cv_scores = cross_val_score(model, X, y_binary, cv=5)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average Accuracy: {np.mean(cv_scores):.2f}")