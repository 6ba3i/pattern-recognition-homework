import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # All four features
y = iris.target  # Three classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

# Step 2: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Logistic Regression using One-vs-Rest (OvR) - default
model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=200)
model_ovr.fit(X_train, y_train)
y_pred_ovr = model_ovr.predict(X_test)

# Step 4: Logistic Regression using Softmax (Multinomial)
model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model_softmax.fit(X_train, y_train)
y_pred_softmax = model_softmax.predict(X_test)

# Step 5: Evaluate both models
print("=== One-vs-Rest (OvR) Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ovr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ovr))
print("Classification Report:\n", classification_report(y_test, y_pred_ovr))

print("\n=== Softmax (Multinomial) Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_softmax))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_softmax))
print("Classification Report:\n", classification_report(y_test, y_pred_softmax))
