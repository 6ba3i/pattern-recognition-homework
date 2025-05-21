import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# Step 1: Load Data and Select Feature
iris = datasets.load_iris()
X = iris.data[:, 0]  # Sepal length (cm)
y = (iris.target == 0).astype(int)  # Binary label: 1 if Setosa, else 0

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Apply Threshold Classification
threshold = 5.0
y_pred_train = (X_train > threshold).astype(int)
y_pred_test = (X_test > threshold).astype(int)

# Accuracy Calculation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 2: Compute ROC Curve using raw feature values as score
fpr, tpr, thresholds = roc_curve(y_test, X_test)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Sepal Length Threshold Classification')
plt.legend(loc="lower right")
plt.grid()
plt.show()