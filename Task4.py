import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (Iris dataset from Scikit-Learn as an example)
from sklearn.datasets import load_iris

data = load_iris()

# Convert to DataFrame
iris_df = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_df['target'] = data.target

# Features and target
X = iris_df[data.feature_names]
y = iris_df['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(model.feature_importances_, index=data.feature_names)
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()

# Save the results to a file
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Classification Report": classification_report(y_test, y_pred, output_dict=True)
}
pd.DataFrame(results).to_csv("model_results.csv", index=False)

print("Task completed! The model has been trained and evaluated. Results are saved in 'model_results.csv'.")