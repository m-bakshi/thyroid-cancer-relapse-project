import zipfile
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from ZIP file
with zipfile.ZipFile("thyroid_cancer.zip", "r") as z:
    with z.open("thyroid_cancer/dataset.csv") as f:
        data = pd.read_csv(f)

# Check column names 
print("Columns in dataset:")
print(data.columns.tolist())

# Handle missing values
data.ffill(inplace=True)  # forward-fill missing values

# Convert target column into binary values
data['Recurred'] = data['Recurred'].map({'Yes': 1, 'No': 0})

# Define categorical columns to encode
categorical_columns = [
    'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

#  One-hot encode categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Features and label
X = data_encoded.drop('Recurred', axis=1)
y = data_encoded['Recurred']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "thyroid_recurrence_model.joblib")

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance from the trained model
importances = model.feature_importances_
feature_names = X.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot top 10 important features
plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Features for Thyroid Cancer Recurrence Prediction")
plt.barh(range(10), importances[indices][:10][::-1], align='center', color='teal')
plt.yticks(range(10), feature_names[indices][:10][::-1])
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()