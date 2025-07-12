# Thyroid Cancer Recurrence Prediction

This project uses machine learning (Random Forest Classifier) to predict the likelihood of thyroid cancer recurrence/relapse based on patient data. The dataset includes clinical, pathological and demographic features, and the model provides insights into which features most impact recurrence.


## Dataset

File: thyroid_cancer.zip, contains dataset.csv

The dataset includes information such as gender, smoking history, pathology, tumor stage (T,N,M) and treatment response.
Target variable: Recurred (Yes/No) - mapped to binary (1/0)


## Libraries Used

pandas

numpy

scikit-learn

matplotlib

joblib

zipfile


## Project Workflow

- Data Extraction: Reads the compressed CSV file from a zip archive.

- Preprocessing: Handles missing values using forward fill. Encodes categorical variables using one-hot encoding and scales numerical features using StandardScaler.

- Model Building: Splits data into training and test sets. Trains a RandomForestClassifier to predict recurrence.

- Evaluation: Evaluates the model using accuracy, confusion matrix and classification report.

- Model Export: Saves the trained model as a .joblib file for future use.

- Feature Importance Visualization: Visualizes the top 10 most important features in predicting recurrence.


## Results

Model Evaluation:
Accuracy: 0.987012987012987

Confusion Matrix:
[[58  0]
 [ 1 18]]

Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99        58
           1       1.00      0.95      0.97        19

    accuracy                           0.99        77
   macro avg       0.99      0.97      0.98        77
weighted avg       0.99      0.99      0.99        77

<img width="1000" height="600" alt="Top_10_features" src="https://github.com/user-attachments/assets/c217a277-6230-4660-8d2e-c0da350ee6ac" />
