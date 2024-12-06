# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing and Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Flask API for Deployment
from flask import Flask, request, jsonify
# Load dataset
data = pd.read_csv('creditcard.csv')  # Make sure the dataset is in the working directory

# Check class distribution
print(data['Class'].value_counts())

# Scale the 'Amount' column
scaler = MinMaxScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Separate features and target variable
X = data.drop(['Class', 'Time'], axis=1)  # Drop unnecessary columns
y = data['Class']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# AUC-ROC Score
auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC Score: {auc_score}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Visualize class distribution
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Correlation heatmap
corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# Fraud vs. Non-Fraud comparison
fraud_data = data[data['Class'] == 1]
non_fraud_data = data[data['Class'] == 0]

plt.figure(figsize=(8, 4))
sns.histplot(fraud_data['Amount'], kde=True, color='red', label='Fraud')
sns.histplot(non_fraud_data['Amount'], kde=True, color='blue', label='Non-Fraud')
plt.legend()
plt.title('Fraud vs. Non-Fraud Transaction Amounts')
plt.show()
import joblib

# Save the trained model
joblib.dump(model, 'fraud_detection_model.pkl')

# Load the model later
model = joblib.load('fraud_detection_model.pkl')
