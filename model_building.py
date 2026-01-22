# Breast Cancer Prediction System - Model Development
# Algorithm: Support Vector Machine (SVM)
# Selected Features: radius_mean, texture_mean, area_mean, concavity_mean, symmetry_mean

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load the Breast Cancer Wisconsin dataset
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Map diagnosis: 0 = Malignant, 1 = Benign (as per sklearn convention)
print(f"\nDataset shape: {df.shape}")
print(f"Diagnosis distribution:\n{df['diagnosis'].value_counts()}")

# 2. Data Preprocessing

# 2.1 Check for missing values
print(f"\nMissing values:\n{df.isnull().sum().sum()}")

# 2.2 Feature Selection - Select 5 features
selected_features = [
    'mean radius',
    'mean texture', 
    'mean area',
    'mean concavity',
    'mean symmetry'
]

X = df[selected_features]
y = df['diagnosis']

print(f"\nSelected features: {selected_features}")
print(f"Features shape: {X.shape}")

# 2.3 Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 2.4 Feature Scaling (mandatory for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed using StandardScaler")

# 3. Implement and Train the Model - Support Vector Machine (SVM)
print("\n" + "="*50)
print("Training Support Vector Machine (SVM) Model...")
print("="*50)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

print("Model training completed!")

# 4. Model Evaluation
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

# Predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Malignant', 'Benign']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# 5. Save the model and scaler
print("\n" + "="*50)
print("Saving Model and Scaler")
print("="*50)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save using Joblib
joblib.dump(svm_model, 'model/breast_cancer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(selected_features, 'model/feature_names.pkl')

print("Model saved to: model/breast_cancer_model.pkl")
print("Scaler saved to: model/scaler.pkl")
print("Feature names saved to: model/feature_names.pkl")

# 6. Demonstrate model reloading and prediction
print("\n" + "="*50)
print("Demonstrating Model Reloading")
print("="*50)

# Load the saved model
loaded_model = joblib.load('model/breast_cancer_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
loaded_features = joblib.load('model/feature_names.pkl')

print("Model successfully loaded from disk!")

# Test prediction with sample data
sample_data = X_test.iloc[0:3]
print(f"\nSample input data:\n{sample_data}")

# Scale and predict
sample_scaled = loaded_scaler.transform(sample_data)
predictions = loaded_model.predict(sample_scaled)

print("\nPredictions:")
for i, pred in enumerate(predictions):
    diagnosis = "Benign" if pred == 1 else "Malignant"
    actual = "Benign" if y_test.iloc[i] == 1 else "Malignant"
    print(f"Sample {i+1}: Predicted = {diagnosis}, Actual = {actual}")

print("\n" + "="*50)
print("Model Development Complete!")
print("="*50)
print("\nNOTE: This system is strictly for educational purposes")
print("and must not be presented as a medical diagnostic tool.")