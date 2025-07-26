
#Import necessary libraries
import pandas as pd                     # For data handling
from sklearn.model_selection import train_test_split  # To split dataset into train and test sets
from sklearn.linear_model import LogisticRegression   # Machine learning model
from sklearn.metrics import classification_report, confusion_matrix  # To evaluate model
from imblearn.over_sampling import SMOTE              # To handle imbalanced dataset
import seaborn as sns                                 # For visualization
import matplotlib.pyplot as plt                       # For plotting graphs

# Load the dataset
# Make sure creditcard.csv is in the same folder as this script
# You can download it from: https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv("creditcard.csv")

# Display first few rows of the data to understand its structure
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values or data issues
print("\nDataset info:")
print(data.info())

# Prepare features (X) and labels (y)
# 'Class' column is the target: 0 = Not Fraud, 1 = Fraud
X = data.drop('Class', axis=1)  # Features: all columns except 'Class'
y = data['Class']               # Target: 'Class' column

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Handle imbalanced data using SMOTE (Synthetic Minority Oversampling Technique)
# Fraud cases are very few, so we oversample the minority class to balance the dataset
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nBalanced dataset after SMOTE:")
print("Fraud cases:", sum(y_train_res == 1))
print("Non-Fraud cases:", sum(y_train_res == 0))

# Train the Logistic Regression model
# Logistic Regression is good for binary classification (fraud/not fraud)
model = LogisticRegression(max_iter=1000)  # max_iter ensures the model has enough iterations to converge
model.fit(X_train_res, y_train_res)        # Train the model on the balanced dataset

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
# Confusion Matrix shows correct vs incorrect predictions
# Classification Report shows precision, recall, and F1-score
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

