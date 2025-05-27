import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
df = pd.read_csv("heart-disease (1).csv")

# Prepare data
X = df.drop("target", axis=1)
y = df["target"]

# Split data
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_sc, y_train)
knn_preds = knn.predict(X_test_sc)
knn_accuracy = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# Train XGBoost model with default parameters
xgb_default = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_default.fit(X_train_sc, y_train)
xgb_default_preds = xgb_default.predict(X_test_sc)
xgb_default_accuracy = accuracy_score(y_test, xgb_default_preds)
print(f"XGBoost (Default) Accuracy: {xgb_default_accuracy:.4f}")

# Train XGBoost model with tuned parameters
xgb_tuned = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1
)
xgb_tuned.fit(X_train_sc, y_train)
xgb_tuned_preds = xgb_tuned.predict(X_test_sc)
xgb_tuned_accuracy = accuracy_score(y_test, xgb_tuned_preds)
print(f"XGBoost (Tuned) Accuracy: {xgb_tuned_accuracy:.4f}")

# Compare results
print("\nAccuracy Comparison:")
print(f"KNN:                {knn_accuracy:.4f}")
print(f"XGBoost (Default):  {xgb_default_accuracy:.4f}")
print(f"XGBoost (Tuned):    {xgb_tuned_accuracy:.4f}")

# Find the best model
if xgb_tuned_accuracy > max(knn_accuracy, xgb_default_accuracy):
    print("\nXGBoost with tuned parameters performs best")
    print("\nXGBoost (Tuned) Classification Report:")
    print(classification_report(y_test, xgb_tuned_preds))
elif xgb_default_accuracy > knn_accuracy:
    print("\nXGBoost with default parameters performs best")
    print("\nXGBoost (Default) Classification Report:")
    print(classification_report(y_test, xgb_default_preds))
else:
    print("\nKNN performs best")
    print("\nKNN Classification Report:")
    print(classification_report(y_test, knn_preds))
