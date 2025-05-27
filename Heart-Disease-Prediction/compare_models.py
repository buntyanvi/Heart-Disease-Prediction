import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
print("Loading data...")
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
print("Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_sc, y_train)
knn_preds = knn.predict(X_test_sc)
knn_accuracy = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# Train XGBoost model
print("Training XGBoost model...")
xgb = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0
)
xgb.fit(X_train_sc, y_train)
xgb_preds = xgb.predict(X_test_sc)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Compare results
print("\nAccuracy Comparison:")
print(f"KNN:     {knn_accuracy:.4f}")
print(f"XGBoost: {xgb_accuracy:.4f}")
print(f"Difference: {(xgb_accuracy - knn_accuracy) * 100:.2f}%")

if xgb_accuracy > knn_accuracy:
    print("\nXGBoost performs better than KNN")
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_preds))
else:
    print("\nKNN performs better than or equal to XGBoost")
    print("\nKNN Classification Report:")
    print(classification_report(y_test, knn_preds))
