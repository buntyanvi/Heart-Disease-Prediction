# Heart Disease Prediction

A machine learning project to predict the presence of heart disease based on clinical parameters. This project uses the University of California Irvine's heart disease dataset to train and evaluate multiple machine learning models.

## Project Overview

The goal of this project is to accurately predict whether someone has heart disease based on their clinical data. This model could potentially assist healthcare professionals in early screening and diagnosis. Multiple machine learning algorithms (KNN, Logistic Regression, Random Forest, and XGBoost) are compared to find the most accurate predictive model.

### Project Structure

```
Heart-Disease-Prediction/
├── Heart Disease Prediction Original.ipynb    # Main project notebook with EDA and modeling
├── heart_disease_classifier_model             # Saved KNN model (best performer)
├── xgboost_heart_disease_model                # Saved XGBoost model (if it outperformed KNN)
├── heart-disease (1).csv                      # Dataset
├── heart_disease_prediction_web_app.py        # Web application for model deployment
├── compare_models.py                          # Script to compare model performance
├── test_xgboost.py                            # Script to test XGBoost performance
└── requirements.txt                           # Dependencies
```

## Dataset

The dataset comes from the University of California Irvine's heart disease dataset and includes the following features:

* **age**: The age of the patient
* **sex**: The gender of the patient (1 = male, 0 = female)
* **cp**: Type of chest pain (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
* **trestbps**: Resting blood pressure in mmHg
* **chol**: Serum Cholesterol in mg/dl
* **fbs**: Fasting Blood Sugar (1 = fasting blood sugar > 120mg/dl, 0 = otherwise)
* **restecg**: Resting ElectroCardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
* **thalach**: Maximum heart rate achieved
* **exang**: Exercise induced angina (1 = yes, 0 = no)
* **oldpeak**: ST depression induced by exercise relative to rest
* **slope**: Peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
* **ca**: Number of major vessels (0-3) colored by fluoroscopy
* **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
* **target**: Diagnosis of heart disease (0 = absence, 1 = presence)

## Models Evaluated

The project evaluated several machine learning models:

1. Logistic Regression - A linear approach for binary classification
2. K-Nearest Neighbors (KNN) - A non-parametric method using proximity to make classifications
3. Random Forest - An ensemble learning method using multiple decision trees
4. XGBoost - A gradient boosting framework known for its performance and efficiency

All models were evaluated using cross-validation, and their hyperparameters were optimized using RandomizedSearchCV and GridSearchCV. The project automatically selects the best-performing model between KNN and XGBoost for the final prediction system.

## Key Findings

- The KNN model with n_neighbors=7 provided the highest accuracy (~92%)
- Feature importance analysis revealed the most significant predictors of heart disease
- Model performance was evaluated using accuracy, precision, recall, F1 score, and ROC/AUC

## Web Application

The project includes a web application (`heart_disease_prediction_web_app.py`) that allows users to input patient data and receive a prediction about the likelihood of heart disease.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: If you encounter issues with XGBoost installation, you can install it separately:
   ```bash
   pip install xgboost==1.7.5
   ```

3. Ensure you have the dataset file `heart-disease (1).csv` in the project directory.

## Usage

## Requirements

The project requires the following main dependencies:
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Pickle

A complete list of dependencies is available in `requirements.txt`.

## Model Performance

The best-performing model (KNN with n_neighbors=7) achieved:
- Accuracy: 92%
- Precision: 92%
- Recall: 92%
- F1 Score: 92%

## Future Work

- Explore additional feature engineering approaches
- Implement a more sophisticated web interface
- Deploy the model as a RESTful API
- Collect and incorporate additional patient data
- Explore deep learning approaches

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The scikit-learn and XGBoost teams for their excellent libraries
