Weather Forecast Rain Classification with SVC
This project applies a Support Vector Classifier (SVC) to predict whether it will rain based on historical weather data. It includes data cleaning, normalization, hyperparameter tuning, and model evaluation using a confusion matrix.

Dataset
Source: Kaggle - Weather Forecast Dataset

The dataset includes features such as Temperature, Humidity, Wind Speed, Cloud Cover, and Pressure.

Target variable: Rain (binary classification: 'rain' or 'no rain').

Project Structure
weather_forecast_data.csv: Original dataset from Kaggle.

Weather_Cleaned.csv: Cleaned dataset used for training.

clean_data.py: Script for cleaning and saving the dataset.

svc_model.py: Main training and evaluation script.

README.md: This file.

Model Overview
Model: Support Vector Classifier (RBF kernel)

Feature preprocessing: Standardization (mean = 0, std = 1)

Hyperparameter tuning: Grid search over 20 values of C (0.01 to 0.2)

Evaluation: Accuracy and normalized confusion matrix

Results
The model was trained on 70% of the data and tested on the remaining 30%.

Hyperparameter tuning used 5-fold cross-validation with R² scoring.

Final performance may vary by run; refer to terminal output for optimal C and test accuracy.
