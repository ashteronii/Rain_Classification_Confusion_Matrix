# Weather Forecast Rain Classification using Support Vector Classifier (SVC)

This project uses a Support Vector Classifier (SVC) with an RBF kernel to predict whether it will rain based on historical weather data. The workflow includes data preprocessing, normalization, hyperparameter tuning, and performance evaluation using a confusion matrix.

---

## Dataset

- Zeeshan Ullah (2023). *Weather Forecast Dataset*. Kaggle. https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset  
- The dataset contains daily weather observations including temperature, humidity, wind speed, cloud cover, and pressure.
- Target labels include: `rain` and `no rain`.

---

## Project Structure

- `weather_forecast_data.csv`: Raw dataset from Kaggle.
- `Weather_Cleaned.csv`: Cleaned dataset used for training.
- `clean_data.py`: Data cleaning script.
- `svc_model.py`: Model training and evaluation script.
- `README.md`: This file.

---

## Model Overview

The classification model is a Support Vector Machine (SVC) using a radial basis function (RBF) kernel.  
The pipeline includes:

- **Feature Set**: Temperature, humidity, wind speed, cloud cover, and pressure.
- **Target Variable**: Binary label indicating rain occurrence.
- **Preprocessing**: Standardization (mean = 0, std = 1).
- **Hyperparameter Tuning**: Grid search over 20 values of C (0.01 to 0.2) using 5-fold cross-validation.
- **Evaluation Metric**: R² score during validation; accuracy and normalized confusion matrix on the test set.

---

## Results

The model was trained on a 70% training split and evaluated on a 30% test split.  
**Performance Metrics:**
- Test accuracy: varies by run (~85% typical)
- Normalized confusion matrix visualized with `matplotlib`.

---

## How to Run

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/yourusername/weather-forecast-svc.git
   cd weather-forecast-svc

---

## Dependencies
Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
