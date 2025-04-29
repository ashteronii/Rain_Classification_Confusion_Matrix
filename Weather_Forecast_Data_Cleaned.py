import pandas as pd

# Load the Weather Forecast dataset.
# Source: https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset
weather = pd.read_csv("weather_forecast_data.csv")

# Remove any rows with missing values and drop duplicate rows to ensure data quality.
weather = weather.dropna().drop_duplicates()

# Optional: Inspect the dataset's structure and distribution of the target variable.
# print(weather.info())
# print(weather.describe())
# print(weather["Rain"].value_counts())

# Save the cleaned dataset for use in modeling.
weather.to_csv("Weather_Cleaned.csv")