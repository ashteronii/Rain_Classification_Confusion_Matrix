# Import necessary libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the cleaned weather forecast dataset.
weather = pd.read_csv("Weather_Cleaned.csv")

# Extract target variable ('Rain') and encode labels: 1 = 'rain', 0 = 'no rain'.
y = weather['Rain'].map({'rain': 1, 'no rain': 0}).copy().to_numpy()

# Extract features by dropping the target column (drop 'Rain').
X = weather.drop(columns=['Rain']).copy().to_numpy()

# Normalize the features (mean = 0, standard deviation = 1).
X -= np.average(X, axis=0)
X /= np.std(X, axis=0)

# Split the data into training and validation sets (70%/30% split).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define a grid of C values for hyperparameter tuning.
parameters = {"C": np.linspace(0.01, 0.2, num=20)}

# Initialize the classifier (Support Vector Classifier with RBF kernel).
clf = SVC(kernel='rbf')

# Perform a grid search with 5-fold cross-validation using R^2 score to find the optimal C.
grid_search = GridSearchCV(clf,param_grid=parameters,cv=5,scoring="r2")
grid_search.fit(X_train,y_train)

# Display the cross-validation results.
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[['param_C','mean_test_score','rank_test_score']])

# Extract the best C value from the grid search and use it to train the final model.
best_C = grid_search.best_params_['C']
print(f"Optimal value for C: {best_C}")
best_model = SVC(C=best_C, kernel='rbf')
best_model.fit(X_train, y_train)

# Evaluate the model on the test set.
test_score = best_model.score(X_test, y_test)
print(f"Test Score with optimal C: {test_score:.3f}")

# Plot the normalized confusion matrix.
cm = confusion_matrix(y_test, best_model.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)
disp_cm.plot()
plt.show()
