import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.data_loader import load_data, preprocess
from src.model import train_random_forest, train_svc, train_svc_tuned
from src.evaluate import evaluate_model, plot_confusion_matrix


def explore_data(wine_df):
      """EDA - visualise features vs quality."""
      features = ['fixed acidity', 'volatile acidity', 'citric acid',
                  'residual sugar', 'sulphates', 'alcohol']
      for feature in features:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='quality', y=feature, data=wine_df)
                plt.show()

      print(wine_df['quality'].value_counts())
      fig = plt.figure(figsize=(10, 6))
      sns.countplot(wine_df['quality'])
      plt.show()


def tune_svc(x_train, y_train):
      """GridSearchCV to find best SVC parameters."""
      param_grid = {'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
                    'kernel': ['linear', 'rbf'],
                    'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]}
      grid_svc = GridSearchCV(estimator=train_svc(x_train, y_train).__class__(),
                              param_grid=param_grid,
                              scoring='accuracy')
      grid_svc.fit(x_train, y_train)
      print("Best parameters for SVC:", grid_svc.best_params_)
      return grid_svc.best_params_


def main():
      # Load data
      wine_df = load_data('winequality-red.csv')

    # Exploratory data analysis
      explore_data(wine_df)

    # Preprocess
      x_train, x_test, y_train, y_test = preprocess(wine_df)

    # --- Random Forest Classifier ---
      print("\n2. RANDOM FOREST CLASSIFIER")
      rf_model = train_random_forest(x_train, y_train)
      y_pred_rf = evaluate_model(rf_model, x_test, y_test, "Random Forest")
      plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix-Random Forest Classifier")

    rf_eval = cross_val_score(estimator=rf_model, X=x_train, y=y_train, cv=10)
    print("RF Cross-Val Mean Accuracy:", rf_eval.mean())

    # --- Support Vector Classifier ---
    print("\nSupport Vector Classifier(SVC)")
    svc_model = train_svc(x_train, y_train)
    y_pred_svc = evaluate_model(svc_model, x_test, y_test, "SVC")
    plot_confusion_matrix(y_test, y_pred_svc, "Confusion Matrix-Support Vector Classifier")

    # --- Tuned SVC ---
    svc_tuned = train_svc_tuned(x_train, y_train, C=1.2, gamma=0.9, kernel='rbf')
    y_pred_svc2 = evaluate_model(svc_tuned, x_test, y_test, "SVC (Tuned)")
    plot_confusion_matrix(y_test, y_pred_svc2, "Confusion Matrix-Support Vector Classifier(improved)")


if __name__ == "__main__":
      main()
  
