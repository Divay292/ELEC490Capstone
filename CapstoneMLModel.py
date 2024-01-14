# Import necessary libraries
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, models, transforms
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler

argParser = argparse.ArgumentParser()
# training options
argParser.add_argument('-e', type=int, help='Epochs')
argParser.add_argument('-b', type=int, help='Batch Size')
argParser.add_argument('-m', type=str, help='Mode')
argParser.add_argument('-s', type=str, help='Weight File')
argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')
args = argParser.parse_args()
lr = 1e-3


def main():
    # pd.set_option('display.max_rows', None)
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    gender_mapping = {'Male': 0, 'Female': 1}
    bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['BMI Category'] = pd.factorize(df['BMI Category'])[0]
    # print(df.dtypes)
    x = df.iloc[:, 0: 8]
    y = df.iloc[:, 8:9]
    # Data preprocessing
    '''x = df[['Age', 'Stress Level', 'BMI Category', 'Heart rate (bpm)',
            'Daily Steps', 'Time asleep']]  # Features
    y = df['Sleep Quality']  # Target variable'''

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
    x_train = x_train.sort_values(by='Person ID')
    x_test = x_test.sort_values(by='Person ID')

    ''' print('x_train: ', x_train)
    print('y_train: ', y_train)
    print('x_test: ', x_test)
    print('y_test: ', y_test)'''

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=6)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    scaler = StandardScaler()
    x_train_selected_scaled = scaler.fit_transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)

    print('Before flattening - y_train shape:', y_train.shape)
    print('Before flattening - y_test shape:', y_test.shape)
    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)
    print('After flattening - y_train shape:', y_train.shape)
    print('After flattening - y_test shape:', y_test.shape)

    # Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(x_train_selected_scaled, y_train)
    lr_accuracy = lr_model.score(x_test_selected_scaled, y_test)

    # Feature selection using Recursive Feature Elimination (RFE)
    estimator = DecisionTreeClassifier()
    rfe_selector = RFE(estimator, n_features_to_select=6)
    x_train_selected_rfe = rfe_selector.fit_transform(x_train, y_train)
    x_test_selected_rfe = rfe_selector.transform(x_test)

    # Decision Tree Model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train_selected_rfe, y_train)
    dt_accuracy = dt_model.score(x_test_selected_rfe, y_test)

    # Random Forest Model
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train_selected_rfe, y_train)
    rf_accuracy = rf_model.score(x_test_selected_rfe, y_test)

    # Support Vector Machines (SVM) Model
    svm_model = SVC()
    svm_model.fit(x_train_selected_rfe, y_train)
    svm_accuracy = svm_model.score(x_test_selected_rfe, y_test)

    # Displaying model accuracies
    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Decision Tree Accuracy:", dt_accuracy)
    print("Random Forest Accuracy:", rf_accuracy)
    print("SVM Accuracy:", svm_accuracy)

    '''
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    plt.scatter(x_train, y_test, color='black', label='Actual')
    plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Predicted')
    plt.xlabel('Sleep Features')
    plt.ylabel('Sleep Quality')
    plt.legend()
    plt.show()'''


if __name__ == "__main__":
    main()
