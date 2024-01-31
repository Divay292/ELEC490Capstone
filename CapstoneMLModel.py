# Import necessary libraries
import numpy as np
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

argParser = argparse.ArgumentParser()
# training options
argParser.add_argument('-m', type=str, help='Mode')
argParser.add_argument('-s', type=str, help='Weight File')
argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')
args = argParser.parse_args()


def save_weights(model, filename):
    joblib.dump(model, filename)


def main():
    # pd.set_option('display.max_rows', None)
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df2 = pd.read_csv('test.csv')
    gender_mapping = {'Male': 0, 'Female': 1}
    # bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}

    '''df['Gender'] = df['Gender'].map(gender_mapping)
    df['BMI Category'] = pd.factorize(df['BMI Category'])[0]
    df['Quality of Sleep'] = df['Quality of Sleep']

    df2['Gender'] = df2['Gender'].map(gender_mapping)
    df2['BMI Category'] = pd.factorize(df2['BMI Category'])[0]
    df2['Quality of Sleep'] = df2['Quality of Sleep'].astype(int)'''

    # print(df2)
    # print(df.dtypes)

    # Data preprocessing
    x_train = df.iloc[:, 1: 8]
    y_train = df.iloc[:, 8:9]
    x_test = df2.iloc[:, 1: 8]
    y_test = df2.iloc[:, 8:9]

    # Split the dataset into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)


    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=6)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    scaler = StandardScaler()
    x_train_selected_scaled = scaler.fit_transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)

    # Define the parameter grid for Logistic Regression
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                     'max_iter': [100, 500, 1000]}
    # Logistic Regression Model
    lr_model = LogisticRegression(C= 0.001, max_iter= 100)     # C is the regularization
    '''grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5)
    grid_search_lr.fit(x_train_selected_scaled, y_train)'''
    # print("Best hyperparameters for Logistic Regression:", grid_search_lr.best_params_)
    lr_model.fit(x_train_selected_scaled, y_train)
    lr_accuracy_train = lr_model.score(x_train_selected_scaled, y_train)
    lr_accuracy_test = lr_model.score(x_test_selected_scaled, y_test)
    # Save LR model weight
    #save_weights(lr_model, 'lr_model_weights.pth')

    # Feature selection using Recursive Feature Elimination (RFE)
    estimator = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion='gini',
                                       max_features=None)
    rfe_selector = RFE(estimator, n_features_to_select=6)
    x_train_selected_rfe = rfe_selector.fit_transform(x_train, y_train)
    x_test_selected_rfe = rfe_selector.transform(x_test)

    # Define the parameter grid for Decision Tree
    param_grid_dt = {'max_depth': [3, 5, 7, None],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]}
    # Decision Tree Model
    # min_samples_split and min_samples_leaf control structure of tree
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, min_samples_split=10)
    '''grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    grid_search_dt.fit(x_train_selected_rfe, y_train)
    print("Best hyperparameters for Decision Tree:", grid_search_dt.best_params_)'''

    dt_model.fit(x_train_selected_rfe, y_train)
    dt_accuracy_train = dt_model.score(x_train_selected_rfe, y_train)
    dt_accuracy_test = dt_model.score(x_test_selected_rfe, y_test)
    #  Save DT model
    save_weights(dt_model, 'dt_model_weights.pth')

    # Define the parameter grid for Random Forest
    param_grid_rf = {'n_estimators': [0, 100, 200],
                     'max_depth': [None, 10, 20],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]}
    # Random Forest Model
    rf_model = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=50)
    '''grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5)
    grid_search_rf.fit(x_train_selected_rfe, y_train)
    print("Best hyperparameters for Random Forest:", grid_search_rf.best_params_)'''

    rf_model.fit(x_train_selected_rfe, y_train)
    rf_accuracy_train = rf_model.score(x_train_selected_rfe, y_train)
    rf_accuracy_test = rf_model.score(x_test_selected_rfe, y_test)
    # Save RF model
    save_weights(rf_model, 'rf_model_weights.pth')

    param_grid_svm = {'C': [0.1, 1, 10],
                      'kernel': ['linear', 'rbf', 'poly'],
                      'gamma': ['scale', 'auto']}
    # Support Vector Machines (SVM) Model
    # C parameter controls the trade-off between having a smooth decision
    # boundary and classifying the training points correctly
    # Kernel parameter determines the type of kernel used (linear, poly, radial basis function(rbf), sigmoid)
    svm_model = SVC()
    '''grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5)
    grid_search_svm.fit(x_train_selected_rfe, y_train)
    print("Best hyperparameters for SVM:", grid_search_svm.best_params_)'''

    svm_model.fit(x_train_selected_rfe, y_train)
    svm_accuracy_train = svm_model.score(x_train_selected_rfe, y_train)
    svm_accuracy_test = svm_model.score(x_test_selected_rfe, y_test)

    # Displaying model accuracies
    print("Logistic Regression Training Accuracy:", lr_accuracy_train)
    print("Logistic Regression Test Accuracy:", lr_accuracy_test)

    print("Decision Tree Training Accuracy:", dt_accuracy_train)
    print("Decision Tree Test Accuracy:", dt_accuracy_test)

    print("Random Forest Training Accuracy:", rf_accuracy_train)
    print("Random Forest Test Accuracy:", rf_accuracy_test)

    print("SVM Training Accuracy:", svm_accuracy_train)
    print("SVM Test Accuracy:", svm_accuracy_test)

    '''if lr_accuracy_test < 0.8 or dt_accuracy_test < 0.8 or rf_accuracy_test < 0.8 or svm_accuracy_test < 0.8:
        # Transfer the first line of test.csv to Sleep_health_and_lifestyle_dataset.csv
        first_line_test = df2.iloc[0:1, :]
        df = pd.concat([df, first_line_test], ignore_index=True)

        # Delete the first line from test.csv
        df2 = df2.drop(0)

        df.to_csv('Sleep_health_and_lifestyle_dataset_modified.csv', index=False)
        df.to_csv('Sleep_health_and_lifestyle_dataset.csv', index=False)
        df2.to_csv('test.csv', index=False)
        df2.to_csv('test_modified.csv', index=False)'''


if __name__ == "__main__":
    main()

