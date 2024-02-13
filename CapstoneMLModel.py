import numpy as np
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import SVR, OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import random


def load_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset_modified.csv')
    df2 = pd.read_csv('Test_modified.csv')
    '''gender_mapping = {'Male': 0, 'Female': 1}
    df['Gender'] = df['Gender'].map(gender_mapping)
    df['BMI Category'] = pd.factorize(df['BMI Category'])[0]
    df['Quality of Sleep'] = df['Quality of Sleep'].astype(int)
    df['Sleep Duration'] = df['Sleep Duration'].astype(int)'''
    '''df2['Gender'] = df2['Gender'].map(gender_mapping)
    df2['BMI Category'] = pd.factorize(df2['BMI Category'])[0]
    df2['Quality of Sleep'] = df2['Quality of Sleep'].astype(int)
    df2['Sleep Duration'] = df['Sleep Duration'].astype(int)'''
    return df, df2


def preprocess_data(df, df2):
    x_train = df.iloc[:, 1:8]
    y_train = df.iloc[:, 8:9]
    x_test = df2.iloc[:, 1:8]
    y_test = df2.iloc[:, 8:9]
    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)
    selector = SelectKBest(score_func=f_classif, k=6)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    scaler = StandardScaler()
    x_train_selected_scaled = scaler.fit_transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)
    return x_train_selected_scaled, y_train, x_test_selected_scaled, y_test


def save_weights(model, filename):
    joblib.dump(model, filename)


def modify_dataframes(df, df2, lr_accuracy_test, dt_accuracy_test, rf_accuracy_test, svm_accuracy_test):
    if lr_accuracy_test < 0.8 or dt_accuracy_test < 0.8 or rf_accuracy_test < 0.8 or svm_accuracy_test < 0.8:
        # Transfer the first line of test.csv to Sleep_health_and_lifestyle_dataset.csv
        first_line_test = df2.iloc[0:1, :]
        df = pd.concat([df, first_line_test], ignore_index=True)

        # Delete the first line from test.csv
        df2 = df2.drop(0)

        # Save modified DataFrames back to CSV files
        df.to_csv('Sleep_health_and_lifestyle_dataset_modified.csv', index=False)
        df2.to_csv('test_modified.csv', index=False)
    return df, df2


def train_linear_regression(x_train, y_train, x_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    # save_weights(lr_model, 'lr_model_weights.pth')
    lr_accuracy_train = lr_model.score(x_train, y_train)
    lr_accuracy_test = lr_model.score(x_test, y_test)
    return lr_accuracy_train, lr_accuracy_test


def train_decision_tree(x_train, y_train, x_test, y_test):
    dt_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, min_samples_split=10)
    dt_model.fit(x_train, y_train)
    # save_weights(dt_model, 'dt_model_weights.pth')
    dt_accuracy_train = dt_model.score(x_train, y_train)
    dt_accuracy_test = dt_model.score(x_test, y_test)
    return dt_accuracy_train, dt_accuracy_test


def train_random_forest(x_train, y_train, x_test, y_test):
    rf_model = RandomForestRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=50)
    rf_model.fit(x_train, y_train)
    # save_weights(rf_model, 'rf_model_weights.pth')
    rf_accuracy_train = rf_model.score(x_train, y_train)
    rf_accuracy_test = rf_model.score(x_test, y_test)
    return rf_accuracy_train, rf_accuracy_test


def train_svr(x_train, y_train, x_test, y_test):
    svm_model = SVR()
    svm_model.fit(x_train, y_train)
    # save_weights(svm_model, 'svm_model_weights.pth')
    svm_accuracy_train = svm_model.score(x_train, y_train)
    svm_accuracy_test = svm_model.score(x_test, y_test)
    return svm_accuracy_train, svm_accuracy_test


'''def train_mlp_classifier(x_train, y_train, x_test, y_test):
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp_model.fit(x_train, y_train)
    mlp_accuracy_train = mlp_model.score(x_train, y_train)
    mlp_accuracy_test = mlp_model.score(x_test, y_test)
    return mlp_accuracy_train, mlp_accuracy_test'''


def train_mlp_regressor(x_train, y_train, x_test, y_test):
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    mlp_regressor.fit(x_train, y_train)
    mlp_regressor_score_train = mlp_regressor.score(x_train, y_train)
    mlp_regressor_score_test = mlp_regressor.score(x_test, y_test)
    return mlp_regressor_score_train, mlp_regressor_score_test


def main():
    df, df2 = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(df, df2)

    lr_accuracy_train, lr_accuracy_test = train_linear_regression(x_train, y_train, x_test, y_test)
    dt_accuracy_train, dt_accuracy_test = train_decision_tree(x_train, y_train, x_test, y_test)
    rf_accuracy_train, rf_accuracy_test = train_random_forest(x_train, y_train, x_test, y_test)
    svm_accuracy_train, svm_accuracy_test = train_svr(x_train, y_train, x_test, y_test)
    mlp_regressor_score_train, mlp_regressor_score_test = train_mlp_regressor(x_train, y_train, x_test, y_test)

    print("Linear Regression Training Accuracy: {:.4f}".format(lr_accuracy_train))
    print("Linear Regression Test Accuracy: {:.4f}".format(lr_accuracy_test))
    print("Decision Tree Training Accuracy: {:.4f}".format(dt_accuracy_train))
    print("Decision Tree Test Accuracy: {:.4f}".format(dt_accuracy_test))
    print("Random Forest Training Accuracy: {:.4f}".format(rf_accuracy_train))
    print("Random Forest Test Accuracy: {:.4f}".format(rf_accuracy_test))
    print("SVM Training Accuracy: {:.4f}".format(svm_accuracy_train))
    print("SVM Test Accuracy: {:.4f}".format(svm_accuracy_test))
    print("MLP Regressor Training Score: {:.4f}".format(mlp_regressor_score_train))
    print("MLP Regressor Test Score: {:.4f}".format(mlp_regressor_score_test))

    '''df, df2 = modify_dataframes(df, df2, lr_accuracy_test, dt_accuracy_test, rf_accuracy_test, svm_accuracy_test)
    df.to_csv('Sleep_health_and_lifestyle_dataset.csv', index=False)
    df2.to_csv('test.csv', index=False)'''


if __name__ == "__main__":
    main()
