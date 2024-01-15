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
    # bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['BMI Category'] = pd.factorize(df['BMI Category'])[0]
    # print(df.dtypes)

    # Data preprocessing
    x = df.iloc[:, 0: 8]
    y = df.iloc[:, 8:9]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
    x_train = x_train.sort_values(by='Person ID')
    x_test = x_test.sort_values(by='Person ID')

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=6)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    scaler = StandardScaler()
    x_train_selected_scaled = scaler.fit_transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)

    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)

    # Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000, C=1.0)     # C is the regularization
    lr_model.fit(x_train_selected_scaled, y_train)
    lr_accuracy = lr_model.score(x_test_selected_scaled, y_test)

    # Feature selection using Recursive Feature Elimination (RFE)
    estimator = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion='gini',
                                       max_features=None)
    rfe_selector = RFE(estimator, n_features_to_select=6)
    x_train_selected_rfe = rfe_selector.fit_transform(x_train, y_train)
    x_test_selected_rfe = rfe_selector.transform(x_test)

    # Decision Tree Model
    # min_samples_split and min_samples_leaf control structure of tree
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion='gini',
                                      max_features=None)
    dt_model.fit(x_train_selected_rfe, y_train)
    dt_accuracy = dt_model.score(x_test_selected_rfe, y_test)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                     criterion='gini', max_features='auto')
    rf_model.fit(x_train_selected_rfe, y_train)
    rf_accuracy = rf_model.score(x_test_selected_rfe, y_test)

    # Support Vector Machines (SVM) Model
    # C parameter controls the trade-off between having a smooth decision
    # boundary and classifying the training points correctly
    # Kernel parameter determines the type of kernel used (linear, poly, radial basis function(rbf), sigmoid)
    svm_model = SVC(C=1.0, kernel='rbf', gamma='scale')
    svm_model.fit(x_train_selected_rfe, y_train)
    svm_accuracy = svm_model.score(x_test_selected_rfe, y_test)

    # Displaying model accuracies
    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Decision Tree Accuracy:", dt_accuracy)
    print("Random Forest Accuracy:", rf_accuracy)
    print("SVM Accuracy:", svm_accuracy)

    '''# Define the parameter grid for Logistic Regression
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                     'max_iter': [100, 500, 1000]}

    # Create Logistic Regression model
    lr_model = LogisticRegression()

    # Perform GridSearchCV
    grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5)
    grid_search_lr.fit(x_train_selected_scaled, y_train)

    # Print best hyperparameters
    print("Best hyperparameters for Logistic Regression:", grid_search_lr.best_params_)'''


if __name__ == "__main__":
    main()
