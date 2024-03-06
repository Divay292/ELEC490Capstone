import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
import joblib

epochs = 100


def load_data():
    df = pd.read_csv('ML_Model_Sleep_Data_modified.csv')
    df2 = pd.read_csv('ML_Model_Test_Data.csv')
    '''gender_mapping = {'Male': 0, 'Female': 1}
    # df['Gender'] = df['Gender'].map(gender_mapping)
    df['BMI Category'] = pd.factorize(df['BMI Category'])[0]
    df['Quality of Sleep'] = df['Quality of Sleep'].astype(int)
    df['Sleep Duration'] = df['Sleep Duration'].astype(int)'''
    '''df2['Gender'] = df2['Gender'].map(gender_mapping)
    df2['BMI Category'] = pd.factorize(df2['BMI Category'])[0]
    df2['Quality of Sleep'] = df2['Quality of Sleep'].astype(int)
    df2['Sleep Duration'] = df['Sleep Duration'].astype(int)'''
    return df, df2


def preprocess_data(df, df2):
    variance_threshold = VarianceThreshold()
    x_train = df.iloc[:, 5:12]
    y_train = df.iloc[:, 12:13]
    x_test = df2.iloc[:, 5:12]
    y_test = df2.iloc[:, 12:13]
    x_train = variance_threshold.fit_transform(x_train)
    x_test = variance_threshold.transform(x_test)
    if (np.var(y_train) == 0).any() or (np.var(y_test) == 0).any():
        print("Error: Target variable has zero variance. Aborting.")
        return None, None, None, None
    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)
    selector = SelectKBest(score_func=f_classif, k=6)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    scaler = StandardScaler()
    x_train_selected_scaled = scaler.fit_transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)
    print("Shapes - x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
        x_train.shape, y_train.shape, x_test.shape, y_test.shape))

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
    lr_predictions_train = lr_model.predict(x_train)
    lr_predictions_test = lr_model.predict(x_test)
    lr_accuracy_train = lr_model.score(x_train, y_train)
    lr_accuracy_test = lr_model.score(x_test, y_test)
    return lr_accuracy_train, lr_accuracy_test, lr_predictions_test


def train_decision_tree(df, x_train, y_train, x_test, y_test, max_depth=6, min_samples_leaf=4,
                        min_samples_split=4):
    dt_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                     min_samples_split=min_samples_split)
    dt_model.fit(x_train, y_train)

    # Compute training and validation loss for each level of tree growth
    training_mse = []
    validation_mse = []
    for depth in range(1, max_depth + 1):
        dt_model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf,
                                         min_samples_split=min_samples_split)
        dt_model.fit(x_train, y_train)
        y_train_pred = dt_model.predict(x_train)
        y_test_pred = dt_model.predict(x_test)
        training_loss = mean_squared_error(y_train, y_train_pred)
        validation_loss = mean_squared_error(y_test, y_test_pred)
        training_mse.append(training_loss)
        validation_mse.append(validation_loss)

    # Plot the loss
    plt.figure(figsize=(10, 8))
    plt.plot(training_mse, label="train")
    # plt.plot(validation_mse, label="validation")
    plt.title('Decision Tree Loss Plot')
    plt.xlabel('Tree Depth')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Compute accuracy and predictions
    dt_accuracy_train = dt_model.score(x_train, y_train)
    dt_accuracy_test = dt_model.score(x_test, y_test)
    dt_predictions_train = dt_model.predict(x_train)
    dt_predictions_test = dt_model.predict(x_test)

    return dt_accuracy_train, dt_accuracy_test, dt_predictions_train, dt_predictions_test


def train_random_forest(df, x_train, y_train, x_test, y_test, max_depth=5, min_samples_leaf=1,
                        min_samples_split=2, n_estimators=50):

    rf_model = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                     min_samples_split=min_samples_split, n_estimators=n_estimators)
    rf_model.fit(x_train, y_train)

    # Compute training and validation loss for each number of trees in the forest
    training_mse = []
    validation_mse = []
    for n_trees in range(1, n_estimators + 1):
        rf_model.set_params(n_estimators = n_trees)
        rf_model.fit(x_train, y_train)
        y_train_pred = rf_model.predict(x_train)
        y_test_pred = rf_model.predict(x_test)
        training_loss = mean_squared_error(y_train, y_train_pred)
        validation_loss = mean_squared_error(y_test, y_test_pred)
        training_mse.append(training_loss)
        validation_mse.append(validation_loss)

    # Plot the loss
    plt.figure(figsize=(10, 8))
    plt.plot(training_mse, label="train")
    # plt.plot(validation_mse, label="validation")
    plt.title('Random Forest Loss Plot')
    plt.xlabel('Number of Trees')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Compute accuracy and predictions
    rf_accuracy_train = rf_model.score(x_train, y_train)
    rf_accuracy_test = rf_model.score(x_test, y_test)
    rf_predictions_train = rf_model.predict(x_train)
    rf_predictions_test = rf_model.predict(x_test)

    return rf_accuracy_train, rf_accuracy_test, rf_predictions_train, rf_predictions_test

def train_svr(x_train, y_train, x_test, y_test):
    svm_model = SVR()
    svm_model.fit(x_train, y_train)
    svm_predictions_train = svm_model.predict(x_train)
    svm_predictions_test = svm_model.predict(x_test)
    svm_accuracy_train = svm_model.score(x_train, y_train)
    svm_accuracy_test = svm_model.score(x_test, y_test)
    return svm_accuracy_train, svm_accuracy_test, svm_predictions_train, svm_predictions_test


def train_mlp_regressor(x_train, y_train, x_test, y_test):
    mlp_regressor = MLPRegressor(activation="relu",
                                 max_iter=500,
                                 solver="adam",
                                 random_state=2,
                                 early_stopping=True,
                                 n_iter_no_change=10,
                                 warm_start=True)
    training_mse = []
    validation_mse = []
    mlp_regressor.fit(x_train, y_train)
    loss_df = pd.DataFrame(mlp_regressor.loss_curve_)
    loss_df.plot()

    # Customize the plot with labels
    plt.title('MLP Regressor Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(['Training Loss'])

    plt.show()
    '''for epoch in range(1,epochs):
        mlp_regressor.fit(x_train, y_train) 
        y_pred = mlp_regressor.predict(x_train)
        curr_train_score = mean_squared_error(y_train, y_pred) # training performances
        y_pred = mlp_regressor.predict(x_train)
        curr_valid_score = mean_squared_error(y_train, y_pred) # validation performances
        training_mse.append(curr_train_score)                  # list of training perf to plot
        validation_mse.append(curr_valid_score)                # list of valid perf to plot
    plt.figure(figsize=(10, 8))
    plt.plot(training_mse, label="train")
    plt.plot(validation_mse, label="validation")
    plt.title('MLP Regressor Loss Plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()'''
    mlp_regressor_predictions_train = mlp_regressor.predict(x_train)
    mlp_regressor_predictions_test = mlp_regressor.predict(x_test)
    mlp_regressor_score_train = mlp_regressor.score(x_train, y_train)
    mlp_regressor_score_test = mlp_regressor.score(x_test, y_test)
    return (mlp_regressor_score_train, mlp_regressor_score_test, mlp_regressor_predictions_train,
            mlp_regressor_predictions_test)


def main():
    df, df2 = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(df, df2)

    if 0 in [x_train.shape[0], y_train.shape[0], x_test.shape[0], y_test.shape[0]]:
        print("Error: Training or test data has zero samples. Aborting.")
        return
    
    if np.isnan(x_train).any() or np.isnan(y_train).any() or np.isnan(x_test).any() or np.isnan(y_test).any():
        print("Error: NaN values found in data. Aborting.")
        return
    
    if np.isinf(x_train).any() or np.isinf(y_train).any() or np.isinf(x_test).any() or np.isinf(y_test).any():
        print("Error: Infinite values found in data. Aborting.")
        return

    lr_accuracy_train, lr_accuracy_test, lr_predictions_test = train_linear_regression(x_train, y_train, x_test, y_test)
    dt_accuracy_train, dt_accuracy_test, dt_loss_history, dt_predictions_test = train_decision_tree(df, x_train, y_train, x_test, y_test)
    rf_accuracy_train, rf_accuracy_test, rf_loss_history, rf_predictions_test = train_random_forest(df, x_train, y_train, x_test, y_test)
    svm_accuracy_train, svm_accuracy_test, svm_loss_history, svm_predictions_test = train_svr(x_train, y_train, x_test, y_test)
    mlp_regressor_score_train, mlp_regressor_score_test, mlp_loss_history, mlp_regressor_predictions_test = train_mlp_regressor(x_train, y_train, x_test, y_test)

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
    
    # Plotting test values
    plt.figure(figsize=(10, 8))
    # plt.scatter(y_test, lr_predictions_test, color='blue', label='Linear Regression')
    plt.scatter(y_test, dt_predictions_test, color='green', label='Decision Tree')
    plt.scatter(y_test, rf_predictions_test, color='red', label='Random Forest')
    # plt.scatter(y_test, svm_predictions_test, color='orange', label='SVM')
    plt.scatter(y_test, mlp_regressor_predictions_test, color='purple', label='MLP Regressor')
    plt.plot(y_test, y_test, color='black', linestyle='--', linewidth=0.5)

    plt.title('Predicted Test Value vs Actual Test Value')
    plt.xlabel('Actual Test Value')
    plt.ylabel('Predicted Test Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
