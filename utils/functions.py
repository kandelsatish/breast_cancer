
# importing necessary liberaries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import logging
from utils.variables import *
from utils.functions import *

root_dir = os.path.dirname(os.path.abspath(__file__))


def load_and_prepare_data():
    # Loading the breast cancer dataset
    data = datasets.load_breast_cancer()
    df_bc = pd.DataFrame(data=data.data, columns=data.feature_names)
    df_bc['target'] = data.target

    # Splitting the data into input and output variables
    X = df_bc.drop('target', axis=1)
    y = df_bc['target']
    features=select_features(X,y)
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=42)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, sc,features

def select_features(X,y):
    print("Selecting The Best Features..")
    k = 10
    kb_sel = SelectKBest(score_func=f_regression, k=k)
    X_selected = kb_sel.fit_transform(X, y)
    selected_indices = np.argsort(kb_sel.scores_)[::-1][:k]
    selected_features = X.columns[selected_indices]
    return selected_features



def tune_mlp_hyperparameters(X_train, y_train):
    # Define parameter grid for GridSearch
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],  # varying number of neurons in hidden layers
        'activation': ['relu', 'tanh'],  # activation functions
        'solver': ['adam', 'sgd'],  # optimizers
        'max_iter': [500, 1000],  # number of iterations
    }

    # initialize the MLPClassifier
    model = MLPClassifier(random_state=42)

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

    # Fit the model using grid search
    grid_search.fit(X_train, y_train)

    # Display best parameters and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Cross-validation Accuracy: ", grid_search.best_score_)

    return grid_search.best_estimator_


def build_and_train_mlp(X_train, X_test, y_train, y_test):
    best_estimator_=tune_mlp_hyperparameters(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=best_estimator_.hidden_layer_sizes, max_iter=best_estimator_.max_iter,solver=best_estimator_.solver,activation=best_estimator_.activation,random_state=42)
    mlp.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return mlp


def build_and_train_ann(X_train, y_train):
    # Building the ANN model
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping to avoid overfitting
    early_stopping =  EarlyStopping(monitor='val_accuracy', mode='max', patience=5, min_delta=0.01, verbose=1)

    # Train the model
    hist = ann.fit(X_train, y_train, batch_size=32, epochs=40, validation_split=0.2, callbacks=[early_stopping])

    return ann