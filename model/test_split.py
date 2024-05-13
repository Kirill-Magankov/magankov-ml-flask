import os
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_shoe_size_test_set():
    dataset_df = pd.read_csv(rf'{current_dir}/datasets/shoe-size.csv')
    X = dataset_df.drop(['Shoe size'], axis=1)
    y = dataset_df['Shoe size']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test


def get_shoe_size_gender_test_set():
    dataset_df = pd.read_csv(rf'{current_dir}/datasets/shoe-size.csv')
    X = dataset_df.drop(['Gender'], axis=1)
    y = dataset_df['Gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test


def get_diabetes_test_set():
    dataset_df = pd.read_csv(rf'{current_dir}/datasets/diabetes.csv')
    dataset_df = dataset_df.drop(["DiabetesPedigreeFunction"], axis=1)
    X = dataset_df.drop(['Outcome'], axis=1)
    y = dataset_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return y_test, X_test
