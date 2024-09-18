import json
import math
import pickle
import warnings
from enum import Enum, auto

import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score

from model.neuron.neuron import SingleNeuron
from model.test_split import get_shoe_size_test_set, get_shoe_size_gender_test_set, get_diabetes_test_set

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron/neuron_weights.txt')

app = Flask(__name__)

menu = [
    {"name": "Лаба 1", "url": "p_knn"},
    {"name": "Лаба 2", "url": "p_lab2"},
    {"name": "Лаба 3", "url": "p_lab3"},
    {"name": "Лаба 4", "url": "p_lab4"},
    {"name": "Лаба 5", "url": "p_lab5"},
]

diabetes_model = pickle.load(open('model/diabetes.pickle', 'rb'))
diabetes_tree_model = pickle.load(open('model/diabetes_decision_tree.pickle', 'rb'))
gender_shoe_model = pickle.load(open('model/shoe-size-gender.pickle', 'rb'))
shoe_model = pickle.load(open('model/shoe-size_predict.pickle', 'rb'))

diabetes_status = ["нет", "есть"]
gender_list = ["женский", "мужской"]


class ModelTypes(Enum):
    DIABETES = auto()
    DIABETES_TREE = auto()
    GENDER_SHOE = auto()
    SHOE_MODEL = auto()


def get_regression_metrics(model_type: ModelTypes):
    if model_type == ModelTypes.SHOE_MODEL:
        y_test, x_test = get_shoe_size_test_set()
        y_pred = shoe_model.predict(x_test)  # noqa

    else:
        return None

    return [{"title": "MSE", "value": round(mean_squared_error(y_test, y_pred), 4)},
            {"title": "RMSE", "value": round(math.sqrt(mean_absolute_error(y_test, y_pred)), 4)},
            {"title": "MSPE", "value": f"{round(np.mean(np.square((y_test - y_pred) / y_test)) * 100, 2)} %"},
            {"title": "MAE", "value": round(mean_absolute_error(y_test, y_pred), 4)},
            {"title": "MAPE", "value": f"{round(mean_absolute_percentage_error(y_test, y_pred), 2)} %"},
            {"title": "MRE", "value": round(mean_squared_error(y_test, y_pred), 4)},
            {"title": "R-Квадрат", "value": round(r2_score(y_test, y_pred), 4)}]


def get_classification_metrics(model_type: ModelTypes, to_json=False):
    if model_type == ModelTypes.GENDER_SHOE:
        y_test, x_test = get_shoe_size_gender_test_set()
        y_pred = gender_shoe_model.predict(x_test)  # noqa

    elif model_type == ModelTypes.DIABETES:
        y_test, x_test = get_diabetes_test_set()
        y_pred = diabetes_model.predict(x_test)  # noqa

    elif model_type == ModelTypes.DIABETES_TREE:
        y_test, x_test = get_diabetes_test_set()
        y_pred = diabetes_tree_model.predict(x_test)  # noqa

    else:
        return

    frame = {'y_Actual': y_test, 'y_Predicted': y_pred}
    df = pd.DataFrame(frame, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    return [{"title": "Confusion matrix", "value": confusion_matrix.to_dict() if to_json else confusion_matrix},
            {"title": "Accuracy", "value": round(accuracy_score(y_test, y_pred), 4)},
            {"title": "Precision", "value": round(precision_score(y_test, y_pred), 4)},
            {"title": "Recall", "value": round(recall_score(y_test, y_pred), 4)},
            {"title": "F1-мера", "value": round(f1_score(y_test, y_pred), 4)},
            ]


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы по ML", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    metrics = get_classification_metrics(ModelTypes.GENDER_SHOE)

    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='',
                               metrics=metrics)
    if request.method == 'POST':
        X_new = np.array([[float(request.form['height']),
                           float(request.form['weight']),
                           float(request.form['shoeSize'])]])

        pred = gender_shoe_model.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model=f"Пол: {gender_list[pred[0]]}", metrics=metrics)


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    metrics = get_regression_metrics(ModelTypes.SHOE_MODEL)

    if request.method == 'POST':
        X_new = np.array([[float(request.form['height']),
                           float(request.form['weight']),
                           float(request.form['gender'])]])

        pred = shoe_model.predict(X_new)

        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model=f"Размер обуви: {math.ceil(pred[0])}", metrics=metrics)

    return render_template('lab2.html', title="Логистическая регрессия", menu=menu, metrics=metrics)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    metrics = get_classification_metrics(ModelTypes.DIABETES)

    if request.method == 'POST':
        X_new = np.array([[float(request.form['pregnancies']),
                           float(request.form['glucose']),
                           float(request.form['bloodPressure']),
                           float(request.form['skinThickness']),
                           float(request.form['insulin']),
                           float(request.form['bmi']),
                           float(request.form['age'])]])

        pred = diabetes_model.predict(X_new)
        return render_template('lab3.html', title="Классификация", menu=menu,
                               class_model=f"Диабет: {diabetes_status[pred[0]]}", metrics=metrics)

    return render_template('lab3.html', title="Классификация", menu=menu, metrics=metrics)


@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    metrics = get_classification_metrics(ModelTypes.DIABETES_TREE)

    if request.method == 'POST':
        X_new = np.array([[float(request.form['pregnancies']),
                           float(request.form['glucose']),
                           float(request.form['bloodPressure']),
                           float(request.form['skinThickness']),
                           float(request.form['insulin']),
                           float(request.form['bmi']),
                           float(request.form['age'])]])

        pred = diabetes_tree_model.predict(X_new)
        return render_template('lab4.html', title="Древо решений", menu=menu,
                               class_model=f"Диабет: {diabetes_status[pred[0]]}", metrics=metrics)

    return render_template('lab4.html', title="Древо решений", menu=menu, metrics=metrics)


@app.route("/p_lab5", methods=['POST', 'GET'])
def f_lab():
    if request.method == 'GET':
        return render_template('lab5.html', title="Первый нейрон", menu=menu, class_model='')

    if request.method == 'POST':
        X_new = np.array([[
            float(request.form['age']) - 35,
            float(request.form['income']) - 60_000,
            float(request.form['experience']),
        ]])

        pred = new_neuron.forward(X_new)
        return render_template('lab5.html', title="Первый нейрон", menu=menu,
                               class_model=f"Доход: {np.where(pred >= 0.5, 'Высокий', 'Низкий')}")


# region Api Endpoints
api_endpoint = "/api/v1"


@app.route(f'{api_endpoint}/knn', methods=['POST'])
def api_knn_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['shoeSize'])]])

    pred = gender_shoe_model.predict(X_new)
    return jsonify(msg=f"Gender: {gender_list[pred[0]]}")


@app.route(f'{api_endpoint}/shoeSize', methods=['POST'])
def api_shoe_size_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['gender'])]])

    pred = shoe_model.predict(X_new)
    return jsonify(msg=f"Shoe size: {math.ceil(pred[0])}")


@app.route(f'{api_endpoint}/diabetes', methods=['POST'])
def api_diabetes_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['pregnancies']),
                       float(request_data['glucose']),
                       float(request_data['bloodPressure']),
                       float(request_data['skinThickness']),
                       float(request_data['insulin']),
                       float(request_data['bmi']),
                       float(request_data['age'])]])

    pred = diabetes_model.predict(X_new)
    return jsonify(msg=f"Diabetes: {diabetes_status[pred[0]]}")


@app.route(f'{api_endpoint}/diabetesTree', methods=['POST'])
def api_diabetes_tree_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['pregnancies']),
                       float(request_data['glucose']),
                       float(request_data['bloodPressure']),
                       float(request_data['skinThickness']),
                       float(request_data['insulin']),
                       float(request_data['bmi']),
                       float(request_data['age'])]])

    pred = diabetes_tree_model.predict(X_new)
    return jsonify(msg=f"Diabetes: {diabetes_status[pred[0]]}")


@app.route(f'{api_endpoint}/metrics', methods=['GET'])
def api_metrics_get():
    metrics = {
        "diabetes_decision_tree": get_classification_metrics(ModelTypes.DIABETES_TREE, True),
        "diabetes_model": get_classification_metrics(ModelTypes.DIABETES, True),
        "gender_shoe_model": get_classification_metrics(ModelTypes.GENDER_SHOE, True),
        "shoe_model": get_regression_metrics(ModelTypes.SHOE_MODEL)
    }

    return jsonify(msg=metrics)


# endregion

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    app.run(debug=True)
