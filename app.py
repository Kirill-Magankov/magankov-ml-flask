import math
import warnings

import numpy as np
from flask import Flask, render_template, request

from api import rest_api
from helpers import get_classification_metrics, ModelTypes, gender_list, gender_shoe_model, get_regression_metrics, \
    shoe_model, diabetes_model, diabetes_status, diabetes_tree_model, new_neuron

app = Flask(__name__)

app.register_blueprint(rest_api, url_prefix='/api/v1')

menu = [
    {"name": "Лаба 1", "url": "p_knn"},
    {"name": "Лаба 2", "url": "p_lab2"},
    {"name": "Лаба 3", "url": "p_lab3"},
    {"name": "Лаба 4", "url": "p_lab4"},
    {"name": "Лаба 5", "url": "p_lab5"},
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


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    app.run(debug=True)
