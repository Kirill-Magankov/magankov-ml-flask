import math
import pickle
import warnings

import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

diabetes_model = pickle.load(open('model/diabetes.pickle', 'rb'))
gender_shoe_model = pickle.load(open('model/shoe-size-gender.pickle', 'rb'))
shoe_model = pickle.load(open('model/shoe-size_predict.pickle', 'rb'))

diabetes_status = ["нет", "есть"]
gender_list = ["женский", "мужской"]


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы по ML", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['height']),
                           float(request.form['weight']),
                           float(request.form['shoeSize'])]])

        pred = gender_shoe_model.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model=f"Пол: {gender_list[pred[0]]}")


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'POST':
        X_new = np.array([[float(request.form['height']),
                           float(request.form['weight']),
                           float(request.form['gender'])]])

        pred = shoe_model.predict(X_new)

        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model=f"Размер обуви: {math.ceil(pred[0])}")

    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'POST':
        X_new = np.array([[float(request.form['pregnancies']),
                           float(request.form['glucose']),
                           float(request.form['bloodPressure']),
                           float(request.form['skinThickness']),
                           float(request.form['insulin']),
                           float(request.form['bmi']),
                           float(request.form['age'])]])

        pred = diabetes_model.predict(X_new)
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model=f"Диабет: {diabetes_status[pred[0]]}")

    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'POST':
        X_new = np.array([[float(request.form['pregnancies']),
                           float(request.form['glucose']),
                           float(request.form['bloodPressure']),
                           float(request.form['skinThickness']),
                           float(request.form['insulin']),
                           float(request.form['bmi']),
                           float(request.form['age'])]])

        pred = diabetes_model.predict(X_new)
        return render_template('lab4.html', title="Древо решений", menu=menu,
                               class_model=f"Диабет: {diabetes_status[pred[0]]}")

    return render_template('lab4.html', title="Древо решений", menu=menu)


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    app.run(debug=True)
