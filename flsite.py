import pickle

import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)


@app.route("/p_lab2")
def f_lab2():
    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
