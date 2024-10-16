import math
import numpy as np
from flask import Blueprint, jsonify, request

from helpers import get_classification_metrics, ModelTypes, gender_list, gender_shoe_model, get_regression_metrics, \
    shoe_model, diabetes_model, diabetes_status, diabetes_tree_model, new_neuron, model_class, model_reg

rest_api = Blueprint('api', __name__, )


# region Tensorflow
@rest_api.route('/tensorflow-regression', methods=['POST'])
def tensorflow_regression_get():
    data = request.get_json()
    input_data = np.array([[
        int(data.get('temperature')),
        int(data.get('humidity')),
        int(data.get('wind_speed')),
    ]])

    pred = model_reg.predict(input_data)[0][0]
    return jsonify(
        msg=f"Потребление энергии: {round(float(pred), 2)} (киловатт-часы)",
        inputs=data,
    )


@rest_api.route('/tensorflow-classification', methods=['POST'])
def tensorflow_classification_get():
    data = request.get_json()
    input_data = np.array([[
        int(data.get('age')) - 35,
        int(data.get('income')) - 65_000,
        int(data.get('experience')),
    ]])

    pred = model_class.predict(input_data)
    result = np.where(pred > 0.5, "Высокий", "Низкий")[0][0]
    return jsonify(msg=f"Уровень дохода: {result}", probability=str(round(pred[0][0], 6)))


# endregion

# region Basic Models
@rest_api.route('/knn', methods=['POST'])
def api_knn_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['shoeSize'])]])

    pred = gender_shoe_model.predict(X_new)
    return jsonify(msg=f"Gender: {gender_list[pred[0]]}")


@rest_api.route('/shoe-size', methods=['POST'])
def api_shoe_size_get():
    request_data = request.get_json()

    X_new = np.array([[float(request_data['height']),
                       float(request_data['weight']),
                       float(request_data['gender'])]])

    pred = shoe_model.predict(X_new)
    return jsonify(msg=f"Shoe size: {math.ceil(pred[0])}")


@rest_api.route('/diabetes', methods=['POST'])
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


@rest_api.route('/diabetes-tree', methods=['POST'])
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


@rest_api.route('/metrics', methods=['GET'])
def api_metrics_get():
    metrics = {
        "diabetes_decision_tree": get_classification_metrics(ModelTypes.DIABETES_TREE, True),
        "diabetes_model": get_classification_metrics(ModelTypes.DIABETES, True),
        "gender_shoe_model": get_classification_metrics(ModelTypes.GENDER_SHOE, True),
        "shoe_model": get_regression_metrics(ModelTypes.SHOE_MODEL)
    }

    return jsonify(msg=metrics)
# endregion
