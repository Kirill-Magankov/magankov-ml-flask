import pickle
from enum import Enum, auto

import math
import numpy as np
import pandas as pd

from model.neuron.neuron import SingleNeuron
from model.test_split import get_shoe_size_test_set, get_shoe_size_gender_test_set, get_diabetes_test_set

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron/neuron_weights.txt')

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
