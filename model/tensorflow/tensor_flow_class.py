import tensorflow as tf

import numpy as np

X_class = np.array([[-14., -20000., 2.],
                    [1., 20000., 10.],
                    [-18., -20000., 1.],
                    [-3., -10000., 7.],
                    [-8., 0., 5.],
                    [7., 30000., 20.],
                    [-13., 10000., 6.],
                    [27., 70000., 30.],
                    [12., 60000., 25.],
                    [-16., -10000., 3.]])
y_class = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 0])

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=100, batch_size=32)

# Прогноз
age = 20 - 35
income = 25_000 - 65_000
experience = 4

test_data = np.array([[age, income, experience]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'Высокий', 'Низкий'))
# Сохранение модели для классификации
model_class.save('classification_model.h5')
