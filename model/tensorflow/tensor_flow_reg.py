import tensorflow as tf

import numpy as np

# температура, влажность, скорость ветра (км/ч)
X_train = np.array([
    [20, 60, 5],
    [25, 70, 10],
    [22, 50, 7],
    [28, 80, 12],
    [30, 90, 15],
    [18, 40, 3],
    [24, 55, 8],
    [26, 65, 11],
    [29, 75, 13],
    [32, 85, 16]
])
# потребление энергии (киловатт-часы)
y_train = np.array([100, 120, 90, 140, 160, 80, 110, 130, 150, 170])

# Создание модели для регрессии
model_reg = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='linear')  # Один выход для регрессии
])

model_reg.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model_reg.fit(X_train, y_train, epochs=500)

# Прогноз
print(model_reg.predict(np.array([[20, 65, 5]])))
# Сохранение модели для регрессии
model_reg.save('regression_model.h5')
