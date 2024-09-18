import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[-14., -20000.,     2.],
       [  1.,  20000.,    10.],
       [-18., -20000.,     1.],
       [ -3., -10000.,     7.],
       [ -8.,      0.,     5.],
       [  7.,  30000.,    20.],
       [-13.,  10000.,     6.],
       [ 27.,  70000.,    30.],
       [ 12.,  60000.,    25.],
       [-16., -10000.,     3.]])
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 0])
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')