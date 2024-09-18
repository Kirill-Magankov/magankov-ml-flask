import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data = np.array([[ 23-35,  2_000_000 - 60_000.,    20.]])
predictions = new_neuron.forward(test_data)
print("Уровень дохода:", predictions, *np.where(predictions >= 0.5, 'Высокий', 'Низкий'))