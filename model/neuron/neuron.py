import numpy as np

# Функция активации sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Нейрон
class SingleNeuron:
    def __init__(self, input_size):
        # Инициализация весов случайным образом
        self.weight = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weight) + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, y, learning_rate=0.01):
        # Вычисляем ошибку
        error = self.output - y
        # Вычисляем градиенты
        d_output = error * sigmoid_derivative(self.output)
        d_weight = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output)

        # Обновление весов и смещения
        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for _ in range(epochs):
            self.forward(X)   # Прямое распространение
            self.backward(y, learning_rate)  # Обратное распространение

    def save_weights(self, filename):
        np.savetxt(filename, np.hstack((self.weight, self.bias)))

    def load_weights(self, filename):
        data = np.loadtxt(filename)
        self.weight = data[:-1]
        self.bias = data[-1]