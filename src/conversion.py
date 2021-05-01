import numpy


class Neuron:
    def __init__(self, learning_rate=0.1):
        self.weights = numpy.random.sample()
        self.learning_rate = learning_rate

    def predict(self, data):
        return self.weights * data

    def training(self, data, expected_result):
        actual_data = self.predict(data)
        error = expected_result - actual_data
        self.weights = self.weights + (error / actual_data * self.learning_rate)


def average_error(val1, val2):
    return numpy.mean((val1 - val2)**2)


dataset = [10, 50]

epochs = 5000
learn_rate = 0.077

neuron = Neuron(learning_rate=learn_rate)


for i in range(epochs):
    neuron.training(dataset[0], dataset[1])

user_input = 10

print(neuron.predict(user_input))
