import numpy as np


class layer_dense:
    def __init__(self, weights, neurons):
        self.weights = 0.10 * np.random.randn(weights, neurons)
        self.neurons = np.ones((1, neurons)) * -1

    def foward(self, input):
        self.output = np.dot(input, self.weights) + self.neurons


class activation_relu:
    def foward(self, input):
        self.output = np.maximum(0, input)


class activation_softmax:
    def foward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class loss_cross_entropy(loss):
    def foward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidence = y_pred_clipped[range(samples), y_true]
        return -np.log(correct_confidence)


class mlp:
    def __init__(self, input, output, hidden_layers=[1, 2, 3, 4], max_epochs=100, learning_rate=0.1):
        self.input = input,
        self.output = output,
        self.learning_rate = learning_rate,
        self.hidden_layers = hidden_layers,
        self.max_epochs = max_epochs,

        # init weights
        layers = []
        layers.append(layer_dense(input.shape[1], input.shape[1]))
        print("criado primeiro layer")
        for i in range(len(hidden_layers)):
            print("Criando layer", i)
            layers.append(layer_dense(layers[i].neurons.shape[1], hidden_layers[i]))
        layers.append(layer_dense(hidden_layers[-1], output.shape[1]))
        print("Criado ultimo layer")

        self.layers = layers

    def print(self):
        print([i.neurons for i in self.layers])

    def train(self):
        print("Initializing the train process....")

        epochs = 0,
        while epochs < self.max_epochs:
            print(f"[EPOCH] {epochs} ")

        for x, d in zip(self.input, self.output):

            pass
        epochs += 1


input = np.array([[0.1, 0.2], [0.3, 0.4]])
labels = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
hidden_layers = [1, 2, 3]
test = mlp(input, labels, hidden_layers=hidden_layers)

test.print()
