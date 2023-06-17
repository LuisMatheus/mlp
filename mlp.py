import numpy as np


class layer_dense:
    def __init__(self, weights, neurons):
        self.weights = 0.10 * np.random.randn(weights, neurons)
        self.neurons = np.ones((1, neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.neurons


class activation:
    def forward(self, input):
        self.output = 1.0 / (1.0 + np.exp(-input))


class activation_softmax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class loss_cross_entropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidence = y_pred_clipped[range(samples), y_true]
        return -np.log(correct_confidence)


class mlp:
    def __init__(self, input, output, hidden_layers=[1, 2, 3, 4], max_epochs=100, learning_rate=0.1):
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.max_epochs = max_epochs

        # init weights
        layers = []
        layers.append(layer_dense(input.shape[1], input.shape[1]))
        print("criado primeiro layer")
        for i in range(len(hidden_layers)):
            print("Criando layer", i)
            layers.append(layer_dense(
                layers[i].neurons.shape[1], hidden_layers[i]))
        layers.append(layer_dense(hidden_layers[-1], output.shape[1]))
        print("Criado ultimo layer")

        # init functions
        activations = []
        for i in range(len(hidden_layers)):
            activations.append(activation())
        activations.append(activation_softmax())
        self.activation = activations

        self.layers = layers

    def train(self):
        print("Initializing the train process....")

        epochs = 0
        while epochs < self.max_epochs:
            print(f"[EPOCH] {epochs} ")

            for x, d in zip(self.input, self.output):

                self.layers[0].forward(x)
                self.activation[0].forward(self.layers[0].output)

                for i in range(1, len(self.layers)-1):
                    self.layers[i].forward(self.layers[i-1].output)
                    self.activation[i].forward(self.layers[i].output)

                self.layers[-1].forward(self.layers[-2].output)
                self.activation[-1].forward(self.layers[-1].output)

                print(f"output: {self.layers[-1].output} d: {d}")
            epochs = epochs + 1


input = np.array([[0.1, 0.2], [0.3, 0.4]])
labels = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
hidden_layers = [1, 2, 3]
test = mlp(input, labels, hidden_layers=hidden_layers)
test.train()
