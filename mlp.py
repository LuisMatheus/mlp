from random import random
import numpy as np
import pandas as pd


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    if activation >= 0:
        return 1.0 / (1.0 + np.exp(-activation))
    else:
        return np.exp(activation) / (np.exp(activation) + 1.0)


def forward_propagate(network, row):
    inputs = row
    for i in range(len(network)):
        layer = network[i]
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for X, y in zip(train, outputs):
            predicted_outputs = forward_propagate(network, X)
            sum_error += sum([(y[i]-predicted_outputs[i])
                             ** 2 for i in range(len(y))])
            backward_propagate_error(network, y)
            update_weights(network, X, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = 2 * (array - min_value) / (max_value - min_value) - 1
    return normalized_array


dataset = pd.read_csv('EMG.csv', header=None, sep=',', dtype=np.float64)
dataset = normalize_array(dataset.values)
labels = pd.read_csv('Rotulos.csv', header=None, sep=',', dtype=np.float64)
labels = labels.values
labels = np.where(labels == -1, 0, labels)

n_inputs = len(dataset[0])
n_outputs = len(labels[0])

n_layers = 20
l_rate = 0.2
n_epoch = 100

network = initialize_network(n_inputs, n_layers, n_outputs)
train_network(network, dataset, l_rate, n_epoch, labels)
# for layer in network:
#   pass
