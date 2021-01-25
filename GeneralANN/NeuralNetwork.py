#######################################################################################################################
# Made with love by me (Christian Di Maio) and thanks to StackOverflow and GitHub users
#######################################################################################################################

# Initialize a network
import math
from random import random
from GeneralANN.Neuron import Neuron
from random import *
from enum import Enum
import numpy as np
from keras import *
from ann_visualizer.visualize import ann_viz

class NeuralNetworkScope(Enum):
    Classification = 1
    Regression = 2

class NeuralNetwork:
    def __init__(self,Scope : NeuralNetworkScope, n_inputs, n_hidden_layers, n_neuron_for_hidden, n_outputs = 1):
        '''
        Initialize a fully connected neural network
        :param n_inputs: Number of inputs
        :param n_hidden_layers: number of hidden layer
        :param n_neuron_for_hidden: number of neuron for each hidden layer
        :param n_outputs: number of output
        :return: the network

        Examples
        network = initialize_network(2, 1, 2, 1)
        for layer in network:
            print(layer)
        '''
        if n_inputs < 1:
            raise Exception("Hey man, you need at least 1 input! :D")
        self.Scope = Scope
        self.N_inputs = n_inputs
        self.N_outputs = n_outputs
        self.network = list()
        self.N_neuron_for_hidden_layer = n_neuron_for_hidden
        self.N_hidden_layers = n_hidden_layers

        #First hidden layer
        first_hidden_layer = []
        for i in range(n_neuron_for_hidden):
            _neuron = Neuron(list([uniform(-0.5, +0.5) for j in range(n_inputs + 1)]))
            first_hidden_layer.append(_neuron)
        self.network.append(first_hidden_layer)
        #Deep layer
        for i in range(n_hidden_layers-1):
            deep_layer = []
            for j in range(n_neuron_for_hidden):
                _neuron = Neuron(list([uniform(-0.5, +0.5) for j in range(n_neuron_for_hidden + 1)]))
                deep_layer.append(_neuron)
            #hidden_layer = [Neuron(list([random() for j in range(n_inputs + 1)]) for i in range(n_neuron_for_hidden))]
            self.network.append(deep_layer)

        output_layer = [Neuron(list([uniform(-0.5, +0.5) for i in range(n_inputs + 1 if n_hidden_layers==0 else n_neuron_for_hidden+1)]))
                                                                            for i in range(n_outputs)]
        self.network.append(output_layer)

    # Forward propagate input to a network output
    def propagate(self, inputs):
        _inputs = inputs
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                response = neuron.response(inputs)
                new_inputs.append(response)
            inputs = new_inputs
        return inputs           # In the last iteration the inputs will be the output,
                                # so even if is called inputs they represent the output

    # Backpropagate error and store in neurons
    def backward_propagate(self,expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1: #Calculate error on the hidden layers
                for neuron_idx in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        # I assumed that the neurons of the layer above have already calculated their own deltas
                        #                     # and it works bcz the reversed loop :)
                        error += (neuron.get_weights()[neuron_idx] * neuron.get_delta())
                    errors.append(error)
            else:
                for j in range(len(layer)): #Calculate error on the output layer
                    neuron = layer[j]
                    errors.append(expected[j] - neuron.get_output())
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.set_delta(errors[j] * neuron.response_derivative())

    # def backward_propagate(self,expected):
    #     for layer_idx in reversed(range(len(self.network))):
    #         layer = self.network[layer_idx]
    #         errors = list()
    #         if layer_idx != len(self.network) - 1: # Calculate error on the hidden layers
    #             for neuron_idx in range(len(layer)): # get the index of j-th neuron of this layer and iterate on it
    #                 error = 0.0
    #                 for neuron_above in self.network[layer_idx + 1]: # Get all neurons above
    #                     # I assumed that the neurons of the layer above have already calculated their own deltas
    #                     # and it works bcz the reversed loop :)
    #                     error += (neuron_above.get_weights()[neuron_idx] * neuron.get_delta())
    #                 errors.append(error)
    #         else: # Calculate error on the output layer
    #             for neuron_idx in range(len(layer)):
    #                 neuron = layer[neuron_idx]
    #                 errors.append(expected[neuron_idx] - neuron.get_output())
    #         for neuron_idx in range(len(layer)):
    #             neuron = layer[neuron_idx] # Pick a neuron of this layer
    #             # Indipendently if the layer is hidden or its the output i just calculate the delta as
    #             # error * derivative of the response
    #             neuron.set_delta(errors[neuron_idx] * neuron.response_derivative())

    def update_weights(self, inputs, l_rate): # Stochastic Gradient Descent
        for layer_idx in range(len(self.network)):
            inputs = inputs[:-1]
            if layer_idx != 0:
                inputs = [neuron.get_output() for neuron in self.network[layer_idx - 1]]
            for neuron in self.network[layer_idx]:
                for j in range(len(inputs)):
                    neuron.get_weights()[j] += l_rate * neuron.get_delta() * inputs[j]
                neuron.get_weights()[-1] += l_rate * neuron.get_delta()

    # Train a network for a fixed number of epochs
    def train_network_as_Classificator(self,train, l_rate, n_epoch, n_outputs):
        np.random.shuffle(train)
        if self.Scope == NeuralNetworkScope.Regression:
            raise Exception("Hey man, you defined this network to work with regression problem, pay attention! :D")
        # Normalize the input
        Input = train[:,:-1]
        Normalized_input = Input / Input.max()
        Target = train[:,-1:]
        for epoch in range(n_epoch):
            sum_error = 0
            for inputs,target in zip(Normalized_input,Target):
                outputs = self.propagate(inputs)
                # Apply the one hot encoding into 2 line of code, thanks python :)
                if n_outputs != 1:
                    expected = [0 for i in range(n_outputs)]
                    expected[np.asscalar(target) if type(np.asscalar(target)) is int else np.asscalar(target).value] = 1
                expected = target
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]) / len(train)
                self.backward_propagate(expected)
                self.update_weights(inputs, l_rate)
            print(f'>epoch={epoch:d}, lrate={l_rate:.3f}, error={math.sqrt(sum_error):.3f},'
                  f'     accuracy={(1-math.sqrt(sum_error))*100:.2f}%')

    # Train a network for a fixed number of epochs
    def train_network_as_Regression(self, train, l_rate, n_epoch):
        if self.Scope == NeuralNetworkScope.Classification:
            raise Exception("Hey man, you defined this network to work with classification problem, pay attention! :D")
        X_train = train[:,:self.N_inputs]
        Y_train = train[:,-1:]
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.propagate(row)
                # Apply the one hot encoding into 2 line of code, thanks python :)

                sum_error += (row[-1] - outputs) ** 2  / len(train)
                self.backward_propagate(list([row[-1]])) # Even if in this case we have only 1 expected value, we want to generalize our solution
                self.update_weights(row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f, accuracy=%.2f' % (epoch, l_rate, math.sqrt(sum_error),(1-math.sqrt(sum_error))*100))

    def predict(self, row):
        outputs = self.propagate(row)
        # Assume that in the classificator each output neuron is ordered from left to right i pick up the index of the
        # neuron that respond with the maximum value :)
        return outputs.index(max(outputs)) if self.Scope == NeuralNetworkScope.Classification else max(outputs)


    def show_me(self):
        _network = models.Sequential()  # Output of each layer we add, is the input to the next layer we specify
        #1st Hidden Layer#1
        _network.add(layers.Dense(units=self.N_neuron_for_hidden_layer,  # Dense stay for a fully connected layer
                                 activation='sigmoid',
                                 kernel_initializer='uniform',
                                 input_dim=self.N_inputs))
        #Deep arch.
        for i in range(self.N_hidden_layers -1):
            _network.add(layers.Dense(units=self.N_neuron_for_hidden_layer,
                                     activation='sigmoid',
                                     kernel_initializer='uniform'))
        #Output layer
        _network.add(layers.Dense(units=self.N_outputs,
                                 activation='sigmoid',
                                 kernel_initializer='uniform'))
        #Visualize it
        ann_viz(_network, title="My ANN")
    #
