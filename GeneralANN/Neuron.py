#######################################################################################################################
# Made with love by me (Christian Di Maio) and thanks to StackOverflow and GitHub users
#######################################################################################################################

from math import exp
class Neuron:
    '''
    Class who define a Neuron with a sigmoid activation function
    '''
    # Calculate neuron activation for an input

    def __init__(self,weights):
        '''
        Create a Neuron with a given weights or create a random one
        :param args: the array of weights
        '''
        self.weights = weights
        self.delta = 0
        self.output = 0

    def activate(self, inputs):
        activation = self.weights[-1] #Recover the bias
        for i in range(len(self.weights) - 1):
            activation += self.weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def response(self, inputs):
        '''
        Represent the response of the neuron
        :return: return the response according to a logistic sigmoidal function
        '''
        self.output = 1.0 / (1.0 + exp(-self.activate(inputs)))
        return self.output

    # Calculate the derivative of an neuron output
    def response_derivative(self):
        # We are using the sigmoid transfer function, the derivative of which can be calculated as follows:
        return self.output * (1.0 - self.output)

    def get_weights(self):
        return self.weights

    def set_weights(self,weights):
        self.weights = weights

    def get_delta(self):
        return self.delta

    def set_delta(self,delta):
        self.delta = delta

    def get_output(self):
        return self.output


