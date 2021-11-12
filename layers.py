import numpy as np

# Classe base

class Layer :
    def __init__(self) :
        self.input
        self.output

    def forward_propagation(self, input) :
        # Calcul output Y à partir de l'input X
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate) :
        # Calcul dE/dX à partir de dE/dY
        raise NotImplementedError



# Fully Connected Layer
class FCLayer(Layer) :
    def __init__(self, input_size, output_size) :
        # input_size = nombre de neuronnes d'entré
        # output_size = nombre de neuronnes de sortie
        self.weights = np.random.rand(input_size , output_size) - 0.5 # variable aléatoire centrée autour de 0
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data) :
        self.input = input_data
        self.output = np.dot(self.input , self.weights) + self.bias
        # Y = XW + B
        return self.output

    def backward_propagation(self, output_error, learning_rate) :
        # Calcul dE/dW, dE/dB pour output_error = dE/dY donnée
        # dE/dX = dE/dY*t(W)
        # dE/dW = t(X)dE/dY
        # dE/dB = dE/dY
        # retourne dE/dX
        input_error = np.dot(output_error , self.weights.T)
        weights_error = np.dot(self.input.T , output_error)

        # Update FCLayer
        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*output_error

        return input_error


# Activation Layer
class ActivationLayer(Layer) :
    def __init__(self, activation, activation_prime) :
        self.activation = activation #fonction d'activation
        self.activation_prime = activation_prime #fonction d'activation dérivée

    def forward_propagation(self, input_data) :
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate) :
        # formule dE/dX = dE/dX.f'(X)
        # où '.' est la multiplication termes à terme de matrices
        return self.activation_prime(self.input)*output_error
