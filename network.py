from layers import Layer
import numpy as np

class Network :
    def __init__(self) :
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer) :
        #Ajoute une couche
        self.layers.append(layer)

    def set_loss(self, loss, loss_prime) :
        self.loss = (loss)
        self.loss_prime = loss_prime

    def predict(self, input_data) :
        # Prédit output pour un input
        # Passe dans les couches
        samples = len(input_data)
        result = []

        for i in range(samples) :
            # forward propagation
            output = input_data[i]
            for layer in self.layers :
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, nb_train, learning_rate) :
        samples = len(x_train)

        for i in range(nb_train) :
            err = 0
            for j in range(samples) :
                # forward propagation
                output = x_train[j]
                for layer in self.layers :
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j] , output) # calcul erreur pour affichage

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers) :
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples #erreur moyenne
            print(f'Passage numéro {i}/{nb_train} erreur = {err}')
