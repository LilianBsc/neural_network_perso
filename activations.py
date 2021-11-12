import numpy as np

#Les différentes fonctions d'activation et leures dérivées

def tanh(x) :
    return np.tanh(x)

def tanh_prime(x) :
    return 1 - np.tanh(x)**2
