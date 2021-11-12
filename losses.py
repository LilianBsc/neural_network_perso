import numpy as np

# Les différentes fonctions de pertes (E) i.e. les fonctions qui calculent l'erreur sur la couche finale
# Ainsi que leur dérivée dE/dY

def mse(ytrue, ypred) :
    #Mean Square Error
    return np.mean(np.power(ytrue - ypred, 2))

def mse_prime(ytrue, ypred) :
    return 2*(ypred - ytrue)/ytrue.size
