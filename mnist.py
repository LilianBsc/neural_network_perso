import numpy as np

from network import Network
from layers import FCLayer, ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

# reconnaître des chiffres à partir d'une image
# load MNIST
(x_train , y_train) , (x_test , y_test) = mnist.load_data()

# training data = 60 000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0] , 1 , 28*28)
x_train = x_train.astype('float32')
x_train /= 255

# output est un nombre entre 0 et 9
# on le traduit en un vecteur de taille n de 0 et 1
# cela représente les 10 neuronnes de la dernière couche
# 3 = [0,0,0,1,0,0,0,0,0]
y_train = np_utils.to_categorical(y_train)

# Même chose pour les test data : 10 000 samples
x_test = x_test.reshape(x_test.shape[0] , 1 , 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
net.add(FCLayer(28*28 , 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100 , 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50 , 10))
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.set_loss(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], nb_train=35, learning_rate=0.1)

# Retourner le résultat de MNIST
def mnist_result(predic) :
    #prend en paramètre l'array de predict et retourne le chiffre prédit par le réseau de neurones
    predic = list(predic[0])
    #print(predic)
    return predic.index(max(predic))

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("predicted values : \n")
for array in out :
    print(mnist_result(array))
print(y_test[0:3])
