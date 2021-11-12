import numpy as np

from network import Network
from layers import FCLayer, ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]] , [[0,1]] , [[1,0]] , [[1,1]]])
y_train = np.array([[[0]] , [[1]] , [[1]] , [[0]]])

# network
net = Network()
net.add(FCLayer(2 , 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3 , 1))
net.add(ActivationLayer(tanh, tanh_prime))

# training
net.set_loss(mse, mse_prime)
net.fit(x_train, y_train, nb_train=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
