# -*- coding: utf-8 -*-
import sys, os
from collections import OrderedDict
sys.path.append(os.pardir)
import numpy as np
import src.activation
import src.differentiation
import src.loss
import src.layer

def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network


def forward(network, x, out_activation):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = src.activation.sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = src.activation.sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = out_activation(a3)

  return y


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = src.activation.softmax(z)
        loss = src.loss.cross_entropy_error(y, t)

        return loss


class TwoLayerNet:
    def __init__(
            self, input_size=2, hidden_size=3,
            output_size=1, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = src.layer.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = src.layer.Relu()
        self.layers['Affine2'] = src.layer.Affine(self.params['W2'], self.params['b2'])

        self.last_layer = src.layer.SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = src.differentiation.numerical_gradient(
                loss_W, self.params['W1'])
        grads['b1'] = src.differentiation.numerical_gradient(
                loss_W, self.params['b1'])
        grads['W2'] = src.differentiation.numerical_gradient(
                loss_W, self.params['W2'])
        grads['b2'] = src.differentiation.numerical_gradient(
                loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
