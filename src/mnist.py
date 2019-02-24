# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
import thirdparty.mnist
import src.nn, src.activation
from PIL import Image
import pickle
import pathlib


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = \
        thirdparty.mnist.load_mnist(normalize, flatten, one_hot_label)

    return (x_train, t_train), (x_test, t_test)


def show_image(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


def init_network():
    p = pathlib.Path(os.path.dirname(__file__) + '/../thirdparty/mnist.pkl')
    with open(str(p.resolve()), 'rb') as f:
        network = pickle.load(f)

    return network


def evaluate(batch_num):
    _, (x_test, t_test) = load_mnist()
    network = init_network()

    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_num):
        x = x_test[i : i + batch_num]
        y = src.nn.forward(network, x, src.activation.softmax)
        p = np.argmax(y, axis=1)
        accuracy_cnt += np.sum(p == t_test[i : i + batch_num])

    print('Accuracy:' + str(float(accuracy_cnt) / len(x_test)))


def load_train_mini_batch(batch_size, flatten=True, normalize=True):
    (x_train, t_train), _ = \
            load_mnist(flatten, normalize)

    batch_mask = np.random_choice(x_train.shape[0])
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]

    return x_train_batch, t_train_batch
