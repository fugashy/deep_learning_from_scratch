# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir)
import numpy as np
import src.mnist as mnist
import src.nn as nn
import src.optimizer as optimizer


def check_gradient():
    (x_train, t_train), _ = mnist.load_mnist(normalize=True, one_hot_label=True)
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    network = nn.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        print(str(np.average(np.abs(grad_backprop[key]))) + ' - ' + str(np.average(np.abs(grad_numerical[key]))))
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ':' + str(diff))


def train_neural_net(
        iter_num=10000, batch_size=100, lr=0.1,
        opt_mod='StochasticGradientDescent'):
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(one_hot_label=True)
    train_size = x_train.shape[0]

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    network = nn.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    param_optimizer = optimizer.AdaGrad()


    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        params = network.params
        param_optimizer.update(params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    return train_loss_list, train_acc_list, test_acc_list