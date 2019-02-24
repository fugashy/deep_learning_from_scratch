# -*- coding: utf-8 -*-
import sys, os
import pathlib
p = pathlib.Path(os.path.dirname(__file__) + '/..')
root_dir_path = str(p.resolve())
sys.path.append(root_dir_path)

import numpy as np


class Trainer:
    def __init__(self, network, optimizer,
                 x_train, t_train, x_test, t_test, epochs, mini_batch_size,
                 evaluate_sample_num_per_epoch=None, epoch_callback=None,
                 verbose=True):
        u"""
        ニューラルネットワークの訓練を行う

        Args:
            network: 学習させるネットワーク
            optimizer: パラメータ更新器
            x_train: 学習データ(numpy.array)
            t_train: 正解データ(numpy.array)
            x_test: テストデータ(numpy.array)
            t_test: テスト正解データ(numpy.array)
            epochs: エポック数(int)
            mini_batch_size: ミニバッチサイズ(int)
            evaluate_sample_num_per_epoch:
            epoch_callback: 1エポック終了時に呼ぶコールバック(callable)
            verbose: データ毎に損失を表示する(bool)
        """
        self.__network = network
        self.__optimizer = optimizer
        self.__x_train = x_train
        self.__t_train = t_train
        self.__x_test = x_test
        self.__t_test = t_test
        self.__epochs = epochs
        self.__batch_size = mini_batch_size
        self.__evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.__epoch_callback = epoch_callback
        self.__verbose = verbose

        # 学習データ数 / バッチ数 = 学習回数 [N / epoch]
        self.__iter_per_epoch = max(self.__x_train.shape[0] / mini_batch_size, 1)
        # 最大学習回数 = エポック数 x 学習回数 [N]
        self.__max_iter = int(epochs * self.__iter_per_epoch)

        self.__current_iter = 0
        self.__current_epoch = 0

        self.__train_loss_list = []
        self.__train_acc_list = []
        self.__test_acc_list = []

        print('size of data: {}'.format(self.__x_train.shape[0]))
        print('size of batch: {}'.format(self.__batch_size))
        print('number of iteration per epoch: {}'.format(self.__iter_per_epoch))
        print('number of epoch: {}'.format(self.__epochs))
        print('number of max_iteration: {}'.format(self.__max_iter))

    def train(self):
        u"""
        学習サイクルを回す

        Returns:
            なし
        """
        for i in range(self.__max_iter):
            self.__one_cycle()

        if self.__verbose:
            print('\nfinal testing...')

        test_acc = self.__network.accuracy(self.__x_test, self.__t_test)

        if self.__verbose:
            print("final accuracy: {}".format(test_acc))

    def __one_cycle(self):
        u"""
        学習の1サイクル

        Returns:
            なし
        """
        # テストデータからランダムでbatch数だけ選択
        batch_mask = np.random.choice(self.__x_train.shape[0], self.__batch_size)
        x_batch = self.__x_train[batch_mask]
        t_batch = self.__t_train[batch_mask]

        # forward
        grads = self.__network.gradient(x_batch, t_batch)
        self.__optimizer.update(self.__network.params, grads)

        # back-propagation
        loss = self.__network.loss(x_batch, t_batch)
        self.__train_loss_list.append(loss)
        if self.__verbose:
            print('\r[{0}] current loss :{1}'.format(self.__current_iter, loss), end="")

        # 1エポック分学習が済んだらテストデータでaccを求める
        if self.__current_iter % self.__iter_per_epoch == 0:
            if self.__verbose:
                print('\ntesting...')

            self.__current_epoch += 1

            x_train_sample, t_train_sample = self.__x_train, self.__t_train
            x_test_sample, t_test_sample = self.__x_test, self.__t_test

            # accを求めるのに使用するデータ数を制限する
            if self.__evaluate_sample_num_per_epoch is not None:
                t = self.__evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.__x_train[:t], self.__t_train[:t]
                x_test_sample, t_test_sample = self.__x_test[:t], self.__t_test[:t]

            train_acc = self.__network.accuracy(x_train_sample, t_train_sample)
            self.__train_acc_list.append(train_acc)
            test_acc = self.__network.accuracy(x_test_sample, t_test_sample)
            self.__test_acc_list.append(test_acc)

            if self.__verbose:
                print("epoch: {0}, train acc: {1}, test acc: {2}".format(
                    str(self.__current_epoch), str(train_acc), str(test_acc)))

            if self.__epoch_callback is not None:
                self.__epoch_callback(self.__train_acc_list,
                                      self.__train_loss_list,
                                      self.__test_acc_list)

        self.__current_iter += 1


def create(config_dict, network, optimizer,
           x_train, t_train, x_test, t_test, epoch_callback=None):
    epochs = config_dict['epochs']
    mini_batch_size = config_dict['mini_batch_size']
    evaluate_sample_num_per_epoch = None
    if config_dict['evaluate_sample_num_per_epoch'] > 0:
        evaluate_sample_num_per_epoch = config_dict['evaluate_sample_num_per_epoch']
    verbose = config_dict['verbose']

    return Trainer(network, optimizer,
            x_train, t_train, x_test, t_test,
            epochs, mini_batch_size, evaluate_sample_num_per_epoch,
            epoch_callback, verbose)
