# -*- coding: utf-8 -*-
import numpy as np


# numpyに対応した関数にする
def identity(x):
  return x


def step(x):
  y = x > 0
  return y.astype(np.int)


def sigmoid(x):
  return 1. / (1. + np.exp(-x))


def relu(x):
  return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
