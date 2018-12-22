# -*- coding: utf-8 -*-

import numpy as np

def numerical(f, x):
    u"""
    数値解析的な微分を行う
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2. * h)


def numerical_gradient(f, x):
    u"""
    数値解析的に勾配を求める
    """
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad
