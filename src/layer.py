# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir)
import numpy as np
import src.activation
import src.loss

u"""
ニューラルネットワークを構成する様々な層の定義
順方向の処理・逆方向の処理(解析的微分計算)をそれぞれ定義する
"""


class Mul:
    u"""
    掛け算レイヤー
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        u"""
        2つの入力をかけて、1つの値として返す
        """
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        u"""
        1つの入力を得、連鎖率に基づいた値をかけて2つの値を返す
        """
        # xyをxについて偏微分したものはy
        dx = dout * self.y
        # xyをyについて偏微分したものはx
        dy = dout * self.x

        return dx, dy


class Add:
    u"""
    足し算レイヤー
    """
    def forward(self, x, y):
        u"""
        順方向の計算
        単純に足し合わせる
        """
        return x + y

    def backward(self, dout):
        u"""
        逆伝搬時の計算
        x + yをxについて偏微分したものは1
        x + yをyについて偏微分したものは1
        dx, dy = 1 * dout, 1 * dout
        """
        dx, dy = dout, dout

        return dx, dx


class Relu:
    u"""
    活性化関数Reluを施すレイヤー
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        u"""
        順方向の計算
        0より大きいかどうかで入力をそのまま返すかどうかを変える
        y = x (x >  0)
          = 0 (x <= 0)

        Args:
            x: 実数ベクトル(np.array)
        """
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        u"""
        逆伝搬時の計算
        dx =  dout (x >  0)
           =     0 (x <= 0)
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    u"""
    活性化関数Sigmoidを施すレイヤー
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        u"""
        順方向の計算
        y = 1 / (1 + e^(-x))
        """
        out = 1. / (1. + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        u"""
        逆伝搬時の計算
        順方向の計算を計算グラフで表現し、出力側から解析的に微分を定義する
        連鎖率によりそれを繋いで最終的な微分計算式を得る
        y = 1 / x ノード
            dy/dx = -1/x^2
                  = -y^2    ...A
        y = x + 1 ノード
            dy/dx = 1       ...B
        y = exp(-x) ノード
            dx/dy = exp(-x) ...C
        y = -1 * x ノード
            dx/dy = -1      ...D
        最終的な微分計算式
            dx/dy = A * B * C * D
                  = -y^2 * 1 * e^(-x) * -1
                  = y^2 * e^(-x)   ...もう少し整理できる
                  = y(1 - y)
        """
        dx = dout * (1. - self.out) * self.out

        return dx


class Affine:
    u"""
    Affine計算を施すレイヤー
    """
    def __init__(self, W, b):
        u"""
        Args:
            W: 重み(np.array)
            b: バイアス(np.array)
        """
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        u"""
        順方向の計算
        ex)
            x = [1, 2]
            W = [1, 2, 3]
                [4, 5, 6]
            b = [1, 2, 3]

        Args:
            x: 入力(np.array)
        """
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        u"""
        逆伝搬時の計算
        これはむずい
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    u"""
    Softmax関数とCEEを組み合わせた処理を施すレイヤー
    出力層として用いる
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        u"""
        順方向の計算
        Softmaxによる正規化と、その結果と教師データの差を計算する

        Args:
            x: 入力ベクトル(np.array)
            t: 教師データ(np.array)

        Returns:
            エラーベクトル(np.array)
        """
        self.t = t
        self.y = src.activation.softmax(x)
        self.loss = src.loss.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        u"""
        逆伝搬時の計算
        計算内容はむずい

        Args:
            dout: 伝搬係数
        """
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
