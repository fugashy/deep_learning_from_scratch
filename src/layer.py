# -*- coding: utf-8 -*-
from collections import OrderedDict
import sys, os
sys.path.append(os.pardir)
import numpy as np
import src.activation
import src.loss
import src.img_col_bridge

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

    def get_description(self):
        desc = OrderedDict()
        desc['NONE'] = 'Dynamic Configuable'
        return desc


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

    def get_description(self):
        desc = OrderedDict()
        desc['NONE'] = 'Dynamic Configuable'
        return desc


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
        # テンソル対応
        self.original_x_shape = None
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
        # テンソル対応
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)

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
        # 入力データの形状に戻す（テンソル対応）
        dx = dx.reshape(*self.original_x_shape)

        return dx

    def get_description(self):
        desc = OrderedDict()
        desc['Weight'] = self.W.shape
        desc['bias'] = self.b.shape
        return desc


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

    def get_description(self):
        desc = OrderedDict()
        desc['NONE'] = 'Dynamic Configuable'
        return desc

class BatchNormalization:
    def __init__(self,
            gamma, beta,
            momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    def get_description(self):
        desc = OrderedDict()
        desc['gamma'] = ('value: {}'.format(self.gamma[0]), self.gamma.shape)
        desc['beta'] = ('value: {}'.format(self.beta[0]), self.beta.shape)
        desc['momentum'] = self.momentum
        desc['running_mean'] = 'None' if self.running_mean is None else self.running_mean
        desc['running_var'] = 'None' if self.running_var is None else self.running_var

        return desc


class Dropout:
    u"""
    ニューロンをランダムに消去しながら学習させるための層
    効率的な実装はChainerを見ると良い
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        u"""
        順方向の処理
        伝搬の都度、消去するニューロンはmaskのFalseとして更新・保持される
        """
        if train_flg:
            # 設定値より大きいものはTrue
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

    def get_description(self):
        desc = OrderedDict()
        desc['ratio'] = self.dropout_ratio

        return desc


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.bridge = src.img_col_bridge.ImgColBridge(W.shape[2], W.shape[3])

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, _, FH, FW = self.W.shape
        N, _, H, W = x.shape

        out_h = 1 + int((H + 2*self.bridge.pad - FH) / self.bridge.stride)
        out_w = 1 + int((W + 2*self.bridge.pad - FW) / self.bridge.stride)

        col = self.bridge.to_col(x)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = self.bridge.to_img(dcol, self.x.shape)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.bridge = src.img_col_bridge.ImgColBridge(self.pool_h, self.pool_w, stride, pad)

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.bridge.stride)
        out_w = int(1 + (W - self.pool_w) / self.bridge.stride)

        col = self.bridge.to_col(x)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = self.bridge.to_img(dcol, self.x.shape)

        return dx
