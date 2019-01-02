# -*- coding: utf-8 -*-
import numpy as np


class ImgColBridge:
    u"""
    用いるフィルターに合わせて画像を展開する
    または、展開済みのデータを画像に戻す
    """
    def __init__(self, filter_h, filter_w, stride=1, pad=0):
        u"""
        Args:
            filter_h: フィルターの高さ(int)
            filter_w: フィルターの幅(int)
            stride:   フィルター移動間隔(int)
            pad:      パディングパラメータ(int)
        """
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.pad = pad
        self.shape = None

    def to_col(self, data):
        u"""
        2Dデータに展開する

        Args:
            data: 4Dデータ(データ数、ch, height, width)(numpy.array)
        """
        N, C, H, W = data.shape
        self.shape = data.shape

        out_h = (H + 2*self.pad - self.filter_h)//self.stride + 1
        out_w = (W + 2*self.pad - self.filter_w)//self.stride + 1

        img = np.pad(
                data,
                [(0,0), (0,0),
                 (self.pad, self.pad),
                 (self.pad, self.pad)],
                 'constant')
        col = np.zeros((N, C, self.filter_h, self.filter_w, out_h, out_w))

        for y in range(self.filter_h):
            y_max = y + (self.stride * out_h)
            for x in range(self.filter_w):
                x_max = x + (self.stride * out_w)
                col[:, :, y, x, :, :] = \
                    img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def to_img(self, col):
        if self.shape is None:
            raise Exception('Data shape is required.')

        N, C, H, W = self.shape
        out_h = (H + 2*self.pad - self.filter_h)//self.stride + 1
        out_w = (W + 2*self.pad - self.filter_w)//self.stride + 1
        col = col.reshape(
                N, out_h, out_w, C, self.filter_h, self.filter_w).transpose(
                        0, 3, 4, 5, 1, 2)

        img = np.zeros((
            N,
            C,
            H + (2 * self.pad) + self.stride - 1,
            W + (2 * self.pad) + self.stride - 1))
        for y in range(self.filter_h):
            y_max = y + self.stride*out_h
            for x in range(self.filter_w):
                x_max = x + self.stride*out_w
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.pad:H + self.pad, self.pad:W + self.pad]

    def to_img(self, col, shape):
        self.shape = shape
        return to_img(col)
