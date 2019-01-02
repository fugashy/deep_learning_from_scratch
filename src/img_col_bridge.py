# -*- coding: utf-8 -*-
import numpy as np


class ImgColBridge:
    u"""
    用いるフィルターに合わせて画像を展開する
    または、展開済みのデータを画像に戻す
    """
    def __init__(self, filter_h, filter_w, shape, stride=1, pad=0):
        u"""
        Args:
            filter_h: フィルターの高さ(int)
            filter_w: フィルターの幅(int)
            shape:    入出力するデータの形状(データ数、ch, height, width) (tuple)
            stride:   フィルター移動間隔(int)
            pad:      パディングパラメータ(int)
        """
        self.__filter_h = filter_h
        self.__filter_w = filter_w
        self.__shape = shape
        self.__stride = stride
        self.__pad = pad

    def to_col(self, data):
        u"""
        2Dデータに展開する

        Args:
            data: 4Dデータ(データ数、ch, height, width)(numpy.array)
        """
        N, C, H, W = self.__shape
        out_h = (H + 2*self.__pad - self.__filter_h)//self.__stride + 1
        out_w = (W + 2*self.__pad - self.__filter_w)//self.__stride + 1

        img = np.pad(
                data,
                [(0,0), (0,0),
                 (self.__pad, self.__pad),
                 (self.__pad, self.__pad)],
                 'constant')
        col = np.zeros((N, C, self.__filter_h, self.__filter_w, out_h, out_w))

        for y in range(self.__filter_h):
            y_max = y + (self.__stride * out_h)
            for x in range(self.__filter_w):
                x_max = x + (self.__stride * out_w)
                col[:, :, y, x, :, :] = \
                    img[:, :, y:y_max:self.__stride, x:x_max:self.__stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def to_img(self, col):
        N, C, H, W = self.__shape
        out_h = (H + 2*self.__pad - self.__filter_h)//self.__stride + 1
        out_w = (W + 2*self.__pad - self.__filter_w)//self.__stride + 1
        col = col.reshape(
                N, out_h, out_w, C, self.__filter_h, self.__filter_w).transpose(
                        0, 3, 4, 5, 1, 2)

        img = np.zeros((
            N,
            C,
            H + (2 * self.__pad) + self.__stride - 1,
            W + (2 * self.__pad) + self.__stride - 1))
        for y in range(self.__filter_h):
            y_max = y + self.__stride*out_h
            for x in range(self.__filter_w):
                x_max = x + self.__stride*out_w
                img[:, :, y:y_max:self.__stride, x:x_max:self.__stride] += col[:, :, y, x, :, :]

        return img[:, :, self.__pad:H + self.__pad, self.__pad:W + self.__pad]
