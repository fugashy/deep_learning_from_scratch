# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import src.differentiation


def y(ys, y_max=1.1, y_min=-0.1, label=None):
    if type(ys) is not list:
        print('input should be list of float')
        return
    elif len(ys) == 0:
        print('input is empty')
        return

    if type(ys[0]) is list:
        for y in ys:
            plt.plot(y)
    else:
        plt.plot(ys)

    plt.ylim(y_min, y_max)

    if label is not None and type(label) is tuple:
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    plt.show()


def fx(func, x_max=5., x_min=-5, x_step=0.1, y_max=1.1, y_min=-0.1):
  x = np.arange(x_min, x_max, x_step)
  y = func(x)
  plt.plot(x, y)
  plt.ylim(y_min, y_max)
  plt.show()


def gradient(
       func, x0_max=2.5, x0_min=-2., x0_step=0.25,
       x1_max=2.5, x1_min=-2., x1_step=0.25,
       x_max=2., x_min=-2., y_max=2., y_min=-2.):
    x0_elements = np.arange(x0_min, x0_max, x0_step)
    x1_elements = np.arange(x1_min, x1_max, x1_step)

    X, Y = np.meshgrid(x0_elements, x1_elements)
    X = X.flatten()
    Y = Y.flatten()

    y = differentiation.numerical_gradient(func, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, y[0], y[1],  angles="xy",color="#666666")
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


def gradient_descent(func, init_x=np.array([-3., -4.]), lr=0.1, step_num=20):
    x, x_history = differentiation.gradient_descent(
        func, init_x, lr=lr, step_num=step_num)

    plt.plot([-5, 5], [0,0], '--b')
    plt.plot([0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()


class OncePlotter:
    u"""
    シンプルな描画クラス
    画面更新後、すぐに制御を返す
    """
    def __init__(self, label=('x', 'y')):
        u"""
        Args:
            name: ウィンドウ名(str)
            label: ラベル名(tuple of str)
        """
        plt.xlabel(label[0])
        plt.xlabel(label[0])

    def plot_once(self, data):
        u"""
        データを描画する

        Args:
            data: データ(list of value)
        """
        plt.cla()
        plt.plot(data)
        plt.pause(0.01)
