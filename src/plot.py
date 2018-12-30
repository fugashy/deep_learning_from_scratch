# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import src.differentiation

class SimplePlotter:
    def __init__(self, name, label=('x', 'y'), y_range=(-0.1, 1.1)):
        self.__fig = plt.figure(name)
        self.__ax = self.__fig.add_subplot(1, 1, 1)

        self.__label = label
        self.__y_range = y_range


    def plot(self, ys, clear=True):
        if type(ys) is not list:
            print('input should be list of float. {}'.format(type(ys)))
            return
        elif len(ys) == 0:
            print('input is empty')
            return

        if clear:
            self.__ax.cla()

        self.__ax.set_xlabel(self.__label[0])
        self.__ax.set_ylabel(self.__label[1])
        self.__ax.set_ylim(self.__y_range[0], self.__y_range[1])

        if type(ys[0]) is list:
            for y in ys:
                self.__ax.plot(y)
        else:
            self.__ax.plot(ys)

        plt.pause(0.01)


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
