# -*- coding: utf-8 -*-
import sys, os
import pathlib
p = pathlib.Path(os.path.dirname(__file__) + '/..')
root_dir_path = str(p.resolve())
sys.path.append(root_dir_path)

from src import plotters

def create(config_dict, params):
    acc_plotter = None
    loss_plotter = None
    param_plotter = None
    if 'acc' in config_dict and  config_dict['acc']:
        acc_plotter = plotters.SimplePlotter(
                'Accuracy', label=('epoch', 'accuracy[%]'), y_range=(-0.1, 1.1))
    if 'loss' in config_dict and  config_dict['loss']:
        loss_plotter = plotters.SimplePlotter(
                'Loss', label=('num', 'loss'), y_range=(-0.1, 1.1))
    if 'dnn_params' in config_dict and  config_dict['dnn_params']:
        param_plotter = plotters.NeuralNetworkParamVisualizer(params)

    return Viewer(acc_plotter, loss_plotter, param_plotter)

class Viewer:
    def __init__(self, acc_plotter, loss_plotter, param_plotter):
        self.__acc_plotter = acc_plotter
        self.__loss_plotter = loss_plotter
        self.__param_plotter = param_plotter

    def view(self, train_acc_list, train_loss_list, test_acc_list):
        if self.__param_plotter is not None:
            self.__param_plotter.plot()

        if self.__acc_plotter is not None:
            self.__acc_plotter.plot([train_acc_list, test_acc_list])

        if self.__loss_plotter is not None:
            self.__loss_plotter.plot(train_loss_list)
