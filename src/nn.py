# -*- coding: utf-8 -*-
import sys, os
from collections import OrderedDict
sys.path.append(os.pardir)
import numpy as np
import src.activation
import src.differentiation
import src.loss
import src.layer

def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network


def forward(network, x, out_activation):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = src.activation.sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = src.activation.sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = out_activation(a3)

  return y


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = src.activation.softmax(z)
        loss = src.loss.cross_entropy_error(y, t)

        return loss


class TwoLayerNet:
    def __init__(
            self, input_size=2, hidden_size=3,
            output_size=1, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = src.layer.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = src.layer.Relu()
        self.layers['Affine2'] = src.layer.Affine(self.params['W2'], self.params['b2'])

        self.last_layer = src.layer.SoftmaxWithLoss()


    def predict(self, x):
        u"""
        順方向の計算(出力層を除く)

        Args:
            x: データ(np.array)

        Returns:
            推論結果(np.array)
        """
        # 出力層以外の順方向計算を行う
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        u"""
        順方向の計算を走らせて誤差を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            エラーベクトル(np.array)
        """
        # 出力層を除く順方向の計算を行う
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        u"""
        精度を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            推定値が教師データと一致している割合
        """
        # 順方向の処理を一回走らせる
        # 出力層は覗いているが、最大値を見たいだけなので気にしなくてOK
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = src.differentiation.numerical_gradient(
                loss_W, self.params['W1'])
        grads['b1'] = src.differentiation.numerical_gradient(
                loss_W, self.params['b1'])
        grads['W2'] = src.differentiation.numerical_gradient(
                loss_W, self.params['W2'])
        grads['b2'] = src.differentiation.numerical_gradient(
                loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        u"""
        誤差逆伝搬法によって勾配を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            重み・バイアスの勾配(dict of np.array)
        """
        # 順方向の計算を行う
        self.loss(x, t)

        # 微小値について、出力層の逆伝搬処理
        dout = 1
        dout = self.last_layer.backward(dout)

        # 層の順番を反転・逆伝搬
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 重み・バイアスの勾配を取り出す
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

class MultiLayerNet:
    u"""
    任意の層を持つニューラルネットワーク
    """
    def __init__(
            self, input_size, hidden_size_list, output_size,
            activation='relu', weight_init_std=0.01, weight_decay_lambda=0,
            with_batch_norm=True, with_dropout=True, dropout_ratio=0.5):
        u"""
        Args:
            input_size:          入力層の入力要素数(int)
                                 ex) 画像の画素数
            hidden_size_list:    隠れ層のサイズのリスト(list of int)
                                 この数が層の深さを規定する
            output_size:         出力層の出力要素数(int)
                                 ex) mnistの場合は10種の分類問題なので10
            activation:          活性化関数(str)
                                 relu or sigmoid
            weight_init_std:     重み初期化時の標準偏差(float or str)
                                 relu or he or sigmoid or xavier or float value
            weight_decay_lambda: Weight Decay(L2ノルム)の強さ(int)
            with_batch_norm:     BatchNormalizationを行う(bool)
            with_dropout:        Dropoutを用いるかどうか(bool)
            dropout_ratio:       Dropout設定値(float)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.with_batch_norm = with_batch_norm
        self.with_dropout = with_dropout

        self.params = OrderedDict()

        self.__init_weight(weight_init_std)

        act_layers = {'sigmoid': src.layer.Sigmoid,
                      'relu': src.layer.Relu}
        self.layers = OrderedDict()
        # AffineLayerとActivationLayerを隠れ層の数だけ追加する
        for idx in range(1, self.hidden_layer_num + 1):
            # 初期化しておいたパラメータで作る
            self.layers['Affine' + str(idx)] = \
                    src.layer.Affine(self.params['W' + str(idx)],
                                     self.params['b' + str(idx)])
            # BatchNormalization
            if with_batch_norm:
                # パラメータの数は前層を考慮
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])

                self.layers['BatchNorm' + str(idx)] = src.layer.BatchNormalization(
                        self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            if with_dropout:
                self.layers['Dropout' + str(idx)] = src.layer.Dropout(dropout_ratio)

            self.layers['Activation_function' + str(idx)] = \
                    act_layers[activation]()

        # 出力層の前の層
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = \
                src.layer.Affine(self.params['W' + str(idx)],
                                 self.params['b' + str(idx)])

        # 出力層
        self.last_layer = src.layer.SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        u"""
        重みの初期化を行う

        Args:
            weight_init_std: 重みの標準偏差(float)
        """
        # 配列の結合
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        # 層の数だけループ
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                # ReLUを使う場合に推奨される初期値
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                # sigmoidを使う場合に推奨される初期値
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            W = np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['W' + str(idx)] = scale * W
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        u"""
        順方向の計算(出力層を除く)

        Args:
            x: データ(np.array)

        Returns:
            推論結果(np.array)
        """
        # 出力層以外の順方向計算を行う
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        u"""
        順方向の計算を走らせて誤差を求める
        また、荷重減衰も行う
            過学習抑制のために昔からよく用いられる
            大きな重みへペナルティを課す考え方
            （重みが大きくなる時、よく過学習となるため）
            見かけ上の重みを増やすため、重みのL2ノルムを加えて損失を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            エラーベクトル(np.array)
        """
        # 出力層を除く順方向の計算を行う
        y = self.predict(x)

        # 荷重減衰
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        u"""
        精度を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            推定値が教師データと一致している割合
        """
        # 順方向の処理を一回走らせる
        # 出力層は覗いているが、最大値を見たいだけなので気にしなくてOK
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = src.differentiation.numerical_gradient(
                    loss_W, self.params['W' + str(idx)])
            if self.with_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(
                        loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(
                        loss_W, self.params['beta' + str(idx)])
            grads['b' + str(idx)] = src.differentiation.numerical_gradient(
                    loss_W, self.params['b' + str(idx)])


        return grads

    def gradient(self, x, t):
        u"""
        誤差逆伝搬法によって勾配を求める

        Args:
            x: データ(np.array)
            t: 教師データ(np.array)

        Returns:
            重み・バイアスの勾配(dict of np.array)
        """
        # 順方向の計算を行う
        self.loss(x, t)

        # 微小値について、出力層の逆伝搬処理
        dout = 1
        dout = self.last_layer.backward(dout)

        # 層の順番を反転・逆伝搬
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 重み・バイアスの勾配を取り出す
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.layers['Affine' + str(idx)].dW + \
                self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['W' + str(idx)] = W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.with_batch_norm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads

    def show_configulations(self):
        print('Layers')
        for layer_key in self.layers:
            print(layer_key)
            params = self.layers[layer_key].get_description()
            for param_key in params:
                print('\t', sep='', end='')
                print(param_key, params[param_key])
        print('Output layer')
        output_layer_params = self.last_layer.get_description()
        for param_key in output_layer_params:
            print('\t', sep='', end='')
            print(param_key, output_layer_params[param_key])

def create_multilayer_network(config_dict):
    input_size = config_dict['input_size']
    output_size = config_dict['output_size']
    hidden_layer_num = config_dict['hidden_layer_num']
    hidden_size = config_dict['hidden_size']
    activation = config_dict['activation']
    weight_init_std = config_dict['weight_init_std']
    weight_decay_lambda = config_dict['weight_decay_lambda']
    with_batch_norm = config_dict['with_batch_normalization']
    with_dropout = config_dict['dropout']['use']
    dropout_ratio = config_dict['dropout']['ratio']

    hidden_size_list = [hidden_size for i in range(hidden_layer_num + 1)]

    return MultiLayerNet(
            input_size, hidden_size_list, output_size,
            activation=activation, weight_init_std=weight_init_std,
            weight_decay_lambda=weight_decay_lambda,
            with_batch_norm=with_batch_norm,
            with_dropout=with_dropout, dropout_ratio=dropout_ratio)


class SimpleCNN:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num,
                                            input_dim[0],
                                            filter_size,
                                            filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = src.layer.Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = src.layer.Relu()
        self.layers['Pool1'] = src.layer.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = src.layer.Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = src.layer.Relu()
        self.layers['Affine2'] = src.layer.Affine(self.params['W3'], self.params['b3'])

        self.last_layer = src.layer.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        u"""損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]

def create_simple_cnn(config_dict):
    input_dim = tuple(config_dict['input_dim'])
    conv_param = config_dict['conv_param']
    hidden_size = config_dict['hidden_size']
    output_size = config_dict['output_size']
    weight_init_std = config_dict['weight_init_std']

    return SimpleCNN(
            input_dim, conv_param, hidden_size, output_size, weight_init_std)
