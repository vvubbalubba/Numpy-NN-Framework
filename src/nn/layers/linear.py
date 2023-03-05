import numpy as np
from nn.module.parameters import Parameters

class Linear:
    """Реализует линейный слой сети

    ---------
    Параметры
    ---------
    in_dim : int
        Размер входных данных

    out_dim : int
        Размер данных на выходе из слоя

    bias : bool (default=True)
        Использовать смещение или нет
    """
    def __init__(self, in_dim, out_dim, bias=True):
        self.in_dim = in_dim
        self.hid_dim = out_dim
        self.bias = bias

        self.W = Parameters((in_dim, out_dim))
        self.W.init_params()

        self.b = Parameters(out_dim)

        self.inpt = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_out)
            Выход слоя
        """
        self.inpt = inpt

        forward_pass = np.dot(inpt, self.W.params)
        if self.bias:
            forward_pass += self.b.params

        return forward_pass

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.W, self.b)

    def zero_grad(self):
        """Обнуляет градиенты модели"""
        self.W.grads = np.zeros(self.W.shape)
        self.b.grads = np.zeros(self.b.shape)

    def compute_gradients(self, grads):
        """Считает градиенты модели"""

        self.W.grads = np.dot(self.inpt.T, grads)
        if self.bias:
            self.b.grads = np.sum(grads, axis=0)
        input_grads = np.dot(grads, self.W.params.T)

        return input_grads

    def train(self):
        """Переводит модель в режим обучения"""
        pass

    def eval(self):
        """Переводит модель в режим оценивания"""
        pass

    def __repr__(self):
        return "Linear({}, {}, bias={})".format(self.in_dim, self.hid_dim,
                                                self.bias)