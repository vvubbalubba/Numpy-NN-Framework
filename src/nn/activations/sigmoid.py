from nn.module.parameters import Parameters
import numpy as np


class Sigmoid:
    """Реализует сигмоиду"""

    def __init__(self):
        self.params = Parameters(1)

        self.out = None

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
        output : np.ndarray, shape=(M, N_in)
            Выход слоя
        """
        self.out = 1 / (1 + np.exp(-inpt))

        return self.out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return self.params

    def zero_grad(self):
        """Обнуляет градиенты модели

        Не нужен в данном случае,
        оставим для совместимости
        """
        pass

    def compute_gradients(self, grads):
        """Считает градиенты модели"""
        input_grads = grads * self.out * (1 - self.out)
        return input_grads

    def train(self):
        """Переводит модель в режим обучения"""
        pass

    def eval(self):
        """Переводит модель в режим оценивания"""
        pass

    def __repr__(self):
        return "Sigmoid()"