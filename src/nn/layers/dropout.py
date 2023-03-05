from nn.module.parameters import Parameters
import numpy as np


class Dropout:
    """Реализует dropout

    ---------
    Параметры
    ---------
    p : float (default=0.5)
        Вероятность зануления элемента
    """

    def __init__(self, p=0.5):
        self.p = p

        self.params = Parameters(1)
        self.regime = "Train"
        self.mask = None

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
        if self.regime == "Eval":
            return inpt

        if self.mask is None:
            self.mask = np.random.binomial(1, self.p, size=inpt.shape[1:])

        self.out = inpt * self.mask / (1 - self.p)

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
        if self.regime == "Eval":
            raise RuntimeError("Нельзя посчитать градиенты в режиме оценки")
        input_grads = grads * self.mask / (1 - self.p)
        return input_grads

    def train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"Dropout(p={self.p})"