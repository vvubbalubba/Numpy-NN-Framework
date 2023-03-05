from nn.module.parameters import Parameters
import numpy as np

class BatchNorm:
    """Реализует Batch norm
    ---------
    Параметры
    ---------
    in_dim : int
        Размерность входного вектора
    eps : float (default=1e-5)
        Параметр модели,
        позволяет избежать деления на 0
    momentum : float (default=0.1)
        Параметр модели
        Используется для обновления статистик
    """

    def __init__(self, in_dim, eps=1e-5, momentum=0.1):
        self.in_dim = in_dim
        self.eps = eps
        self.momentum = 0.1

        self.regime = "Train"
        # initialize with kaiming method
        self.gamma = Parameters((in_dim,))
        self.gamma.init_params()
        # initialize with zeros
        self.beta = Parameters(in_dim)

        self.mean = np.zeros(in_dim)
        self.var = np.zeros(in_dim)

        self.inpt_norm = None
        self.tmp_std = None

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
            out = (inpt - self.mean) / np.sqrt(self.mean + self.eps) * self.gamma.params + self.beta.params
            return out

        if self.regime == "Train":
            mean = inpt.mean(axis=0)
            var = inpt.var(axis=0)

            if self.mean.all() == 0 and self.mean.all() == 0:
                self.mean = mean
                self.var = var
            else:
                # exponential smoothing
                self.mean = (1 - self.momentum) * self.mean + self.momentum * mean
                self.var = (1 - self.momentum) * self.var + self.momentum * var

            self.tmp_std = np.sqrt(self.var + self.eps)
            self.inpt_norm = (inpt - mean) / self.tmp_std

            out = self.gamma.params * self.inpt_norm + self.beta.params

            return out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def compute_gradients(self, grads):
        """Считает градиенты модели"""
        if self.regime == "Eval":
            raise RuntimeError("Cannot compute gradients in evaluation mode")

        beta = grads.sum(axis=0)
        gamma = (grads * self.inpt_norm).sum(axis=0)

        dinpt_norm = grads * self.gamma.params
        dinpt_centered = dinpt_norm / self.tmp_std

        N = dinpt_norm.shape[0]
        dmean = -(dinpt_centered.sum(axis=0) + 2 / N * (self.inpt_norm * self.tmp_std).sum(axis=0))
        dstd = (dinpt_norm * (self.inpt_norm * self.tmp_std) * (-self.tmp_std) ** (-2)).sum(axis=0)
        dvar = dstd / 2 / self.tmp_std
        input_grads = dinpt_centered + (dmean + dvar * 2 * (self.inpt_norm * self.tmp_std)) / N

        self.beta.grads = (1 - self.momentum) * beta + self.momentum * beta
        self.gamma.grads = (1 - self.momentum) * gamma + self.momentum * gamma

        return input_grads

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.gamma, self.beta)

    def zero_grad(self):
        """Обнуляет градиенты модели"""
        self.gamma.grads = np.zeros(self.gamma.shape)
        self.beta.grads = np.zeros(self.beta.shape)

    def train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"BatchNorm(in_dim={self.in_dim}, eps={self.eps})"