import numpy as np
from nn.loss_functions.loss import Loss


def hinge_loss(inpt, target):
    """Реализует функцию ошибки hinge loss

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Список реальных классов
        Одномерный массив

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошибка
    """
    # Мы должны сконвертировать массив реальных меток
    # в двумерный массив размера (N, C),
    # где N -- число элементов
    # С -- число классов
    N = inpt.array.shape[0]

    correct_labels = (range(N), target)
    correct_class_scores = inpt.array[correct_labels]  # Nx1

    loss_element = inpt.array - correct_class_scores[:, np.newaxis] + 1  # NxC
    correct_classifications = np.where(loss_element <= 0)

    loss_element[correct_classifications] = 0
    loss_element[correct_labels] = 0

    grad = np.ones(loss_element.shape, dtype=np.float16)
    grad[correct_classifications], grad[correct_labels] = 0, 0
    grad[correct_labels] = -1 * grad.sum(axis=-1)
    grad /= N

    loss = np.sum(loss_element) / N

    return Loss(loss, grad, inpt.model)
