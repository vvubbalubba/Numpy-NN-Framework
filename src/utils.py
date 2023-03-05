import time
import numpy as np
from nn.loss_functions.mse_loss import mse_loss
from nn.loss_functions.hinge_loss import hinge_loss
from optimization.gd_optimizer import GD
from optimization.adam_optimizer import Adam
from numpy import linalg as LA

def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')

def gradient_check(x, y, neural_net, num_last=1, epsilon=1e-3):

    # параметры модели
    params = list(neural_net.parameters())

    # список для численного градиента
    gradient_approx_vector = []
    # список для реализованного градиента
    gradient_vector = []

    neural_net.train()
    pred = neural_net(x)
    loss = hinge_loss(pred, y)
    loss.backward()

    # цикл по слоям модели
    for i, layer in enumerate(params):
        shape = layer.shape
        if i >= len(params) - num_last:
            # если двумерный слой -> цикл по строкам и столбцам
            if isinstance(shape, tuple):
                for j, row in enumerate(layer.params):
                    for k, item in enumerate(row):
                        item_copy = item
                        # J_plus
                        layer.params[j][k] = item_copy + epsilon
                        pred = neural_net(x)
                        loss_plus = hinge_loss(pred, y)
                        # J_minus
                        layer.params[j][k] = item_copy - epsilon
                        pred = neural_net(x)
                        loss_minus = hinge_loss(pred, y)

                        layer.params[j][k] = item_copy

                        gradient_approx_vector.append((loss_plus.loss - loss_minus.loss) / (2*epsilon))
                        gradient_vector.append(layer.grads[j][k])
            # одномерный слой -> цикл по элементам
            else:
                for j, item in enumerate(layer.params):
                    item_copy = item
                    # J_plus
                    layer.params[j] = item_copy + epsilon
                    pred = neural_net(x)
                    loss_plus = hinge_loss(pred, y)
                    # J_minus
                    layer.params[j] = item_copy - epsilon
                    pred = neural_net(x)
                    loss_minus = hinge_loss(pred, y)

                    layer.params[j] = item_copy

                    gradient_approx_vector.append((loss_plus.loss - loss_minus.loss) / (2*epsilon))
                    gradient_vector.append(layer.grads[j])

    gradient_approx_vector = np.array(gradient_approx_vector)
    gradient_vector = np.array(gradient_vector)

    #рассчитаем итоговое значение
    print(np.linalg.norm(gradient_approx_vector), np.linalg.norm(gradient_vector))

    numerator = np.linalg.norm(gradient_approx_vector - gradient_vector)
    denominator = np.linalg.norm(gradient_approx_vector) + np.linalg.norm(gradient_vector)
    diff = numerator / denominator

    if diff > epsilon:
        print('Backprop is incorrect!')
    else:
        print('Backprop is correct!')
    return diff