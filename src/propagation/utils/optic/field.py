from typing import Union
import numpy as np


def rect_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную прямоугольную функцию
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ширина
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    return a * (np.abs((x - x0) / w) < 0.5)


def rect_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0.):
    """
    Возвращает 2-мерную прямоугольную функцию
    :param x: np.ndarray 2-мерная координатная сетка по оси X
    :param y: np.ndarray 2-мерная координатная сетка по оси Y
    :param a: Union[float, int] амплитуда
    :param wx: Union[float, int] ширина по оси X
    :param wy: Union[float, int] ширина по оси Y
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :return: np.ndarray
    """
    return a * (rect_1d(x, w=wx, x0=x0) * rect_1d(y, w=wy, x0=y0))


def circ(r, a=1., w=1., r0=0.):
    return a * rect_1d(r, w=w, x0=r0)


def circ_cartesian(x, y, a=1., w=1., x0=0., y0=0.):
    return a * ((np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / w) < 0.5)


def triangle_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную треугольную функцию
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ПОЛУширина
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    return a * (1 - np.abs((x - x0) / w)) * rect_1d(x, w=2 * w, x0=x0)


def triangle_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0.):
    """
    Возвращает 2-мерную прямоугольную функцию
    :param x: np.ndarray 2-мерная координатная сетка по оси X
    :param y: np.ndarray 2-мерная координатная сетка по оси Y
    :param a: Union[float, int] амплитуда
    :param wx: Union[float, int] ПОЛУширина по оси X
    :param wy: Union[float, int] ПОЛУширина по оси Y
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :return: np.ndarray
    """
    raise NotImplementedError("This method not implemented yet")
    # r = np.sqrt( ((x-x0) / wx)**2 + ((y-y0) / wy)**2 )
    # return a * triangle_1d(r)
    return a * (triangle_1d(x, w=wx, x0=x0) * triangle_1d(y, w=wy, x0=y0))


def gauss_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную гауссоиду с явно указанной амплитудой
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ширина (может выступаить как СКО)
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * w ** 2))


def gauss_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0.):
    """
    Возвращает 2-мерную гауссоиду с явно указанной амплитудой
    :param x: np.ndarray 2-мерная координатная сетка по оси X
    :param y: np.ndarray 2-мерная координатная сетка по оси Y
    :param a: Union[float, int] амплитуда
    :param wx: Union[float, int] ширина по оси X (может выступаить как СКО)
    :param wy: Union[float, int] ширина по оси Y (может выступаить как СКО)
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :return: np.ndarray
    """
    return a * np.exp(-((x - x0) ** 2 / (2 * wx ** 2) + (y - y0) ** 2 / (2 * wy ** 2)))


def logistic_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную логистическую функцию: https://en.wikipedia.org/wiki/Logistic_function
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ширина
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    threshold = 70
    precision = 1e-10  # чтобы не хранить значения ~e-31 степени
    """
    Большие значения в степени exp приводят к переполнению, поэтому они отсекаются
    exp(70) = e+30 (30 знаков достаточно)
    exp(709) = e+307
    exp(710) = inf ~ overflow for np.float.64
    """
    k = 10 / w
    exp_term = -k * (x - x0)
    exp_term[exp_term > threshold] = threshold  # Отсечение больших значений в степени
    exp_term[exp_term < -threshold] = -threshold
    res = a / (1 + np.exp(exp_term))
    res[res < precision] = 0  # Отсечение очень маленьких значений в степени
    return res


def sin_1d(
        x: np.ndarray,
        a: Union[float, int] = 1.,
        x0: Union[float, int] = 0.,
        y0: Union[float, int] = 0.,
        T: Union[float, int] = 2 * np.pi,
        **kwargs
) -> np.ndarray:
    """
    1-мерная синусоида
    :param x: координатная сетка
    :param a: амплитуда
    :param x0: смещение по оси X
    :param y0: смещение по оси Y
    :param T: период
    :param clip: вырезать от left до rigth (default один период)
    :param left: default 0
    :param right: default T (вырезать полпериода right = T/2)
    :return:
    """
    result = a * np.sin( (x - x0) / (T / (2 * np.pi)) )

    clip = kwargs.get('clip', False)
    if clip:
        # get boundaries
        left = kwargs.get('left', 0)
        right = kwargs.get('right', T)
        # create masks
        left_mask = x - x0 > left
        right_mask = x - x0 < right
        # multiply masks
        result *= left_mask * right_mask

    return result + y0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    MODE = 1  # 1D | 2D
    SHIFT_GRID = 0  # =1 то независимо от значений (x0; y0) центр функции будет в (0, 0)

    # Параметры исходной функции
    a = 1
    x0 = -1.5
    y0 = a
    wx = 1.
    wy = 1.

    # Параметры сеток
    xleft = -np.pi * 2
    xright = -xleft
    xnum = 1000
    dx = 1 / xnum

    yleft = xleft
    yright = -yleft
    ynum = 1000
    dy = dx

    fig = plt.figure(figsize=(12, 5))

    # 1-мерный график
    if MODE == 1:
        ax = plt.axes()
        # x = np.linspace(xleft, xright, xnum, endpoint=False)
        x = np.arange(xleft, xright, dx)
        # f = lambda _x: triangle_1d(_x, a=a, x0=x0, w=wx) - a
        # y = f(x)
        T = 1
        y = sin_1d(x, a=a, x0=x0, y0=y0, T=T, clip=True, right=T*3)

        if SHIFT_GRID:
            x -= x0
        ax.plot(x, y, label='y=f(x)')
        ax.legend()
        ax.grid()

    # 2-мерный график
    if MODE == 2:
        ax = plt.axes(projection='3d')
        Y, X = np.mgrid[yleft:yright:ynum * 1j, xleft:xright:xnum * 1j]
        # f = logistic_1d(X, w=0, x0=x0)
        f = triangle_2d(X, Y, a, wx=wx, wy=wy, x0=x0, y0=y0)
        # f = defocus(X, Y)
        # f = gauss_2d(X, Y, a, wx=wx, wy=wy, x0=x0, y0=y0) - a
        # f = triangle_1d(X, a, w=wx, x0=x0) - a
        if SHIFT_GRID:
            X -= x0
            Y -= y0

        # from numpy import gradient
        # grads = gradient(f, X[0, :], Y[:, 0])
        # grad_x, grad_y = grads[0], grads[1]

        mappable = ax.plot_surface(X, Y, f, cmap="jet", antialiased=0)
        plt.colorbar(mappable)
        # mappable = ax.plot_surface(X, Y, grad_x, cmap="gray", antialiased=True)
        # plt.colorbar
        # mappable = ax.plot_wireframe(X, Y, grad_y, rcount=20, ccount=20, antialiased=True)
        # plt.colorbar(mappable)
        # ax.view_init(elev=-90, azim=90)  # Вид "сверху"

    # Настройки графика
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
