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
    :param clip: вырезать от left до right (default один период)
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


def cos_1d(
        x: np.ndarray,
        a: Union[float, int] = 1.,
        x0: Union[float, int] = 0.,
        y0: Union[float, int] = 0.,
        T: Union[float, int] = 2 * np.pi,
        **kwargs
) -> np.ndarray:
    """
    1-мерная косинусоида
    :param x: координатная сетка
    :param a: амплитуда
    :param x0: смещение по оси X
    :param y0: смещение по оси Y
    :param T: период
    :param clip: вырезать от left до right (default один период)
    :param left: default 0
    :param right: default T (вырезать полпериода right = T/2)
    :return:
    """
    result = a * np.cos( (x - x0) / (T / (2 * np.pi)) )

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


def semicircle(
    x: np.ndarray,
    r: Union[int, float] = 1,
    sag: Union[int, float] = None,
    x0: Union[int, float] = 0,
    y0: Union[int, float] = 0,
    inverse: Union[bool, int] = False,
):
    """
    Полуокружность
    :param x:
    :param r:
    :param sag: стрелка прогиба
    :param x0:
    :param y0:
    :param inverse:
    :return:
    """
    if sag > r:
        raise ValueError(f'sag {sag} greater than radius {r}')
    if r <= 0:
        raise ValueError(f'radius <= zero')

    # маска, чтобы отсечь nan
    mask = (r ** 2 - (x - x0) ** 2) < 0
    # уравнение полуокружности
    semicircle = np.sqrt(r ** 2 - (x - x0) ** 2)
    # сдвиг по оси y, чтобы получить нужное значение стрелки прогиба
    if sag:
        semicircle -= r - sag
        semicircle[semicircle < 0] = 0
    # приравниваем все nan к 0
    semicircle[mask] = 0
    # сдвиг по оси y
    semicircle += y0

    if inverse:
        semicircle = -semicircle

    return semicircle


def hemisphere(
    x: np.ndarray,
    y: np.ndarray,
    r: Union[int, float] = 1,
    sag: Union[int, float] = None,
    x0: Union[int, float] = 0,
    y0: Union[int, float] = 0,
    z0: Union[int, float] = 0,
    inverse: Union[bool, int] = False,
):
    """
    Полусфера
    :param x:
    :param y:
    :param r:
    :param sag: стрелка прогиба
    :param x0:
    :param y0:
    :param z0:
    :param inverse:
    :return:
    """
    if sag > r:
        raise ValueError(f'sag {sag} greater than radius {r}')
    if r <= 0:
        raise ValueError(f'radius <= zero')

    mask = (r ** 2 - (x - x0) ** 2 - (y - y0) ** 2) < 0
    hemisphere = np.sqrt(r ** 2 - (x - x0) ** 2 - (y - y0) ** 2)

    if sag:
        hemisphere -= r - sag
        hemisphere[hemisphere < 0] = 0

    hemisphere[mask] = 0
    hemisphere += z0

    if inverse:
        hemisphere = -hemisphere

    return hemisphere


def lens_1d(x, focus, wavelength, converge=True):
    k = 2 * np.pi / wavelength
    lens = k * np.sqrt(x ** 2 + focus ** 2)
    if converge:
        lens *= -1
    return lens


def lens_2d(x, y, focus, wavelength, converge=True):
    k = 2 * np.pi / wavelength
    lens = k * np.sqrt(x ** 2 + y ** 2 + focus ** 2)
    if converge:
        lens *= -1
    return lens


def add_tilt(x, y, complex_amplitude, wavelength, alpha, theta=0):
    """
    Added tilt to complex amplitude
    David Voelz. Computational Fourier Optics. A MATLAB® Tutorial. SPIE PRESS. p.89
    :param complex_amplitude: initial field
    :param wavelength: [meter]
    :param alpha: tilt angle [rad]
    :param theta: rotation along XY angle [rad]
    :return:
    """
    k = 2 * np.pi / wavelength
    tilt = (x * np.cos(theta) + y * np.sin(theta)) * np.tan(alpha)
    return complex_amplitude * np.exp(1j * k * tilt)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    MODE = 1  # 1D | 2D
    SHIFT_GRID = 0  # =1 то независимо от значений (x0; y0) центр функции будет в (0, 0)

    # Параметры исходной функции
    a = 0.5
    x0 = 2.5
    y0 = 7
    z0 = 0
    wx = 1.
    wy = 1.

    # Параметры сеток
    xleft = -10
    xright = -xleft
    xnum = 1000
    dx = 1 / xnum

    yleft = xleft
    yright = -yleft
    ynum = 1000
    dy = dx

    # fig = plt.figure(figsize=(12, 5))

    # 1-мерный график
    if MODE == 1:
        ax = plt.axes()
        # x = np.linspace(xleft, xright, xnum, endpoint=False)
        x = np.arange(xleft, xright, dx)
        # f = lambda _x: triangle_1d(_x, a=a, x0=x0, w=wx) - a
        # y = f(x)
        T = 1
        y = semicircle(x, r=a, sag=.1, x0=x0, y0=y0, inverse=0)

        if SHIFT_GRID:
            x -= x0
        ax.plot(x, y, label='y=f(x)')
        ax.legend()
        ax.grid()

    # 2-мерный график
    if MODE == 2:
        # ax = plt.axes(projection='3d')
        Y, X = np.mgrid[yleft:yright:ynum * 1j, xleft:xright:xnum * 1j]

        f = hemisphere(X, Y, r=a, x0=x0, y0=y0, z0=z0)

        if SHIFT_GRID:
            X -= x0
            Y -= y0

        extent = [xleft, xright, yright, yleft]
        plt.imshow(f, cmap='jet', extent=extent)
        plt.grid()
        plt.colorbar()

        # mappable = ax.plot_surface(X, Y, f, cmap="jet", antialiased=0)
        # plt.colorbar(mappable)

    # Настройки графика
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
