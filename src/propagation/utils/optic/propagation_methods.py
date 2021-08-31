import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units
from src.propagation.utils.math.general import get_slice
from src.propagation.utils.optic.field import rect_2d


def angular_spectrum_propagation(wave: Wave, z: float, **kwargs):
    """
    Метод распространения (преобразования) волны методом углового спектра
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """
    frequency_grid = kwargs.get('frequency_grid')

    # волновое число
    wave_number = 2 * np.pi / wave.wavelength

    # частотная сетка
    nu_y_grid, nu_x_grid = frequency_grid.grid.y_grid, frequency_grid.grid.x_grid

    # Фурье-образ исходного поля
    field = fft2(wave.field)

    # передаточная функция слоя пространства
    exp_term = np.sqrt(
        1 - (wave.wavelength * nu_x_grid) ** 2 -
        (wave.wavelength * nu_y_grid) ** 2)
    h = np.exp(1j * wave_number * z * exp_term)

    # todo H((1-(Lambda*U).^2-(Lambda*V).^2)<0) = 0; % neglect evanescent wave

    # обратное преобразование Фурье
    wave.field = ifft2(field * h)


def angular_spectrum_bl_propagation(wave: Wave, z: float):
    """
    Распространение (преобразование) волны при помощи band-limited angular spectrum метода
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """

    # Увеличение транспаранта в 2 раза для трансформации линейной свертки в циклическую
    # (периодические граничные условия)
    height = 2 * wave.field.shape[0]  # количество строк матрицы
    width = 2 * wave.field.shape[1]  # количество элеметов в каждой строке матрицы

    # Индексы для "старого" поля
    left = int(width * .25)
    right = int(width * .75)
    top = int(height * .25)
    bottom = int(height * .75)

    # Вписываем "старое" поле в новое
    new_field = np.zeros((height, width), dtype=wave.field.dtype)
    new_field[top:bottom, left:right] = wave.field

    # Сетка в частотной области
    nu_x = np.arange(-width / 2, width / 2) / (width * wave.grid.pixel_size)
    nu_y = np.arange(-height / 2, height / 2) / (height * wave.grid.pixel_size)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
    nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)
    nu_z_grid = np.sqrt(wave.wavelength ** -2 - nu_x_grid ** 2 - nu_y_grid ** 2)
    nu_z_grid[nu_x_grid ** 2 + nu_y_grid ** 2 > wave.wavelength ** -2] = 0

    # Расчет граничных частот U/V_limit
    dnu_x = 1 / (width * wave.grid.pixel_size)
    dnu_y = 1 / (height * wave.grid.pixel_size)
    nu_x_limit = 1 / (np.sqrt((2 * dnu_x * z) ** 2 + 1) * wave.wavelength)
    nu_y_limit = 1 / (np.sqrt((2 * dnu_y * z) ** 2 + 1) * wave.wavelength)

    # Передаточная функция (угловой спектр)
    h_clipper = rect_2d(nu_x_grid, nu_y_grid, wx=2 * nu_x_limit, wy=2 * nu_y_limit)
    h = np.exp(1j * 2 * np.pi * nu_z_grid * z) * h_clipper

    # обратное преобразование Фурье
    wave.field = ifft2(fft2(new_field) * h)[top:bottom, left:right]


def rayleigh_sommerfeld_propagation_1D(wave: Wave, z: float):
    """
    Распространение (преобразование) одномерной волны при помощи интеграла Рэлея-Зоммерфельда
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """
    # todo: добавить класс одномерной волны, переделать метод под интерфейс для графиков
    # Беру сечение волны по одной из координат
    g0 = get_slice(
        wave.field,
        wave.field.shape[0] // 2,
    )[1]
    # Задаю сетку по X
    X = np.arange(-wave.grid.height / 2, wave.grid.height / 2)

    # Расчет волнового числа
    k = 2 * np.pi / wave.wavelength
    # Завожу массив нулей под g(x, z)
    g_z = np.zeros(g0.shape, dtype=complex)

    for g_, x_ in enumerate(X):
        r = np.sqrt((X - x_) ** 2 + z ** 2)  # расчёт r'
        multiplier = _rayleigh_sommerfeld_multiplier(
            z=z, k=k, r=r, wavelength=wave.wavelength
        )  # расчёт множителя под интегралом RS
        result = multiplier * g0  # значение подинетгрального выражения
        one_sum = _rectangle_rule(result, X[0], X[-1], X.shape[0])
        g_z[g_] = one_sum

    return X, g_z


def _rayleigh_sommerfeld_multiplier(z, k, r, wavelength):
    """
    Коэффициент в интеграле Рэлея-Зоммерфельда
    :param z: дистанция распространения волны
    :param k: волновое число
    :param r: радиус распространения волны
    :param wavelength: длина волны
    :return: float
    """
    res = (z * np.exp(1j * k * r)) / (np.sqrt(r ** 3) * 1j * np.sqrt(wavelength))
    return res


def _rectangle_rule(field, a, b, nseg=512):
    """
    Метод интегрирования прямоугольниками
    :param field: волна
    :param a: левая граница прямоугольника
    :param b: правая графница прямоугольника
    :param nseg: количество отсчётов
    :return: float
    """
    dx = (b - a) / nseg
    summa = 0
    for i in range(nseg):
        cur = field[i] * dx
        summa += cur

    return summa


def fresnel(field: np.ndarray, propagate_distance: float,
            wavelenght: float, pixel_size: float) -> np.ndarray:
    """
    Расчет комплексной амплитуды светового поля прошедшей через слой пространства толщиной propagate_distance
    с использованием передаточной функции Френеля
    :param field: array-like
    :param propagate_distance: float z
    :param wavelenght: float lambda
    :param pixel_size: float px_size
    :return: array-like
    """
    raise NotImplementedError("This method not implemented yet")

    height = field.shape[0]
    width = field.shape[1]

    wave_number = 2 * np.pi / wavelenght

    # Сетка в частотной области
    nu_x = np.arange(-width / 2, width / 2) / (width * pixel_size)
    nu_y = np.arange(-height / 2, height / 2) / (height * pixel_size)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
    nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)

    if propagate_distance != 0 and np.abs(propagate_distance) <= 1 / (
            wavelenght * (nu_x.max() ** 2 + nu_y.max() ** 2) ** 2):
        raise ValueError(f'Не выполняется критерий Релея z < 1 / (lamda*(nu_x^2+nu_y^2): '
                         f'{np.abs(propagate_distance)} <= {1 / (wavelength * (nu_x.max() ** 2 + nu_y.max() ** 2) ** 2)}')

    # Фурье-образ исходного поля
    field = fft2(field)

    exp_term = np.sqrt(1 - ((wavelenght * nu_x_grid) ** 2 - (wavelenght * nu_y_grid) ** 2) / 2)
    h = np.exp(1j * wave_number * propagate_distance * exp_term)

    return ifft2(field * h)
