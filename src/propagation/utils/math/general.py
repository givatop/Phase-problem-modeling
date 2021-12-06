import os
from typing import Any, Union, Tuple
import numpy as np


# todo после вызова функции я изменил матрицу, которую передавал в array и почему-то объект slice-а изменился!!!
def get_slice(array: np.ndarray, index: int, step: int = 1, xslice: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает сечение двумерной матрицы по указанно-му/-й столбцу/строчке с указанным шагом
    :param array: двумерная матрица
    :param index: номер столбца/строки
    :param step: шаг
    :param xslice: строка == True, столбец == False
    :return: (координатная сетка, сформированная с учетом шага; значения)
    """
    values = array[index, ::step] if xslice else array[::step, index]
    args = np.arange(0, values.size * step, step)
    return args, values


def row_slice(array: np.ndarray, row: int, step: int = 1):
    """
    :param array: 2d array
    :param row:
    :param step:
    :return: строчка
    """
    return array[row, ::step]


def col_slice(array: np.ndarray, column: int, step: int = 1):
    """
    :param array: 2d array
    :param column:
    :param step:
    :return: столбец
    """
    return array[::step, column]


def calculate_radius(s: Union[int, float], l: Union[int, float]) -> float:
    """
    Расчитывает значение радиуса окружности по известной стрелке прогиба и хорде.\n
    https://en.wikipedia.org/wiki/Sagitta_(geometry)
    :param s: стрелка прогиба
    :param l: хорда, на которую опирается дуга окружности, образующая стрелку
    :return: радиус окружности
    """
    return (s / 2) + (l ** 2 / (8 * s))


def calculate_sagitta(r, l):
    """Рассчет стрелки прогиба по известным радиусу r и хорде l"""
    return r - np.sqrt(r ** 2 - (l / 2) ** 2)


def calculate_chord(radius, sag):
    """
    Расчет хорды по известному радиусу и стрелке прогиба
    :param radius: радиус окружности
    :param sag: стрелка прогиба
    :return: chord
    """
    return 2 * np.sqrt(2 * radius * sag - sag ** 2)


def calc_amplitude(array):
    return np.abs(np.max(array) - np.min(array))


def normalize(array: np.ndarray, **kwargs) -> np.ndarray:
    """
    Нормирует входной массив в диапазоне от new_min до new_max
    :param array:
    :param kwargs: old_min = min(array), old_max = max(array), new_min = 0., new_max = 1., dtype = np.float64
    :return:
    """
    if array.dtype in (np.complex64, np.complex128, np.csingle, np.cdouble, np.clongdouble):
        raise TypeError(f'Not implemented for complex-valued arrays: array.dtype = {array.dtype}')

    old_min = kwargs.get('old_min', np.min(array))
    old_max = kwargs.get('old_max', np.max(array))
    new_min = kwargs.get('new_min', 0.)
    new_max = kwargs.get('new_max', 1.)
    dtype = kwargs.get('dtype', np.float64)

    if old_max < old_min or new_max < new_min:
        raise ValueError(f'Значения максимумов должны превышать значения минимумов:'
                         f'old_min = {old_min}\nold_max = {old_max}\nnew_min = {new_min}\nnew_max = {new_max}')

    array = (array - old_min) / (old_max - old_min)  # from old_range to 0 ... 1.0
    array = array * (new_max - new_min) + new_min  # from 0 ... 1.0 to new_range

    return np.asarray(array, dtype=dtype)


def energy_center(array):
    """
    Поиск координат жнергетического центра изображения
    :param array: ndarray with ndim = 2
    :return: erow, ecol
    """
    ecol = np.sum(np.sum(array, axis=0) *
                 np.array(np.arange(1, array.shape[1]+1))) / \
                 np.sum(array)

    erow = np.sum(np.sum(array, axis=1) *
                 np.array(np.arange(1, array.shape[0]+1))) / \
                 np.sum(array)

    return erow, ecol


def print_min_max(array: np.ndarray, array_name: str = 'array'):
    print(f'{np.min(array): >10.2e}{np.max(array): >10.2e} - {array_name}')
