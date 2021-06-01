from abc import ABC
from typing import Tuple

import numpy as np
from numpy.fft import fftshift

from ...utils.math.units import px2m


class Grid(ABC):
    def __init__(self, height, width, pixel_size):
        self._height = height
        self._width = width
        self._pixel_size = pixel_size

    @property
    def pixel_size(self) -> float:
        """ Размер пикселя матрицы [м] """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        self._pixel_size = pixel_size

    @property
    def height(self) -> float:
        """ Высота матрицы [px] """
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def width(self) -> float:
        """ Ширина матрицы [px] """
        return self._width

    @width.setter
    def width(self, width):
        self._width = width


class CartesianGrid(Grid):
    """ Центрированная сетка в декартовых координатах (квадратная матрица) """

    def __init__(self, height, width, pixel_size=5.04e-6):
        super().__init__(height, width, pixel_size)
        self._grid = px2m(np.mgrid[
                          -self.height / 2:self.height / 2,
                          -self.width / 2:self.width / 2
                          ])

    @property
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._grid


class PolarGrid(Grid):
    """ Сетка в радиальных координатах """

    def __init__(self, cart_grid: CartesianGrid):
        """ Созданик сетки в полярных координатах на основе сетки в декартовых координатах """
        super().__init__(cart_grid.height, cart_grid.width, cart_grid.pixel_size)
        self._grid = np.sqrt(sum(map(lambda x: x * x, cart_grid.grid)))

    @property
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._grid


class FrequencyGrid(Grid):
    """ Центрированная частотная сетка в декартовых координатах """

    def __init__(self, cart_grid: CartesianGrid):
        """ Создание частотной сетки на основе сетки в декартовых координатах """
        super().__init__(cart_grid.height, cart_grid.width, cart_grid.pixel_size)

        # создание сетки в частотной области при условии выполнения теоремы Котельникова
        nu_x = np.arange(
            -cart_grid.width / 2, cart_grid.width / 2) / (cart_grid.width * cart_grid.pixel_size)
        nu_y = np.arange(
            -cart_grid.height / 2, cart_grid.height / 2) / (cart_grid.height * cart_grid.pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)

        # сдвиг высоких частот к краям сетки
        self._grid = (fftshift(nu_x_grid), fftshift(nu_y_grid))

    @property
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._grid
