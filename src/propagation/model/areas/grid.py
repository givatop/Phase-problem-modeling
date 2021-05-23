from abc import ABC
from typing import Tuple

import numpy as np

from ...utils.math.units import px2m


class Grid(ABC):
    def __init__(self, height, width, pixel_size):
        self.__height = height
        self.__width = width
        self.__pixel_size = pixel_size

    @property
    def pixel_size(self) -> float:
        """ Размер пикселя матрицы [м] """
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        self.__pixel_size = pixel_size

    @property
    def height(self) -> float:
        """ Высота матрицы [px] """
        return self.__height

    @height.setter
    def height(self, height):
        self.__height = height

    @property
    def width(self) -> float:
        """ Ширина матрицы [px] """
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width


class CoordinateGrid(Grid):
    """ Центрированная сетка в декартовых координатах (квадратная матрица) """

    def __init__(self, height, width, pixel_size=5.04e-6):
        super().__init__(height, width, pixel_size)
        self.__grid = px2m(np.mgrid[
                           -self.height / 2:self.height / 2,
                           -self.width / 2:self.width / 2
                           ])

    @property
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.__grid


class PolarGrid(Grid):
    """ Сетка в радиальных координатах """

    def __init__(self, coordinate_grid: CoordinateGrid):
        """ Создаёт сетку  в полярных координатах на основе сетки в квадратичных координатах """
        super().__init__(coordinate_grid.height, coordinate_grid.width, coordinate_grid.pixel_size)
        self.__polar_grid = np.sqrt(sum(map(lambda x: x * x, coordinate_grid.grid)))

    @property
    def polar_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.__polar_grid
