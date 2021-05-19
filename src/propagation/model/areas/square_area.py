from typing import Tuple
import numpy as np

from ..areas.interface.area import Area
from ...utils.math import units


class SquareArea(Area):
    """
    Сетка в квадратичных координатах (квадратная матрица)
    """

    def __init__(self, height, width, pixel_size=5.04e-6):
        """
        Создаёт квадратную матрицу
        :param height: высота матрицы [px]
        :param width: ширина матрицы [px]
        :param pixel_size: размер пикселя матрицы [м]
        """
        self.__height = height
        self.__width = width
        self.__pixel_size = pixel_size

    @property
    def coordinate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        y_grid_array, x_grid_array = np.mgrid[-self.__height / 2:self.__height / 2, -self.__width / 2:self.__width / 2]
        y_grid_array, x_grid_array = (units.px2m(y_grid_array, px_size_m=self.__pixel_size),
                                      units.px2m(x_grid_array, px_size_m=self.__pixel_size))
        return y_grid_array, x_grid_array

    @property
    def frequency_grid(self):
        nu_x = np.arange(-self.__width / 2, self.__width / 2) / (self.__width * self.__pixel_size)
        nu_y = np.arange(-self.__height / 2, self.__height / 2) / (self.__height * self.__pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
        return nu_y_grid, nu_x_grid

    @property
    def pixel_size(self) -> float:
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        self.__pixel_size = pixel_size

    @property
    def height(self) -> float:
        return self.__height

    @height.setter
    def height(self, height):
        self.__height = height

    @property
    def width(self) -> float:
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width


