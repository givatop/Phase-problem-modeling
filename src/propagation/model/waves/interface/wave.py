from abc import ABC, abstractmethod

import numpy as np

from ...areas.aperture import Aperture
from ...areas.grid import CartesianGrid
from ...propagation.interface.propagate import Propagable


class Wave(Propagable, ABC):
    """
    Интерфейс волны
    """

    @abstractmethod
    def get_wrapped_phase(self, aperture: Aperture) -> np.ndarray:
        """
        Возвращает неразвернутую фазу волны
        :param aperture: апертура (circ) для обрезания поля
        :rtype: Aperture
        :return: матрица значений фаз
        """
        pass

    @abstractmethod
    def get_unwrapped_phase(self, aperture: Aperture) -> np.ndarray:
        """
        Возвращает развернутую фазу волны
        :param aperture: апертура (circ) для обрезания поля
        :return: матрица значений фаз
        """
        pass

    @abstractmethod
    def get_wavefront_radius(self, aperture: Aperture) -> float:
        """
        Возвращает радиус волнового фронта, найденный по следующей формуле:
        r = (s / 2) + (l ** 2 / (8 * s))
        s - стрелка прогиба
        l - хорда, являющаяся диаметром апертуры
        :param aperture: апертура (circ) для обрезания поля
        :param z: дистанция, на которую распространилась волна из начала координат
        :return: радиус волнового фронта при заданной обрезающей апертуре
        """
        pass

    @property
    @abstractmethod
    def field(self) -> np.ndarray:
        """
        Распределение поля волны на координатной сетке в комплексной форме
        """
        pass

    @field.setter
    @abstractmethod
    def field(self, field):
        pass

    @property
    @abstractmethod
    def grid(self) -> CartesianGrid:
        """
        Координатная сетка
        """
        pass

    @property
    @abstractmethod
    def phase(self) -> np.ndarray:
        """
        Распределение фазы поля волны
        :return:
        """
        pass

    @property
    @abstractmethod
    def intensity(self) -> np.ndarray:
        """
        Распределение интенсивности поля волны
        :return:
        """
        pass

    @property
    @abstractmethod
    def wavelength(self) -> float:
        """
        Длина волны [м]
        :return:
        """
        pass

    @property
    @abstractmethod
    def focal_len(self) -> float:
        """
        Фокусное расстояние [м]
        :return:
        """
        pass

    # данный метод в дальнейшем нужно изменить на более общий,
    # так как не у всех волн в профиле интенсивности гауссоида
    @property
    @abstractmethod
    def gaussian_width_param(self) -> float:
        """
        Размер гауссоиды на уровне 1/e^2 в [px]
        :return:
        """
        pass
