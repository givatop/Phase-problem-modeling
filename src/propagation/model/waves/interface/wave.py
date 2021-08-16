from typing import Callable

import numpy as np
from skimage.restoration import unwrap_phase

from ...areas.aperture import Aperture
from ...areas.grid import CartesianGrid
from ...propagation.interface.propagate import Propagable


class Wave(Propagable):
    """
    Базовый класс волны
    """
    def __init__(self, intensity: np.ndarray, phase: np.ndarray, grid: CartesianGrid, wavelength: float):
        self._grid = grid
        self._wavelength = wavelength
        self._intensity, self._phase = intensity, phase
        self._field = np.sqrt(intensity) * np.exp(-1j * phase)

    def propagate_on_distance(self, z: float, method: Callable, **kwargs):
        method(self, z, **kwargs)

    def get_wrapped_phase(self, aperture: Aperture) -> np.ndarray:
        """
        Возвращает неразвернутую фазу волны
        :param aperture: апертура (circ) для обрезания поля
        :rtype: Aperture
        :return: матрица значений фаз
        """
        if aperture is None:
            return self._phase
        return self._phase * aperture.aperture_view

    def get_unwrapped_phase(self, aperture: Aperture) -> np.ndarray:
        """
        Возвращает развернутую фазу волны
        :param aperture: апертура (circ) для обрезания поля
        :return: матрица значений фаз
        """
        if aperture is None:
            return unwrap_phase(self._phase)
        return unwrap_phase(self._phase * aperture.aperture_view)

    @property
    def field(self) -> np.ndarray:
        return self._field

    @field.setter
    def field(self, field):
        self._field = field
        self._phase = np.angle(field)
        self._intensity = np.abs(field) ** 2

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    @property
    def grid(self) -> CartesianGrid:
        return self._grid

    @property
    def wavelength(self) -> float:
        return self._wavelength

