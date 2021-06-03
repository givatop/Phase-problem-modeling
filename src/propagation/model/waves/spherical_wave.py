from typing import Tuple

import numpy as np
from skimage.restoration import unwrap_phase

from ..areas.grid import CartesianGrid
from ...model.areas.aperture import Aperture
from ...model.waves.interface.wave import Wave
from ...utils.math import units
from ...utils.math.general import calc_amplitude
from ...utils.math.general import calculate_radius
from ...utils.optic.field import gauss_2d
from ...utils.optic.propagation_methods import angular_spectrum_propagation


class SphericalWave(Wave):
    """ Волна со сферической аберрацией или сходящаяся сферическая волна """

    def __init__(self, coordinate_grid: CartesianGrid, focal_len: float, gaussian_width_param: int, wavelength: float):
        """
        Создание распределения поля на двухмерной координатной сетке
        :param coordinate_grid: двухмерная координатная сетка расчёта распределения поля
        :param focal_len: фокусное расстояние [м]
        :param gaussian_width_param: ширина гауссоиды на уровне интенсивности 1/e^2 [px]
        :param wavelength: длина волны [м]
        """

        self.__coordinate_grid = coordinate_grid
        self.__focal_len = focal_len
        self.__gaussian_width_param = gaussian_width_param
        self.__wavelength = wavelength

        # задание распределения интенсивности волны
        y_grid, x_grid = self.__coordinate_grid.grid
        gaussian_width_param = units.px2m(gaussian_width_param, px_size_m=coordinate_grid.pixel_size)
        self.__intensity = gauss_2d(x_grid, y_grid,
                                    wx=gaussian_width_param / 4,
                                    wy=gaussian_width_param / 4)

        # волновой вектор
        k = 2 * np.pi / self.__wavelength
        # задание распределения комлексной амплитуды поля
        radius_vector = np.sqrt(x_grid ** 2 + y_grid ** 2 + focal_len ** 2)
        self.__field = np.sqrt(self.__intensity) * np.exp(-1j * k * radius_vector)

        # задание распределения фазы волны
        self.__phase = np.angle(self.__field)

    def get_wrapped_phase(self, *, aperture: Aperture = None, z: float = None) -> np.ndarray:
        if (aperture and z) is not None:

            # оптимизация апертуры для правильного разворачивания фазы
            aperture.modify(self, z)

            return self.__phase * aperture.aperture_view
        else:
            return self.__phase

    def get_unwrapped_phase(self, *, aperture: Aperture = None, z: float = None) -> Tuple[np.ndarray, Aperture]:
        if (aperture and z) is not None:

            # оптимизация апертуры для правильного разворачивания фазы
            aperture.modify(self, z)

            return unwrap_phase(self.__phase * aperture.aperture_view), aperture
        else:
            return unwrap_phase(self.__phase), aperture

    def get_wavefront_radius(self, *, aperture: Aperture, z: float) -> float:
        # развернутая фаза, обрезанная апертурой
        cut_phase, new_aperture = self.get_unwrapped_phase(aperture=aperture, z=z)

        # поиск стрелки прогиба
        amplitude = calc_amplitude(cut_phase)
        sagitta = units.rad2mm(amplitude, self.__wavelength)

        # определение радиуса кривизны волнового фронта
        ap_diameter = units.m2mm(new_aperture.aperture_diameter)
        wavefront_radius = calculate_radius(sagitta, ap_diameter)

        return wavefront_radius

    def propagate_on_distance(self, z: float, method=angular_spectrum_propagation, **kwargs):
        method(self, z, **kwargs)

    @property
    def field(self) -> np.ndarray:
        return self.__field

    @field.setter
    def field(self, field):
        self.__field = field
        self.__phase = np.angle(field)
        self.__intensity = np.abs(field) ** 2

    @property
    def coordinate_grid(self) -> CartesianGrid:
        return self.__coordinate_grid

    @coordinate_grid.setter
    def coordinate_grid(self, area):
        self.__coordinate_grid = area

    @property
    def phase(self) -> np.ndarray:
        return self.__phase

    @phase.setter
    def phase(self, phase):
        # todo добавить перерасчет __field
        raise NotImplementedError

    @property
    def intensity(self) -> np.ndarray:
        return self.__intensity

    @intensity.setter
    def intensity(self, intensity):
        # todo добавить перерасчет __field
        raise NotImplementedError

    @property
    def wavelength(self) -> float:
        return self.__wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self.__wavelength = wavelength

    @property
    def focal_len(self) -> float:
        return self.__focal_len

    @focal_len.setter
    def focal_len(self, focal_len):
        self.__focal_len = focal_len

    @property
    def gaussian_width_param(self) -> float:
        return self.__gaussian_width_param

    @gaussian_width_param.setter
    def gaussian_width_param(self, gaussian_width_param):
        self.__gaussian_width_param = gaussian_width_param