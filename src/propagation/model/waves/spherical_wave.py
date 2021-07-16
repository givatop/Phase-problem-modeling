import numpy as np
from skimage.restoration import unwrap_phase

from ..areas.grid import CartesianGrid
from ...model.areas.aperture import Aperture
from ...model.waves.interface.wave import Wave
from ...utils.math.general import calc_amplitude
from ...utils.math.general import calculate_radius
from ...utils.optic.field import gauss_2d
from ...utils.optic.propagation_methods import angular_spectrum_propagation

from ...utils.math.units import (
    px2m,
    m2mm,
    rad2mm
)


class SphericalWave(Wave):
    """ Волна со сферической аберрацией или сходящаяся сферическая волна """

    def __init__(self, grid: CartesianGrid, focal_len: float, gaussian_width_param: int, wavelength: float):
        """
        Создание распределения поля на двухмерной координатной сетке
        :param grid: двухмерная координатная сетка расчёта распределения поля
        :param focal_len: фокусное расстояние [м]
        :param gaussian_width_param: ширина гауссоиды на уровне интенсивности 1/e^2 [px]
        :param wavelength: длина волны [м]
        """

        self._grid = grid
        self._focal_len = focal_len
        self._gaussian_width_param = gaussian_width_param
        self._wavelength = wavelength

        # задание распределения интенсивности волны
        y_grid, x_grid = self._grid.grid
        gaussian_width_param = px2m(gaussian_width_param, px_size_m=grid.pixel_size)
        self._intensity = gauss_2d(x_grid, y_grid,
                                   wx=gaussian_width_param / 4,
                                   wy=gaussian_width_param / 4)

        # волновой вектор
        k = 2 * np.pi / self._wavelength
        # задание распределения комлексной амплитуды поля
        radius_vector = np.sqrt(x_grid ** 2 + y_grid ** 2 + focal_len ** 2)
        self._field = np.sqrt(self._intensity) * np.exp(-1j * k * radius_vector)

        # задание распределения фазы волны
        self._phase = np.angle(self._field)

    def get_wrapped_phase(self, aperture: Aperture) -> np.ndarray:
        if aperture is None:
            return self._phase
        return self._phase * aperture.aperture_view

    def get_unwrapped_phase(self, aperture: Aperture) -> np.ndarray:
        if aperture is None:
            return unwrap_phase(self._phase)
        return unwrap_phase(self._phase * aperture.aperture_view)

    def get_wavefront_radius(self, aperture: Aperture) -> float:
        """
        Расчитывает радиус кривизны волнового фронта
        :param aperture:
        :return: radius, mm
        """
        # развернутая фаза, обрезанная апертурой
        cut_phase = self.get_unwrapped_phase(aperture=aperture)

        # поиск стрелки прогиба
        amplitude = calc_amplitude(cut_phase)
        sagitta = rad2mm(amplitude, self._wavelength)

        # определение радиуса кривизны волнового фронта
        ap_diameter = m2mm(aperture.aperture_diameter)
        wavefront_radius = calculate_radius(sagitta, ap_diameter)

        return wavefront_radius

    def propagate_on_distance(self, z: float, method=angular_spectrum_propagation, **kwargs):
        method(self, z, **kwargs)

    @property
    def field(self) -> np.ndarray:
        return self._field

    @field.setter
    def field(self, field):
        self._field = field
        self._phase = np.angle(field)
        self._intensity = np.abs(field) ** 2

    @property
    def grid(self) -> CartesianGrid:
        return self._grid

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @property
    def focal_len(self) -> float:
        return self._focal_len

    @property
    def gaussian_width_param(self) -> float:
        return self._gaussian_width_param
