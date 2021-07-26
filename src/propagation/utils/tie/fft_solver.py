import numpy as np

from typing import Tuple
from numpy.fft import fftshift
from abc import ABC, abstractmethod
from typing import List, Union

from src.propagation.presenter_depr.loader.loader import load_files
from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, apply_volkov_scheme
from src.propagation.utils.math.derivative.finite_difference import central_2point
from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, clip
from src.propagation.model.areas.grid import CartesianGrid
from src.propagation.utils.math.derivative.fourier import gradient_2d, ilaplacian_2d


class TIESolver(ABC):
    """
    Абстрактный класс для решения TIE
    """

    def __init__(self, paths: Union[List[str], List[np.ndarray]], dz: float, wavelength: Union[float, None], bc: BoundaryConditions):
        """
        :param paths: список с путям к файлам интенсивностей
        :param dz: шаг, м
        :param wavelength: длина волны когерентного излучения, м (None для частично-когерентного случая)
        :param bc: граничные условия
        """

        if len(paths) > 2:
            raise NotImplementedError(f'Expect 2 intensities, instead got {len(paths)}')

        if isinstance(paths[0], str):
            self._intensities = load_files(paths)
        elif isinstance(paths[0], np.ndarray):
            self._intensities = paths
        self._intensities = [apply_volkov_scheme(i, bc) for i in self._intensities]

        self._dz = dz
        self._wavelength = wavelength
        self._boundary_condition = bc

        self._axial_derivative = central_2point(*self._intensities, dz)

    @abstractmethod
    def solve(self, threshold) -> np.ndarray:
        """
        :param threshold:
        :return: unwrapped phase
        """
        pass

    def add_threshold(self, threshold: float):
        """
        Пороговая обработка
        :param threshold:
        :return: Бинарная маска
        """
        if threshold == 0. or 0.0 in self.ref_intensity:
            raise ValueError(f'Нельзя делить на нулевые значения в интенсивности.')

        mask = self.ref_intensity < threshold

        # intensity = self.ref_intensity.copy()  # todo этот менто изменяет опорную интенсивность!!!
        self.ref_intensity[mask] = threshold
        return mask

    @property
    def intensities(self):
        return self._intensities

    @property
    def ref_intensity(self):
        return self._intensities[0]

    @property
    def dz(self):
        return self._dz

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def axial_derivative(self):
        return self._axial_derivative

    @property
    def boundary_condition(self):
        return self._boundary_condition


class FFTSolver(TIESolver):
    """
    Решение TIE методом Фурье.
    D. Paganin and K. A. Nugent, Phys. Rev. Lett. 80, 2586 (1998).
    """

    def __init__(self, paths, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(paths, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx, self._ky = self._get_frequency_coefs()
        # todo метод solve находится вне конструктура чтобы можно было заменить paths без удаления объекта

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength
        eps = 2.2204e-16  # from MatLab 2.2204e-16
        reg_param = eps / self.pixel_size ** 4

        # Умножение на волновое число
        phase = - wave_number * self.axial_derivative

        # Первые Лапласиан и градиент
        phase = ilaplacian_2d(phase, self.kx, self.ky, reg_param, return_spacedomain=False)
        phase_x, phase_y = gradient_2d(phase, phase, self.kx, self.ky, space_domain=False)

        # Деление на опорную интенсивность
        mask = self.add_threshold(threshold)
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity
        phase_x[mask], phase_y[mask] = 0, 0

        # Вторые Лапласиан и градиент
        phase_x, phase_y = gradient_2d(phase_x, phase_y, self.kx, self.ky)
        phase = ilaplacian_2d(phase_x + phase_y, self.kx, self.ky, reg_param)

        phase = clip(phase, self.boundary_condition)
        return phase

    def _get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет частотных коэффициентов
        :return:
        """
        area = CartesianGrid(*self.ref_intensity.shape, self.pixel_size)
        nu_y_grid, nu_x_grid = area.grid

        kx = 1j * 2 * np.pi * fftshift(nu_x_grid)
        ky = 1j * 2 * np.pi * fftshift(nu_y_grid)

        return kx, ky

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky
