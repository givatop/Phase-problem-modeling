import numpy as np

from typing import Tuple, Optional, List
from numpy.fft import fftshift, fftfreq, fft2, ifft2
from abc import ABC, abstractmethod

from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, apply_volkov_scheme
from src.propagation.utils.math.derivative.finite_difference import central_finite_difference
from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, clip
from src.propagation.model.areas.grid import CartesianGrid, FrequencyGrid
from src.propagation.utils.math.derivative.fourier import gradient_2d, ilaplacian_2d, ilaplacian_1d, gradient_1d


class TIESolver(ABC):
    """
    Абстрактный класс для решения TIE
    """

    def __init__(
            self,
            intensities: List[np.ndarray],
            dz: float,
            wavelength: Optional[float],
            bc: BoundaryConditions,
    ):
        """
        :param intensities: интенсивности
        :param dz: шаг, м
        :param wavelength: длина волны когерентного излучения, м (None для частично-когерентного случая)
        :param bc: граничные условия
        """

        # todo добавить проверку на одинаковоть размеров матриц, иначе конечные суммы падают с неочевидной ошибкой
        if len(intensities) > 2:
            raise NotImplementedError(f'Expect 2 intensities, instead got {len(intensities)}')

        self._boundary_condition = bc
        self._intensities = tuple(apply_volkov_scheme(i, bc) for i in intensities)
        self._dz = dz
        self._wavelength = wavelength
        self._axial_derivative = central_finite_difference(self._intensities, dz/2)

        self._ref_intensity = self._intensities[0]

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
        if threshold == 0. and 0.0 in self.ref_intensity:
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
        return self._ref_intensity

    @ref_intensity.setter
    def ref_intensity(self, value):
        self._ref_intensity = value

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


class FFTSolver2D(TIESolver):
    """
    Решение TIE методом Фурье.
    D. Paganin and K. A. Nugent, Phys. Rev. Lett. 80, 2586 (1998).
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx, self._ky = self._get_frequency_coefs()
        # todo метод solve находится вне конструктура чтобы можно было заменить intensities без удаления объекта

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength

        phase = - wave_number * self.axial_derivative

        # 1. Обратный Лапласиан
        zero_k_mask = (self.kx == 0) & (self.ky == 0)
        self.kx[zero_k_mask] = 1. + 0*1j
        self.ky[zero_k_mask] = 1. + 0*1j
        phase = fft2(phase)
        phase = phase / (self.kx ** 2 + self.ky ** 2)
        phase[zero_k_mask] = 0. + 0*1j
        self.kx[zero_k_mask] = 0. + 0*1j
        self.ky[zero_k_mask] = 0. + 0*1j

        # 2. Градиенты
        phase_x = ifft2(phase * self.kx).real
        phase_y = ifft2(phase * self.ky).real

        # 3. Деление на опорную интенсивность
        if 0 in self.ref_intensity:
            raise ValueError(f'Zero value occurred in reference intensity')
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity

        # 4. Градиент
        phase_x = fft2(phase_x) * self.kx
        phase_y = fft2(phase_y) * self.ky

        # 5. Обратный Лапласиан
        self.kx[zero_k_mask] = 1. + 0*1j
        self.ky[zero_k_mask] = 1. + 0*1j
        phase_x = phase_x / (self.kx ** 2 + self.ky ** 2)
        phase_y = phase_y / (self.kx ** 2 + self.ky ** 2)
        phase_x[zero_k_mask] = 0. + 0*1j
        phase_y[zero_k_mask] = 0. + 0*1j
        self.kx[zero_k_mask] = 0. + 0*1j
        self.ky[zero_k_mask] = 0. + 0*1j
        phase_x = ifft2(phase_x).real
        phase_y = ifft2(phase_y).real

        phase = phase_x + phase_y

        # eps = 2.2204e-16  # from MatLab 2.2204e-16
        # reg_param = eps / self.pixel_size ** 4

        # # Умножение на волновое число
        # phase = - wave_number * self.axial_derivative
        #
        # # Первые Лапласиан и градиент
        # phase = ilaplacian_2d(phase, self.kx, self.ky, reg_param, return_spacedomain=False)
        # phase_x, phase_y = gradient_2d(phase, phase, self.kx, self.ky, space_domain=False)
        #
        # # Деление на опорную интенсивность
        # mask = self.add_threshold(threshold)
        # phase_x /= self.ref_intensity
        # phase_y /= self.ref_intensity
        # phase_x[mask], phase_y[mask] = 0, 0
        #
        # # Вторые Лапласиан и градиент
        # phase_x, phase_y = gradient_2d(phase_x, phase_y, self.kx, self.ky)
        # phase = ilaplacian_2d(phase_x + phase_y, self.kx, self.ky, reg_param)

        phase = clip(phase, self.boundary_condition)
        return phase

    def _get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет частотных коэффициентов
        :return:
        """
        h, w = self.ref_intensity.shape
        nu_x = fftfreq(w, d=self.pixel_size)
        nu_y = fftfreq(h, d=self.pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)

        kx = 1j * 2 * np.pi * nu_x_grid
        ky = 1j * 2 * np.pi * nu_y_grid

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


class FFTSolver1D(TIESolver):
    """
    Решение TIE методом Фурье.
    D. Paganin and K. A. Nugent, Phys. Rev. Lett. 80, 2586 (1998).
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx = self._get_frequency_coefs()

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength

        # Умножение на волновое число
        phase = - wave_number * self.axial_derivative

        # Первые Лапласиан и градиент
        phase = ilaplacian_1d(phase, self.kx, return_spacedomain=False)
        phase = gradient_1d(phase, self.kx, space_domain=False)

        # Деление на опорную интенсивность
        mask = self.add_threshold(threshold)
        phase /= self.ref_intensity
        phase[mask] = 0

        # Вторые Лапласиан и градиент
        phase = gradient_1d(phase, self.kx)  # todo убрать 2 лишних fft
        phase = ilaplacian_1d(phase, self.kx)

        phase = clip(phase, self.boundary_condition)
        return phase

    def _get_frequency_coefs(self):
        return 1j * 2 * np.pi * fftfreq(self.ref_intensity.shape[0], d=self.pixel_size)

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def kx(self):
        return self._kx

    # @property
    # def ky(self):
    #     return self._ky


class SimplifiedFFTSolver(TIESolver):
    """
    Решение TIE для случая uniform-intensity
    D. Paganin and K. A. Nugent, Phys. Rev. Lett. 80, 2586 (1998).
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx, self._ky = self.get_frequency_coefs()

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength
        eps = 2.2204e-16  # from MatLab 2.2204e-16
        reg_param = eps / self.pixel_size ** 4

        # Умножение на волновое число
        phase = -wave_number * self.axial_derivative
        phase /= self.ref_intensity
        phase = ilaplacian_2d(phase, self.kx, self.ky, reg_param)

        phase = clip(phase, self.boundary_condition)
        return phase

    def get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет частотных коэффициентов
        :return:
        """
        raise NotImplementedError
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


class SimplifiedFFTSolver1D(TIESolver):
    """
    Одномерное решение для равномерной интенстивности
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        if bc != BoundaryConditions.NONE:
            raise NotImplementedError(f'Граничные условия для одномерного случая не реализованы!')
        super().__init__(intensities, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx, _ = self.get_frequency_coefs()

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength
        eps = 2.2204e-16  # from MatLab 2.2204e-16
        reg_param = eps / self.pixel_size ** 4

        # Умножение на волновое число
        phase = -wave_number * self.axial_derivative
        phase /= self.ref_intensity
        phase = ilaplacian_1d(phase, self.kx, reg_param)

        phase = clip(phase, self.boundary_condition)
        return phase

    def get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет частотных коэффициентов
        :return:
        """
        raise NotImplementedError
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


if __name__ == '__main__':
    import os

    path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\2. FFT 2D\i=rect0.5 phi=sphere propagation'
    fn1 = 'i z=-0.005.npy'
    fn2 = 'i z=0.005.npy'

    dz = 10e-6
    px_size = 5e-6
    wavelength = 555e-9
    threshold = 0.1

    i1 = np.load(os.path.join(path, fn1))
    i2 = np.load(os.path.join(path, fn2))

    solver = FFTSolver2D([i1, i2], dz, wavelength, px_size)
    phase = solver.solve(threshold)
    np.save(os.path.join(path, f'TIE phase.npy'), phase)
