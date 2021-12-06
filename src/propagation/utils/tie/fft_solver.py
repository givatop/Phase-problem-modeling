import numpy as np

from typing import Tuple, Optional, List
from numpy.fft import fftfreq, fft2, ifft2
from abc import ABC, abstractmethod

from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, apply_volkov_scheme
from src.propagation.utils.math.derivative.finite_difference import central_finite_difference
from src.propagation.utils.tie.boundary_conditions import BoundaryConditions, clip
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
            ref_intensity=None
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

        if ref_intensity is None:
            self._ref_intensity = self._intensities[0]
        else:
            self._ref_intensity = ref_intensity

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
        threshold *= np.max(self.ref_intensity)
        mask = self.ref_intensity < threshold
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

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE, ref_intensity=None):
        super().__init__(intensities, dz, wavelength, bc, ref_intensity=ref_intensity)
        self._pixel_size = pixel_size
        self._kx, self._ky = self._get_frequency_coefs()

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
        mask = self.add_threshold(threshold)
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity
        phase_x[mask] = 0
        phase_y[mask] = 0

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
        if intensities[0].ndim > 1:
            raise ValueError(f'not supported ndim = {intensities[0].ndim}. only ndim = 1 supported')
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


if __name__ == '__main__':
    import os
    from src.propagation.utils.math.units import m2mm, mm2m
    from src.propagation.presenter.loader import load_files
    from src.miscellaneous.radius_of_curvature import find_radius

    path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\11. GAUSS\Fresnel'
    fn1 = 'i z=-0.050.txt'
    fn2 = 'i z=0.050.txt'

    dz = 10e-6
    px_size = 5e-6
    wavelength = 555e-9
    threshold = np.exp(-2)

    fp1 = os.path.join(path, fn1)
    fp2 = os.path.join(path, fn2)
    i1, i2 = load_files([fp1, fp2])

    solver = FFTSolver1D([i1, i2], dz, wavelength, px_size)
    phase = solver.solve(threshold)
    radius = find_radius(i2, phase, wavelength, px_size, threshold)

    filename = f'TIE {fn1[:-4]} {fn2[:-4]} dz={m2mm(dz):.3f}mm.npy'
    np.save(os.path.join(path, filename), phase)
