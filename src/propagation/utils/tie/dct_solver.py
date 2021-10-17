from typing import Tuple

from scipy.fftpack import dct, idct
import numpy as np
from numpy.fft import fftfreq
from ..math.derivative.fourier import _dctn, _idctn
from src.propagation.utils.math.derivative.fourier import (
    dct_ilaplacian_2d,
    dct_gradient_2d,
    dct_ilaplacian_1d,
    dct_gradient_1d
)
from src.propagation.utils.tie.boundary_conditions import (
    BoundaryConditions,
    clip
)
from src.propagation.utils.tie.fft_solver import TIESolver


class DCTSolver2D(TIESolver):
    """
    Решение TIE при помощи дискретного косинусного преобразования.
    C. Zuo, Q. Chen, and A. Asundi. Optics Express 22, 9220-9244 (2014).
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, -dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._kx, self._ky = self._get_frequency_coefs()

    def solve(self, threshold):
        wave_number = 2 * np.pi / self.wavelength

        phase = - wave_number * self.axial_derivative

        # 1. Обратный Лапласиан
        zero_k_mask = (self.kx == 0) & (self.ky == 0)
        self.kx[zero_k_mask] = 1
        self.ky[zero_k_mask] = 1
        phase = _dctn(phase)
        phase = phase / (self.kx ** 2 + self.ky ** 2)
        phase[zero_k_mask] = 0
        self.kx[zero_k_mask] = 0
        self.ky[zero_k_mask] = 0

        # 2. Градиенты
        phase_x = _idctn(phase * self.kx)
        phase_y = _idctn(phase * self.ky)

        # 3. Деление на опорную интенсивность
        if 0 in self.ref_intensity:
            raise ValueError(f'Zero value occurred in reference intensity')
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity

        # 4. Градиент
        phase_x = _dctn(phase_x) * self.kx
        phase_y = _dctn(phase_y) * self.ky

        # 5. Обратный Лапласиан
        self.kx[zero_k_mask] = 1
        self.ky[zero_k_mask] = 1
        phase_x = phase_x / (self.kx ** 2 + self.ky ** 2)
        phase_y = phase_y / (self.kx ** 2 + self.ky ** 2)
        phase_x[zero_k_mask] = 0
        phase_y[zero_k_mask] = 0
        self.kx[zero_k_mask] = 0
        self.ky[zero_k_mask] = 0
        phase_x = _idctn(phase_x)
        phase_y = _idctn(phase_y)

        phase = phase_x + phase_y

        phase = clip(phase, self.boundary_condition)
        return phase

    def _get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Расчет частотных коэффициентов """
        h, w = self.ref_intensity.shape
        nu_x = fftfreq(w, d=self.pixel_size)
        nu_y = fftfreq(h, d=self.pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)

        kx = np.pi * nu_x_grid
        ky = np.pi * nu_y_grid

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


class DCTSolver1D(TIESolver):
    """
    Решение TIE для одномерной матрицы при помощи дискретного косинусного преобразования.
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, -dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._k = np.pi * fftfreq(intensities[0].shape[0], self._pixel_size)

    def solve(self, threshold):
        wave_number = 2 * np.pi / self.wavelength

        # Умножение на волновое число
        phase = - wave_number * self.axial_derivative

        # Первые Лапласиан и градиент
        phase = dct_ilaplacian_1d(phase, k=self.k, return_spacedomain=False)
        phase = dct_gradient_1d(phase, k=self.k, space_domain=False)

        # Деление на опорную интенсивность
        mask = self.add_threshold(threshold)
        phase /= self.ref_intensity
        phase[mask] = 0

        # Вторые Лапласиан и градиент
        phase = dct_gradient_1d(phase, k=self.k)
        phase = dct_ilaplacian_1d(phase, k=self.k)

        phase = clip(phase, self.boundary_condition)

        return phase

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def k(self):
        return self._k

