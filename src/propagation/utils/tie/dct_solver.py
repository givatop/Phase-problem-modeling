import numpy as np
from src.propagation.utils.math.derivative.fourier import (
    dct_ilaplacian_2d,
    dct_gradient_2d
)
from src.propagation.utils.tie.boundary_conditions import (
    BoundaryConditions,
    clip
)
from src.propagation.utils.tie.fft_solver import TIESolver


class DCTSolver(TIESolver):
    """
    Решение TIE при помощи дискретного косинусного преобразования.
    C. Zuo, Q. Chen, and A. Asundi. Optics Express 22, 9220-9244 (2014).
    """

    def __init__(self, intensities, dz, wavelength, pixel_size, bc=BoundaryConditions.NONE):
        super().__init__(intensities, dz, wavelength, bc)
        self._pixel_size = pixel_size
        self._lambda_mn = self._get_lambda_mn()

    def solve(self, threshold):
        wave_number = 2 * np.pi / self.wavelength

        # Умножение на волновое число
        phase = - wave_number * self.axial_derivative

        # Первые Лапласиан и градиент
        phase = dct_ilaplacian_2d(phase, lambda_mn=self.lambda_mn, return_spacedomain=False)
        phase_x, phase_y = dct_gradient_2d(phase, phase, space_domain=False)

        # Деление на опорную интенсивность
        mask = self.add_threshold(threshold)
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity
        phase_x[mask], phase_y[mask] = 0, 0

        # Вторые Лапласиан и градиент
        phase_x, phase_y = dct_gradient_2d(phase_x, phase_y)
        phase = dct_ilaplacian_2d(phase_x + phase_y, lambda_mn=self.lambda_mn)

        phase = clip(phase, self.boundary_condition)

        return phase

    def _get_lambda_mn(self):
        """ Расчет собственного значения функции Грина """
        return - np.pi ** 2 * \
               (
                       self.ref_intensity[0] ** 2 / (self.ref_intensity[0] * 2) ** 2 +
                       self.ref_intensity[1] ** 2 / (self.ref_intensity[1] * 2) ** 2
               )

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def lambda_mn(self):
        return self._lambda_mn
