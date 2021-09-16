from numpy.fft import fft2, ifft2, fft, ifft
from numpy import ndarray, real
from scipy.fftpack import dct, idct

"""
Псевдо-дифференциальные операторы, реализованные через FFT.
Первоисточник: D. Paganin "Coherent X-Ray Imaging" p.299-300 2006
"""

NORM = None


def gradient_2d(f_x: ndarray,
                f_y: ndarray,
                kx: ndarray,
                ky: ndarray,
                space_domain: bool = True) -> (ndarray, ndarray):
    """
    Возвращает сумму частных производных первого порядка (функция градиента) от функции f,
    посчитанных при помощи двумерного Фурье преобразования.
    :param f_x: array-like двумерная функция
    :param f_y: array-like двумерная функция
    :param kx: частотный коэффициент 1j * 2*np.pi * fftshift(nu_x_grid)
    :param ky: частотный коэффициент 1j * 2*np.pi * fftshift(nu_y_grid)
    :param space_domain:
    :return: array-like градиент от функции f
    """
    if space_domain:
        f_x = fft2(f_x, norm=NORM)
        f_y = fft2(f_y, norm=NORM)

    return real(ifft2(f_x * kx, norm=NORM)), real(ifft2(f_y * ky, norm=NORM))


def gradient_1d(f: ndarray,
                k: ndarray,
                space_domain: bool = True) -> ndarray:
    """
    Градиент через FFT
    :param f: array-like
    :param k: частотный коэффициент 1j * 2*np.pi * fftshift(nu_x_grid)
    :param space_domain:
    :return: array-like градиент от функции f
    """
    if space_domain:
        f = fft(f, norm=NORM)

    return real(ifft(f * k, norm=NORM))


def ilaplacian_2d(f: ndarray,
                  kx: ndarray,
                  ky: ndarray,
                  reg_param: float,
                  return_spacedomain: bool = True
                  ) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f,
    посчитанных при помощи двумерного Фурье преобразования.
    :param f: array-like двумерная функция
    :param kx: частотный коэффициент 1j * 2*np.pi * fftshift(nu_x_grid)
    :param ky: частотный коэффициент 1j * 2*np.pi * fftshift(nu_y_grid)
    :param reg_param: нужен, чтобы избежать деления на ноль
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    res = fft2(f, norm=NORM) * (kx ** 2 + ky ** 2) / (reg_param + (kx ** 2 + ky ** 2) ** 2)

    if return_spacedomain:
        res = real(ifft2(res, norm=NORM))

    return res


def ilaplacian_1d(f: ndarray,
                  kx: ndarray,
                  return_spacedomain: bool = True
                  ) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f,
    посчитанных при помощи Фурье преобразования.
    :param f: array-like двумерная функция
    :param kx: частотный коэффициент 1j * 2*np.pi * fftshift(nu_x_grid)
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    # Create mask
    mask = (kx == 0)
    kx[mask] = 1. + 0 * 1j
    # Spectral Transformation
    res = fft(f, norm=NORM) / kx.T ** 2
    # Correct result array
    res[mask] = 0. + 0 * 1j
    # Correct kx
    kx[mask] = 0. + 0 * 1j

    if return_spacedomain:
        res = ifft(res, norm=NORM).real

    return res


def _dctn(x, norm='ortho'):
    """ Функция для расчета двумерного косинусного преобразования над матрицей 2D DCT"""
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=norm)
    return x


def _idctn(x, norm='ortho'):
    """ Функция для расчета обратного двумерного косинусного преобразования над матрицей 2D DCT"""
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return x


def dct_gradient_2d(
        f_x: ndarray,
        f_y: ndarray,
        space_domain: bool = True
) -> (ndarray, ndarray):
    """
    Возвращает сумму частных производных первого порядка (функция градиента) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f_x: array-like двумерная функция
    :param f_y: array-like двумерная функция
    :param space_domain:
    :return: array-like градиент от функции f
    """
    if space_domain:
        f_x = _dctn(f_x)
        f_y = _dctn(f_y)

    return _idctn(f_x), _idctn(f_y)


def dct_ilaplacian_2d(f: ndarray,
                      lambda_mn: float,
                      return_spacedomain: bool = True
                      ) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f: array-like двумерная функция
    :param lambda_mn: float собственное число функции Грина
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    res = _dctn(f) * lambda_mn

    if return_spacedomain:
        res = _idctn(res)

    return res
