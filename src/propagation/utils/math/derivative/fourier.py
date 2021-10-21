import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft
from numpy import ndarray, real
from scipy.fftpack import dct, idct
from numpy.linalg import inv

"""
Псевдо-дифференциальные операторы, реализованные через FFT.
Первоисточник: D. Paganin "Coherent X-Ray Imaging" p.299-300 2006
"""


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
        f_x = fft2(f_x)
        f_y = fft2(f_y)

    return real(ifft2(f_x * kx)), real(ifft2(f_y * ky))


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
        f = fft(f)

    return real(ifft(f * k))


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
    res = fft2(f) * (kx ** 2 + ky ** 2) / (reg_param + (kx ** 2 + ky ** 2) ** 2)

    if return_spacedomain:
        res = real(ifft2(res))

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
    res = fft(f) / kx ** 2
    # Correct result array
    res[mask] = 0. + 0 * 1j
    # Correct kx
    kx[mask] = 0. + 0 * 1j

    if return_spacedomain:
        res = ifft(res).real

    return res


def _dctn(x, norm='ortho'):
    """ Функция для расчета двумерного косинусного преобразования над матрицей 2D DCT"""
    return dct(dct(x.T, norm=norm).T, norm=norm)


def _idctn(x, norm='ortho'):
    """ Функция для расчета обратного двумерного косинусного преобразования над матрицей 2D DCT"""
    return idct(idct(x.T, norm=norm).T, norm=norm)


def dct_gradient_2d(
        f_x: ndarray,
        f_y: ndarray,
        kx: ndarray,
        ky: ndarray,
        space_domain: bool = True
) -> (ndarray, ndarray):
    """
    Возвращает сумму частных производных первого порядка (функция градиента) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f_x: array-like двумерная функция
    :param f_y: array-like двумерная функция
    :param kx: частотный коэффициент np.pi * fftshift(nu_x_grid)
    :param ky: частотный коэффициент np.pi * fftshift(nu_y_grid)
    :param space_domain:
    :return: array-like градиент от функции f
    """
    if space_domain:
        f_x = _dctn(f_x)
        f_y = _dctn(f_y)

    return _idctn(f_x * kx), _idctn(f_y * ky)


def dct_ilaplacian_2d(f: ndarray,
                      kx: ndarray,
                      ky: ndarray,
                      reg_param: float,
                      return_spacedomain: bool = True
                      ) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f: array-like двумерная функция
    :param kx: частотный коэффициент np.pi * fftshift(nu_x_grid)
    :param ky: частотный коэффициент np.pi * fftshift(nu_y_grid)
    :param reg_param: нужен, чтобы избежать деления на ноль
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    res = _dctn(f) * (kx ** 2 + ky ** 2) / (reg_param + (kx ** 2 + ky ** 2) ** 2)

    if return_spacedomain:
        res = _idctn(res)

    return res


def dct_gradient_1d(f: ndarray, k: ndarray, space_domain: bool = True) -> ndarray:
    """
    Возвращает сумму частных производных первого порядка (функция градиента) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f: array-like одномерная функция функция
    :param k: float частотная сетка
    :param space_domain:
    :return: array-like градиент от функции f
    """
    if space_domain:
        f = dct(f, norm='ortho')

    return idct(f * k, norm='ortho')


def dct_ilaplacian_1d(f: ndarray,
                      k: ndarray,
                      return_spacedomain: bool = True
                      ) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f,
    посчитанных при помощи двумерного косинусного преобразования.
    :param f: array-like одномерная функция
    :param k: float собственное число функции Грина
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    # Create mask
    mask = (k == 0)
    k[mask] = 1
    # Spectral Transformation
    res = dct(f, norm='ortho') / k ** 2
    # Correct result array
    res[mask] = 0
    # Correct lambda
    k[mask] = 0

    if return_spacedomain:
        res = idct(res, norm='ortho')

    return res


