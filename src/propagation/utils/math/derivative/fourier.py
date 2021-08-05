from numpy.fft import fft2, ifft2, fft, ifft
from numpy import ndarray, real
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
    Возвращает сумму частных производных первого порядка (функция градиента) от функции f.
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


def ilaplacian_2d(f: ndarray,
                  kx: ndarray,
                  ky: ndarray,
                  reg_param: float,
                  return_spacedomain: bool = True) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f.
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
                  return_spacedomain: bool = True) -> ndarray:
    """
    Возвращает сумму частных производных минус второго порядка (обратный Лапласиан) от функции f.
    :param f: array-like двумерная функция
    :param kx: частотный коэффициент 1j * 2*np.pi * fftshift(nu_x_grid)
    :param return_spacedomain:
    :return: array-like градиент от функции f
    """
    # Create mask
    mask = (kx == 0)
    kx[mask] = 1. + 0*1j
    # Spectral Transformation
    res = fft(f, norm=NORM) / kx.T ** 2
    # Correct result array
    res[mask] = 0. + 0*1j
    # Correct kx
    kx[mask] = 0. + 0*1j

    if return_spacedomain:
        res = ifft(res, norm=NORM).real

    return res


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import fft, ifft, ifftshift, fftfreq

    from src.propagation.utils.optic.field import sin_1d, cos_1d, triangle_1d, logistic_1d

    # Координатная сетка
    num = 1024
    dx = 5.06 * 1e-2  # width/num
    width = num * dx
    x = np.linspace(-width//2, width//2, num, endpoint=False)

    # Параметры y(x)
    a = 1
    period = width / 2  # 2 * np.pi
    # x0 = 0
    x0 = -width/8
    left = 0  # np.pi * 3
    right = period*3.5  # period * 4.5
    clip = 0

    # y(x)
    # y = triangle_1d(x, x0=x0, w=period)
    y = logistic_1d(x, x0=x0, w=period) * logistic_1d(-x, x0=x0, w=period)
    # y[0] = y[-1]
    # y = np.roll(y, 511)
    # y = sin_1d(x, x0=x0, T=period, right=right, left=left, clip=clip)

    # y'(x)
    # y_grad_analitic = cos_1d(x, x0=x0, a=a * 2 * np.pi / period, T=period, right=right, left=left, clip=clip)

    # Numpy Gradient
    y_grad_np = np.gradient(y, dx)
    # y_grad_difference_np = abs(y_grad_analitic - y_grad_np)

    # FFT Gradient
    nu_x = fftfreq(num, dx)
    kx = 1j * 2 * np.pi * nu_x
    y_grad_fft = ifft(kx * fft(y)).real
    # y_grad_difference_fft = abs(y_grad_analitic - y_grad_fft)

    # Error
    y_grad_error = abs(y_grad_np - y_grad_fft)

    print(f'N = {num: >5} \t\t '
          f'w = {width: >10} \t\t '
          f'error = {max(y_grad_error)}')

    # Gradient Plots
    # fig = plt.figure(figsize=[7.4, 5.8])  # 6.4, 4.8
    # ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
    #
    # :threshold = len(x)
    # ax1.plot(x[::threshold], y[::threshold], label='f1 = sin(x)')
    # # ax1.plot(x[::threshold], y_grad_analitic[::threshold], label='f2 = cos(x)', linestyle='--')
    # ax1.plot(x[::threshold], y_grad_np[::threshold], label=f'y\'(x) - numpy', linestyle='dotted')
    # ax1.plot(x[::threshold], y_grad_fft[::threshold], label=f'y\'(x) - fft', linestyle='dotted')
    #
    # # ax2.plot(x, y_grad_difference_np, label='abs(f2 - f3)', linestyle='dotted')
    # # ax2.plot(x, y_grad_difference_fft, label='abs(f2 - f4)', linestyle='--')
    # ax2.plot(x, y_grad_error, label='|numpy - fft|', linestyle='--')
    #
    # ax1.grid()
    # ax1.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))  #, bbox_to_anchor=(.5, 1.7)
    # ax1.set_title(f'Grads. Width = {width:.5f}. Num = {num}. T = {period:.5f}')
    #
    # ax2.grid()
    # ax2.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))  #, bbox_to_anchor=(1., 1.7)
    # ax2.set_yscale('log')
    # ax2.set_title('FFT error')
    #
    # fig.suptitle('Gradient')
    # fig.tight_layout()

    # Numpy Laplacian
    y_laplacian_np = np.gradient(y_grad_np, dx)

    # f(x) <-- Gradient
    mask = (kx == 0)
    kx[mask] = 1.+0*1j
    y_fft = fft(y_grad_fft) / kx
    y_fft[mask] = 0.+0*1j
    y_igrad_fft = ifft(y_fft).real
    kx[mask] = 0.+0*1j

    y_igrad_fft += y[0] - y_igrad_fft[0]

    # FFT Laplacian
    y_laplacian_fft = ifft(kx**2 * fft(y)).real

    # FFT Laplacian --> Gradient
    kx[mask] = 1.+0*1j
    y_grad_from_laplacian_fft = fft(y_laplacian_fft) / kx
    y_grad_from_laplacian_fft[mask] = 0. + 0 * 1j
    y_grad_from_laplacian_fft = ifft(y_grad_from_laplacian_fft).real
    kx[mask] = 0.+0*1j

    # FFT f(x) <-- Laplacian
    kx[mask] = 1.+0*1j
    y_from_laplacian_fft = fft(y_laplacian_fft) / kx**2
    y_from_laplacian_fft[mask] = 0. + 0 * 1j
    y_from_laplacian_fft = ifft(y_from_laplacian_fft).real
    kx[mask] = 0.+0*1j

    # Errors
    y_laplacian_error = np.abs(y_laplacian_np - y_laplacian_fft)
    y_error_f_from_grad = np.abs(y - y_igrad_fft)
    y_error_grad_from_laplacian_fft = np.abs(y_grad_fft - y_grad_from_laplacian_fft)
    y_error_from_laplacian_fft = np.abs(y - y_from_laplacian_fft)

    # Laplacian Plots
    fig = plt.figure(figsize=[7.4, 6.8])  # 6.4, 4.8
    ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)

    threshold = 8
    ax1.plot(x[::threshold], y[::threshold], 'b-', label='y(x)')
    ax1.plot(x[::threshold], y_grad_fft[::threshold], 'g-', label='y\'(x) - fft')
    ax1.plot(x[::threshold], y_laplacian_fft[::threshold], 'r-', label='y\'\'(x) - fft')  # # todo уехал вверх...
    # ax1.plot(x[::threshold], y_laplacian_np[::threshold], 'y-', label='y\'\'(x) - np')
    ax1.plot(x[::threshold], y_igrad_fft[::threshold], 'c*', label='y(x) - from gradient')
    ax1.plot(x[::threshold], y_grad_from_laplacian_fft[::threshold], 'm:', label='y\'(x) - from laplacian')
    ax1.plot(x[::threshold], y_from_laplacian_fft[::threshold], 'k+', label='y(x) - from laplacian')

    ax2.plot(x[::threshold], y_grad_error[::threshold], 'g-', label=f'gradient: {max(y_grad_error):.1e}')
    ax2.plot(x[::threshold], y_laplacian_error[::threshold], 'r-', label=f'laplacian: {max(y_laplacian_error):.1e}')
    ax2.plot(x[::threshold], y_error_f_from_grad[::threshold], 'c*', label=f'y(x) from gradient: {max(y_error_f_from_grad):.1e}')
    ax2.plot(x[::threshold], y_error_grad_from_laplacian_fft[::threshold], 'm:', label=f'grad from laplace: {max(y_error_grad_from_laplacian_fft):.1e}')
    ax2.plot(x[::threshold], y_error_from_laplacian_fft[::threshold], 'k+', label=f'y(x) from laplace: {max(y_error_from_laplacian_fft):.1e}')

    ax1.grid()
    ax1.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))
    ax1.set_title(f'Width = {width:.5f}. Num = {num}. T = {period:.5f}')

    ax2.grid()
    ax2.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))
    ax2.set_yscale('log')
    ax2.set_title('Error = |numpy - fft|')

    fig.suptitle('Laplacian')
    fig.tight_layout()

    # Spectral Plots
    # fig2 = plt.figure(figsize=[8.4, 7.8])  #  6.4, 4.8
    # ax3, ax4, ax5, ax6, ax7 = \
    #     fig2.add_subplot(5, 1, 1), \
    #     fig2.add_subplot(5, 1, 2), \
    #     fig2.add_subplot(5, 1, 3), \
    #     fig2.add_subplot(5, 1, 4), \
    #     fig2.add_subplot(5, 1, 5),
    #
    # ax3.plot(ifftshift(nu_x), ifftshift(np.abs(fft(y))), label='Спектр амплитуд')
    # ax3.plot(ifftshift(nu_x), ifftshift(np.abs(fft(y)*kx)), label='Спектр амплитуд * kx', linestyle='dotted')
    # ax4.plot(ifftshift(nu_x), ifftshift(np.angle(fft(y))), label='Спектр фаз')
    # ax4.plot(ifftshift(nu_x), ifftshift(np.angle(fft(y)*kx)), label='Спектр фаз * kx', linestyle='dotted')
    # ax5.plot(ifftshift(nu_x), ifftshift(kx.imag), label='kx')
    # ax6.plot(ifftshift(nu_x), np.abs(ifftshift(fft(y)).imag), label='fft(y).imag')
    # ax6.plot(ifftshift(nu_x), np.abs(ifftshift(fft(y) * kx).imag), label='(fft(y) * kx).imag', linestyle='dotted')
    # ax7.plot(ifftshift(nu_x), np.abs(ifftshift(fft(y)).real), label='fft(y).real')
    # ax7.plot(ifftshift(nu_x), np.abs(ifftshift(fft(y) * kx).real), label='(fft(y) * kx).real', linestyle='dotted')
    #
    # [ax.legend() for ax in [ax3, ax4, ax5, ax6, ax7]]
    # [ax.grid() for ax in [ax3, ax4, ax5, ax6, ax7]]
    # [ax.set_yscale('log') for ax in [ax3]]
    # # [ax.set_yscale('symlog') for ax in [ax6]]
    #
    # fig2.tight_layout()

    plt.show()
