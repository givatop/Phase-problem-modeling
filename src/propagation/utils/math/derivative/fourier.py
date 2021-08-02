from numpy.fft import fft2, ifft2
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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import fft, ifft, ifftshift, fftfreq

    from src.propagation.utils.optic.field import sin_1d, cos_1d

    # width = 15 * np.pi
    num = 1024
    dx = 5.06 * 1e-2  # width/num
    width = num * dx
    a = 1
    period = width / 6  # 2 * np.pi
    x0 = width / 10
    left = 0  # np.pi * 3
    right = period*3.5  # period * 4.5
    clip = 0

    x = np.linspace(0, width, num, endpoint=False)
    y = sin_1d(x, x0=x0, T=period, right=right, left=left, clip=clip)

    # занулим край
    # y[-1] = 0
    # y[0] = 0

    # Аналитическая производная от f'= a * (2pi / T) * cos( 2*2pi*(x-x0)/(T/2pi) )
    y_grad_analitic = cos_1d(x, x0=x0, a=a * 2 * np.pi / period, T=period, right=right, left=left, clip=clip)
    y_grad_np = np.gradient(y, dx)
    y_grad_difference_np = abs(y_grad_analitic - y_grad_np)

    nu_x = fftfreq(num, dx)
    kx = 1j * 2 * np.pi * nu_x
    y_grad_fft = ifft(kx * fft(y)).real
    y_grad_difference_fft = abs(y_grad_analitic - y_grad_fft)

    print(f'N = {num: >5} \t\t '
          f'w = {width: >10} \t\t '
          f'Numpy Error: {np.max(y_grad_difference_np): >15} \t\t '
          f'FFT Error: {np.max(y_grad_difference_fft): >15}')

    fig = plt.figure(figsize=[8.4, 6.8])  # 6.4, 4.8
    ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)

    threshold = len(x)
    ax1.plot(x[:threshold], y[:threshold], label='f1 = sin(x)')
    ax1.plot(x[:threshold], y_grad_analitic[:threshold], label='f2 = cos(x)', linestyle='--')
    ax1.plot(x[:threshold], y_grad_np[:threshold], label='f3 = np.gradient(sin(x))', linestyle='dotted')
    ax1.plot(x[:threshold], y_grad_fft[:threshold], label='f4 = ifft(kx * fft(sin(x)))', linestyle='dotted')

    ax2.plot(x, y_grad_difference_np, label='abs(f2 - f3)', linestyle='dotted')
    ax2.plot(x, y_grad_difference_fft, label='abs(f2 - f4)', linestyle='--')

    ax1.grid()
    ax1.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))  #, bbox_to_anchor=(.5, 1.7)
    ax1.set_title(f'Grads. Width = {width:.5f}. Num = {num}. T = {period:.5f}')

    ax2.grid()
    ax2.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0))  #, bbox_to_anchor=(1., 1.7)
    ax2.set_yscale('log')
    ax2.set_title('FFT error')

    fig.tight_layout()

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
