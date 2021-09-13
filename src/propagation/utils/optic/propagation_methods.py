import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units
from src.propagation.utils.math.general import get_slice
from src.propagation.utils.optic.field import rect_2d


def angular_spectrum_propagation(wave: Wave, z: float, **kwargs):
    """
    Метод распространения (преобразования) волны методом углового спектра
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """
    frequency_grid = kwargs.get('frequency_grid')

    # волновое число
    wave_number = 2 * np.pi / wave.wavelength

    # частотная сетка
    nu_y_grid, nu_x_grid = frequency_grid.grid.y_grid, frequency_grid.grid.x_grid

    # Фурье-образ исходного поля
    field = fft2(wave.field)

    # передаточная функция слоя пространства
    exp_term = np.sqrt(
        1 - (wave.wavelength * nu_x_grid) ** 2 -
        (wave.wavelength * nu_y_grid) ** 2)
    h = np.exp(1j * wave_number * z * exp_term)

    # todo H((1-(Lambda*U).^2-(Lambda*V).^2)<0) = 0; % neglect evanescent wave

    # обратное преобразование Фурье
    wave.field = ifft2(field * h)


def angular_spectrum_bl_propagation(wave: Wave, z: float):
    """
    Распространение (преобразование) волны при помощи band-limited angular spectrum метода
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """

    # Увеличение транспаранта в 2 раза для трансформации линейной свертки в циклическую
    # (периодические граничные условия)
    height = 2 * wave.field.shape[0]  # количество строк матрицы
    width = 2 * wave.field.shape[1]  # количество элеметов в каждой строке матрицы

    # Индексы для "старого" поля
    left = int(width * .25)
    right = int(width * .75)
    top = int(height * .25)
    bottom = int(height * .75)

    # Вписываем "старое" поле в новое
    new_field = np.zeros((height, width), dtype=wave.field.dtype)
    new_field[top:bottom, left:right] = wave.field

    # Сетка в частотной области
    nu_x = np.arange(-width / 2, width / 2) / (width * wave.grid.pixel_size)
    nu_y = np.arange(-height / 2, height / 2) / (height * wave.grid.pixel_size)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
    nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)
    nu_z_grid = np.sqrt(wave.wavelength ** -2 - nu_x_grid ** 2 - nu_y_grid ** 2)
    nu_z_grid[nu_x_grid ** 2 + nu_y_grid ** 2 > wave.wavelength ** -2] = 0

    # Расчет граничных частот U/V_limit
    dnu_x = 1 / (width * wave.grid.pixel_size)
    dnu_y = 1 / (height * wave.grid.pixel_size)
    nu_x_limit = 1 / (np.sqrt((2 * dnu_x * z) ** 2 + 1) * wave.wavelength)
    nu_y_limit = 1 / (np.sqrt((2 * dnu_y * z) ** 2 + 1) * wave.wavelength)

    # Передаточная функция (угловой спектр)
    h_clipper = rect_2d(nu_x_grid, nu_y_grid, wx=2 * nu_x_limit, wy=2 * nu_y_limit)
    h = np.exp(1j * 2 * np.pi * nu_z_grid * z) * h_clipper

    # обратное преобразование Фурье
    wave.field = ifft2(fft2(new_field) * h)[top:bottom, left:right]


def angular_spectrum_band_limited(
    complex_field: np.ndarray,
    distance: float,
    wavelength: float,
    px_size: float
) -> np.ndarray:
    """
    Kyoji Matsushima and Tomoyoshi Shimobaba DOI: 10.1364/OE.17.019662
    :param complex_field:
    :param distance:
    :param wavelength:
    :param px_size:
    :return:
    """
    # Увеличение транспаранта в 2 раза для трансформации линейной свертки в циклическую
    # (периодические граничные условия)
    height = 2 * complex_field.shape[0]
    width = 2 * complex_field.shape[1]

    # Индексы для "старого" поля
    left = int(width * .25)
    right = int(width * .75)
    top = int(height * .25)
    bottom = int(height * .75)

    # Вписываем "старое" поле в новое
    new_field = np.zeros((height, width), dtype=complex_field.dtype)
    new_field[top:bottom, left:right] = complex_field

    # Сетка в частотной области
    nu_x = np.arange(-width / 2, width / 2) / (width * px_size)
    nu_y = np.arange(-height / 2, height / 2) / (height * px_size)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
    nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)
    nu_z_grid = np.sqrt(wavelength ** -2 - nu_x_grid ** 2 - nu_y_grid ** 2)
    nu_z_grid[nu_x_grid ** 2 + nu_y_grid ** 2 > wavelength ** -2] = 0

    # Расчет граничных частот U/V_limit
    dnu_x = 1 / (width * px_size)
    dnu_y = 1 / (height * px_size)
    nu_x_limit = 1 / (np.sqrt((2 * dnu_x * distance) ** 2 + 1) * wavelength)
    nu_y_limit = 1 / (np.sqrt((2 * dnu_y * distance) ** 2 + 1) * wavelength)

    # Передаточная функция (угловой спектр)
    h_clipper = rect_2d(nu_x_grid, nu_y_grid, wx=2 * nu_x_limit, wy=2 * nu_y_limit)
    h = np.exp(1j * 2 * np.pi * nu_z_grid * distance) * h_clipper

    # обратное преобразование Фурье
    return ifft2(fft2(new_field) * h)[top:bottom, left:right]


def rayleigh_sommerfeld_propagation_1D(wave: Wave, z: float):
    """
    Распространение (преобразование) одномерной волны при помощи интеграла Рэлея-Зоммерфельда
    :param wave: волна
    :param z: дистанция распространения
    :return:
    """
    # todo: добавить класс одномерной волны, переделать метод под интерфейс для графиков
    # Беру сечение волны по одной из координат
    g0 = get_slice(
        wave.field,
        wave.field.shape[0] // 2,
    )[1]
    # Задаю сетку по X
    X = np.arange(-wave.grid.height / 2, wave.grid.height / 2)

    # Расчет волнового числа
    k = 2 * np.pi / wave.wavelength
    # Завожу массив нулей под g(x, z)
    g_z = np.zeros(g0.shape, dtype=complex)

    for g_, x_ in enumerate(X):
        r = np.sqrt((X - x_) ** 2 + z ** 2)  # расчёт r'
        multiplier = _rayleigh_sommerfeld_multiplier(
            z=z, k=k, r=r, wavelength=wave.wavelength
        )  # расчёт множителя под интегралом RS
        result = multiplier * g0  # значение подинетгрального выражения
        one_sum = _rectangle_rule(result, X[0], X[-1], X.shape[0])
        g_z[g_] = one_sum

    return X, g_z


def _rayleigh_sommerfeld_multiplier(z, k, r, wavelength):
    """
    Коэффициент в интеграле Рэлея-Зоммерфельда
    :param z: дистанция распространения волны
    :param k: волновое число
    :param r: радиус распространения волны
    :param wavelength: длина волны
    :return: float
    """
    res = (z * np.exp(1j * k * r)) / (np.sqrt(r ** 3) * 1j * np.sqrt(wavelength))
    return res


def _rectangle_rule(field, a, b, nseg=512):
    """
    Метод интегрирования прямоугольниками
    :param field: волна
    :param a: левая граница прямоугольника
    :param b: правая графница прямоугольника
    :param nseg: количество отсчётов
    :return: float
    """
    dx = (b - a) / nseg
    summa = 0
    for i in range(nseg):
        cur = field[i] * dx
        summa += cur

    return summa


def fresnel(field: np.ndarray, propagate_distance: float,
            wavelenght: float, pixel_size: float) -> np.ndarray:
    """
    Расчет комплексной амплитуды светового поля прошедшей через слой пространства толщиной propagate_distance
    с использованием передаточной функции Френеля
    :param field: array-like
    :param propagate_distance: float z
    :param wavelenght: float lambda
    :param pixel_size: float px_size
    :return: array-like
    """
    raise NotImplementedError("This method not implemented yet")

    height = field.shape[0]
    width = field.shape[1]

    wave_number = 2 * np.pi / wavelenght

    # Сетка в частотной области
    nu_x = np.arange(-width / 2, width / 2) / (width * pixel_size)
    nu_y = np.arange(-height / 2, height / 2) / (height * pixel_size)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
    nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)

    if propagate_distance != 0 and np.abs(propagate_distance) <= 1 / (
            wavelenght * (nu_x.max() ** 2 + nu_y.max() ** 2) ** 2):
        raise ValueError(f'Не выполняется критерий Релея z < 1 / (lamda*(nu_x^2+nu_y^2): '
                         f'{np.abs(propagate_distance)} <= {1 / (wavelength * (nu_x.max() ** 2 + nu_y.max() ** 2) ** 2)}')

    # Фурье-образ исходного поля
    field = fft2(field)

    exp_term = np.sqrt(1 - ((wavelenght * nu_x_grid) ** 2 - (wavelenght * nu_y_grid) ** 2) / 2)
    h = np.exp(1j * wave_number * propagate_distance * exp_term)

    return ifft2(field * h)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from numpy.fft import fft, ifft, ifftshift, fftfreq

    from src.propagation.utils.optic.field import sin_1d, cos_1d, triangle_1d, logistic_1d, rect_1d
    from src.propagation.utils.math.units import (nm2m, mm2m, um2m, m2mm, rad2mm)
    from src.propagation.utils.math.derivative.fourier import gradient_2d, ilaplacian_2d, ilaplacian_1d

    # Параметры волны
    wavelength = um2m(0.5)
    k = 2 * np.pi / wavelength
    focus = mm2m(100)
    z = mm2m(5)
    fields = []

    # Координатная сетка
    num = 1024
    dx = um2m(5)
    width = num * dx
    x = np.linspace(-width/2, width/2, num, endpoint=False)

    # Параметры y(x)
    a = .5
    period = width / 2
    # x0 = 0
    x0 = -width / 4
    left = 0
    right = period*3.5
    clip = 0
    phase_amplitude = 3.14

    # Параметры вывода
    precise = 1

    # Формируем поле z = 0
    i0 = a * np.ones(x.shape)
    # i0 = logistic_1d(x, a=a, x0=x0, w=period) * logistic_1d(-x, a=a, x0=x0, w=period)
    phi0 = phase_amplitude * logistic_1d(x, x0=x0, w=period) * logistic_1d(-x, x0=x0, w=period)
    phi0 -= phase_amplitude / 2
    # phi0 = k * np.sqrt(x**2 + focus**2)
    field0 = np.sqrt(i0) * np.exp(-1j * phi0)
    phi0 = np.angle(field0)
    phi0_unwrapped = np.unwrap(phi0)

    # FFT grid
    nu_x = fftfreq(num, dx)
    kx = 1j * 2 * np.pi * nu_x
    field0_spectrum = fft(field0)
    exp_term = np.sqrt(1 - (wavelength * nu_x) ** 2)

    # Прореживание данных
    cut_step = 1
    x = m2mm(x[::cut_step])
    i0 = i0[::cut_step]
    phi0 = phi0[::cut_step]

    # Распространение
    start = 10
    step = 1
    for step in [10]:  # .1, .5, 1, 2, 4, 8, 16, 32, 64
        fields = []

        stop = start + 2 * step + step/10
        for z in map(mm2m, np.arange(start, stop, step)):
            h = np.exp(1j * k * z * exp_term)
            field_z = ifft(field0_spectrum * h)

            # Добавляем компоненты поля в список fields
            i_z = np.abs(field_z) ** 2
            phi_z = np.angle(field_z)
            phi_z_unwrapped = np.unwrap(phi_z)
            fields.append([i_z, phi_z_unwrapped, z])

        # 1D TIE
        i1, phi1, z1 = fields[0]
        i_ref, phi_ref, _ = fields[1]
        i2, phi2, z2 = fields[-1]

        didz = (i2 - i1) / (z2 - z1)

        phase1 = -k * didz
        phase2 = phase1 / i1
        phase = ilaplacian_1d(phase2, kx)

        # TIE Error
        if len(fields) != 3:
            raise ValueError('Нужно 3 интенсивности')

        delta = max(phi_ref) - max(phase)
        error = abs(phase + delta - phi_ref)

        # TIE details plot
        fig2 = plt.figure(figsize=(7.4, 7.8))
        ax4, ax5, ax6, ax7 = \
            fig2.add_subplot(4, 1, 1), \
            fig2.add_subplot(4, 1, 2), \
            fig2.add_subplot(4, 1, 3), \
            fig2.add_subplot(4, 1, 4)

        ax4.plot(x, didz, '-')
        ax4.set_title('didz')
        ax5.plot(x, phase1, '-')
        ax5.set_title('-k * didz')
        ax6.plot(x, phase2, '-')
        ax6.set_title('-k * didz / i1')
        ax7.plot(x, phase, '-')
        ax7.set_title('ilaplacian')

        # Print
        print(f'dz: {m2mm(z2 - z1):.{precise}f} mm', f'max = {max(error):.1e}')

        # Графики
        fig = plt.figure(figsize=(8.4, 5.8))
        ax1, ax2, ax3 = fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)

        ax1.plot(x, i0, '-', label=f'z=0 mm')
        ax2.plot(x, phi0_unwrapped, '-', label=f'z=0 mm; a = {phase_amplitude} rad')

        for i_z, phi_z_unwrapped, z in fields:

            # Прореживание данных
            i_z = i_z[::cut_step]
            phi_z_unwrapped = phi_z_unwrapped[::cut_step]

            ax1.plot(x, i_z, ':', label=f'z={m2mm(z):.{precise}f} mm')
            ax2.plot(x, phi_z_unwrapped, ':', label=f'z={m2mm(z):.{precise}f} mm')

        ax2.plot(x, phase, ':', label=f'TIE dz={m2mm(z2 - z1):.{precise}f} mm')
        ax3.plot(x, error, '-', label=f'max = {max(error):.1e}')

        [ax.grid() for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]]
        [ax.legend(loc='upper right') for ax in [ax1, ax2, ax3]]

        ax1.set_title(f'Intensity')
        ax1.set_xlabel('mm')
        ax1.set_ylabel('a.u.')

        ax2.set_title(f'Phase')
        ax2.set_xlabel('mm')
        ax2.set_ylabel('rad')
        ax2.set_ylim((-3.14, 3.14))

        ax3.set_title(f'Error')

        # [ax.set_yscale('log') for ax in [ax4, ax5]]

        fig.suptitle(f'Angular Spectrum Propagation')
        fig.tight_layout()

        fig2.suptitle(f'TIE detailed')
        fig2.tight_layout()

        # fig.savefig(f'i phi dz={m2mm(z2 - z1):.{precise}f} mm.png', bbox_inches='tight', pad_inches=0.1)
        # fig2.savefig(f'tie dz={m2mm(z2 - z1):.{precise}f} mm.png', bbox_inches='tight', pad_inches=0.1)

        # plt.close(fig)
        # plt.close(fig2)

    plt.show()
