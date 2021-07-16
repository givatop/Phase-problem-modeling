import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils.math.general import get_slice
from ..utils.math.units import m2mm


def make_intensity_plot(
        intensity: np.ndarray
) -> Figure:
    dpi = 100
    linewidth = 1.5
    cmap = 'jet'
    color_bar = True
    ymin, ymax = [None, None]
    xmin, xmax = [None, None]
    intensity_lbl = ''
    geometry_center = False

    fig = plt.figure(dpi=dpi, figsize=(16, 9))
    ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)

    if geometry_center:
        max_indexes = [intensity.shape[0] // 2, intensity.shape[1] // 2]
    else:
        max_indexes = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)

    wrapped_phase_xslice_x, wrapped_phase_xslice_y = get_slice(intensity, max_indexes[0], xslice=True)
    wrapped_phase_yslice_x, wrapped_phase_yslice_y = get_slice(intensity, max_indexes[1], xslice=False)

    wrapped_phase_xslice_x -= wrapped_phase_xslice_x[wrapped_phase_xslice_x.size // 2]
    wrapped_phase_yslice_x -= wrapped_phase_yslice_x[wrapped_phase_yslice_x.size // 2]

    ax1.plot(wrapped_phase_xslice_x, wrapped_phase_xslice_y, linewidth=linewidth, label=f'x: {max_indexes[1]}')
    ax2.plot(wrapped_phase_yslice_x, wrapped_phase_yslice_y, linewidth=linewidth, label=f'y: {max_indexes[0]}')
    img = ax3.imshow(intensity, cmap=cmap,
                     extent=[-intensity.shape[1] // 2, intensity.shape[1] // 2,
                             -intensity.shape[0] // 2, intensity.shape[0] // 2])

    if color_bar:
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)

    ax1.title.set_text('x slice'), ax2.title.set_text('y slice')
    ax3.title.set_text(f'Intensity. {intensity_lbl}')

    [ax.set_ylim([ymin, ymax]) for ax in [ax1, ax2]]
    [ax.set_xlim([xmin, xmax]) for ax in [ax1, ax2]]
    [ax.legend() for ax in [ax1, ax2]]
    [ax.grid(True) for ax in [ax1, ax2]]

    return fig


def make_phase_plot(
        wrapped_phase: np.ndarray,
        unwrapped_phase: np.ndarray,
        wavelength: float,
        wavefront_radius: float,
        z: float
) -> Figure:
    linewidth = 1.5
    unwrapped_ymin, unwrapped_ymax = [None, None]
    wrapped_ymin, wrapped_ymax = -np.pi, np.pi
    xmin, xmax = [None, None]
    grid_centering = False
    geometry_center = False
    crosshair_halfwidth = 0

    k = 2 * np.pi / wavelength
    height, width = wrapped_phase.shape

    unwrapped_phase_lbl = f'[{np.min(unwrapped_phase):.2f}, ' \
                          f'{np.max(unwrapped_phase):.2f}] rad; ' \
                          f'[{np.min(unwrapped_phase) * 1e+6 / k:.1f}, ' \
                          f'{np.max(unwrapped_phase) * 1e+6 / k:.1f}] um'

    wrapped_phase_lbl = f'z: {m2mm(z):.1f} mm; R: {wavefront_radius:.3f} mm'

    y_max, x_max = __find_max_coordinates(geometry_center, wrapped_phase, unwrapped_phase)

    # Поиск сечений по 2-м осям
    wrp_phase_xslice_x, wrp_phase_xslice_y = get_slice(wrapped_phase, x_max, xslice=True)
    wrp_phase_yslice_x, wrp_phase_yslice_y = get_slice(wrapped_phase, y_max, xslice=False)
    uwrp_phase_xslice_x, uwrp_phase_xslice_y = get_slice(unwrapped_phase, x_max, xslice=True)
    uwrp_phase_yslice_x, uwrp_phase_yslice_y = get_slice(unwrapped_phase, y_max, xslice=False)

    # Центрирование координатной сетки
    if grid_centering:
        wrp_phase_xslice_x -= wrp_phase_xslice_x[wrp_phase_xslice_x.size // 2]  # plot()
        wrp_phase_yslice_x -= wrp_phase_yslice_x[wrp_phase_yslice_x.size // 2]
        uwrp_phase_xslice_x -= uwrp_phase_xslice_x[uwrp_phase_xslice_x.size // 2]
        uwrp_phase_yslice_x -= uwrp_phase_yslice_x[uwrp_phase_yslice_x.size // 2]
        extent = [-width // 2, width // 2, height // 2, -height // 2]  # imshow()
    else:
        extent = [0, width, height, 0]

    # Перекрестные линии для визуального различения заданного центра
    if crosshair_halfwidth != 0:
        wrp_phase, uwrp_phase = wrapped_phase.copy(), unwrapped_phase.copy()
        wrp_mean = (wrp_phase.max() - wrp_phase.min()) / 2
        uwrp_mean = (uwrp_phase.max() - uwrp_phase.min()) / 2
        wrp_phase[y_max - crosshair_halfwidth:y_max + crosshair_halfwidth, :] = wrp_mean
        wrp_phase[:, x_max - crosshair_halfwidth:x_max + crosshair_halfwidth] = wrp_mean
        uwrp_phase[y_max - crosshair_halfwidth:y_max + crosshair_halfwidth, :] = uwrp_mean
        uwrp_phase[:, x_max - crosshair_halfwidth:x_max + crosshair_halfwidth] = uwrp_mean

    # Создание окна и 6-ти осей
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2, ax3 = fig.add_subplot(2, 3, 1), fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)
    ax4, ax5, ax6 = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)

    ax1.plot(wrp_phase_xslice_x, wrp_phase_xslice_y, linewidth=linewidth, label=f'x: {y_max}')
    ax2.plot(wrp_phase_yslice_x, wrp_phase_yslice_y, linewidth=linewidth, label=f'y: {x_max}')
    img = ax3.imshow(wrapped_phase, extent=extent, cmap='jet')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax4.plot(uwrp_phase_xslice_x, uwrp_phase_xslice_y, linewidth=linewidth, label=f'x: {y_max}')
    ax5.plot(uwrp_phase_yslice_x, uwrp_phase_yslice_y, linewidth=linewidth, label=f'y: {x_max}')
    img = ax6.imshow(unwrapped_phase, extent=extent, cmap='jet')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax1.title.set_text('x slice'), ax2.title.set_text('y slice')
    ax4.title.set_text('x slice'), ax5.title.set_text('y slice')
    ax3.title.set_text(f'Wrapped. {wrapped_phase_lbl}')
    ax6.title.set_text(f'Unwrapped. {unwrapped_phase_lbl}')

    [ax.set_ylim([wrapped_ymin, wrapped_ymax]) for ax in [ax1, ax2]]
    [ax.set_ylim([unwrapped_ymin, unwrapped_ymax]) for ax in [ax4, ax5]]
    [ax.set_xlim([xmin, xmax]) for ax in [ax4, ax5]]
    [ax.legend() for ax in [ax1, ax2, ax4, ax5]]
    [ax.grid(True) for ax in [ax1, ax2, ax4, ax5]]

    return fig


def make_r_z_plot(
        waves_array: np.ndarray,  # 2d array
        apertures_array: np.ndarray,  # 2d array
        z_array: np.ndarray,
        matrixes: list,
        step: float,
) -> Figure:

    array_wave_array = waves_array[0]
    array_aperture_array = apertures_array[0]
    z_array = z_array[0]
    matrix = matrixes[0]

    fig, ax = plt.subplots()

    marker = '-o'  # вид маркера линий
    markersize = 2  # размер марера линий
    linewidth = 1.  # толщина линии графика реальных радиусов

    # определение реальных и теоретических радиусов кривизны волн и построение их графиков
    for z in np.arange(0, np.shape(matrix)[0], 1):
        radius_y = []
        theory_r_z = []
        waves = array_wave_array[z]
        apertures = array_aperture_array[z]
        for wave, aperture, k in zip(waves, apertures, z_array):
            radius_y.append(wave.get_wavefront_radius(aperture=aperture, z=z))
            theory_r_z.append(np.abs(np.array(k) - m2mm(wave.focal_len)))

        if z == 0:
            theory_r_z = np.abs(np.array(z_array) - m2mm(array_wave_array[0][0].focal_len))
            ax.plot(z_array, theory_r_z,
                    label='Theoretical',
                    color='k',
                    markersize=markersize)

        ax.plot(z_array, radius_y,
                marker,
                label=f'size: {matrix[z]}',
                linewidth=linewidth,
                markersize=markersize)

    #  определение масштаба графиков
    theory_r_z = np.abs(np.array(z_array) - m2mm(array_wave_array[0][0].focal_len))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, theory_r_z[-1])

    # заголовок графика
    ax.title.set_text(f'f\' = {m2mm(np.around(array_wave_array[0][0].focal_len, decimals=3))} mm; '
                      f'g = {array_wave_array[0][0].gaussian_width_param}; '
                      f'step = {step} mm')
    ax.legend()

    return fig


def __find_max_coordinates(geometry_center, wrp_phase, unwrp_phase) -> (float, float):
    """
    Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    """
    height, width = wrp_phase.shape
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)
    return y_max, x_max
