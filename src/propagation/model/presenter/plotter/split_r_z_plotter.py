import matplotlib.pyplot as plt
import numpy as np

from src.propagation import MacSaver
from src.propagation.model.presenter.configuration.configurator import figure_configurator, axes_configurator
from src.propagation.utils.math import units


def save_r_z(array_wave_array, array_aperture_array, z_array, matrixes, step):
    """
    Сохраняет график для интенсивности
    :return:
    """
    saver = MacSaver()
    fig = make_intensity_plot(**{'array_wave_array': array_wave_array,
                                 'array_aperture_array': array_aperture_array,
                                 'z_array': z_array,
                                 'matrixes': matrixes,
                                 'step': step})

    # сохранение графиков
    package_name = 'r(z)'
    filename = f'trz_f_{int(units.m2mm(np.around(array_wave_array[0][0].focal_len, decimals=3)))}_' \
               f'g{array_wave_array[0][0].gaussian_width_param}_matrix_multiple'
    saver.save_image(fig, package_name, filename)

    plt.close(fig)


@figure_configurator
def make_intensity_plot(**kwargs):
    fig, ax = plt.subplots()

    make_r_z_ax(ax, **kwargs)

    return fig


@axes_configurator
def make_r_z_ax(ax, **kwargs):
    array_wave_array = kwargs['array_wave_array'],
    array_aperture_array = kwargs['array_aperture_array'],
    z_array = kwargs['z_array'],
    matrix = kwargs['matrixes'],
    step = kwargs['step']

    array_wave_array = array_wave_array[0]
    array_aperture_array = array_aperture_array[0]
    z_array = z_array[0]
    matrix = matrix[0]

    marker = '-o'  # вид маркера линий
    markersize = kwargs.get('markersize', 2)  # размер марера линий
    linewidth = kwargs.get('linewidth', 1.)  # толщина линии графика реальных радиусов

    # определение реальных и теоретических радиусов кривизны волн и построение их графиков
    for z in np.arange(0, np.shape(matrix)[0], 1):
        radius_y = []
        theory_r_z = []
        waves = array_wave_array[z]
        apertures = array_aperture_array[z]
        for wave, aperture, k in zip(waves, apertures, z_array):
            radius_y.append(wave.get_wavefront_radius(aperture))
            theory_r_z.append(np.abs(np.array(k) - units.m2mm(wave.focal_len)))

        if z == 0:
            theory_r_z = np.abs(np.array(z_array) - units.m2mm(array_wave_array[0][0].focal_len))
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
    theory_r_z = np.abs(np.array(z_array) - units.m2mm(array_wave_array[0][0].focal_len))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, theory_r_z[-1])

    # заголовок графика
    ax.title.set_text(f'f\' = {units.m2mm(np.around(array_wave_array[0][0].focal_len, decimals=3))} mm; '
                      f'g = {array_wave_array[0][0].gaussian_width_param}; '
                      f'step = {step} mm')
    ax.legend()
