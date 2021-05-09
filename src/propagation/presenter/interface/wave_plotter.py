import matplotlib.pyplot as plt
import numpy as np

from src.propagation.presenter.plotter.figure_maker import make_r_z_plot, make_phase_plot, make_intensity_plot
from src.propagation.model.areas.interface.aperture import Aperture
from src.propagation.presenter.saver.mac_saver import SimpleSaver
from src.propagation.presenter.saver.saver import Saver
from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units


class WavePlotter:
    """
    Построение графиков распространения волны в пространстве
    """

    @staticmethod
    def save_phase(wave: Wave, aperture: Aperture, z: float, saver: Saver):
        """
        Сохраняет график для фазы
        :return:
        """
        focus = wave.focal_len
        gaussian_width_param = wave.gaussian_width_param
        k = 2 * np.pi / wave.wavelength

        unwrapped_phase_lbl = f'[{np.min(wave.get_unwrapped_phase(aperture)[0]):.2f}, ' \
                              f'{np.max(wave.get_unwrapped_phase(aperture)[0]):.2f}] rad; ' \
                              f'[{np.min(wave.get_unwrapped_phase(aperture)[0]) * 1e+6 / k:.1f}, ' \
                              f'{np.max(wave.get_unwrapped_phase(aperture)[0]) * 1e+6 / k:.1f}] um'

        wrapped_phase_lbl = f'z: {units.m2mm(z):.1f} mm; R: {wave.get_wavefront_radius(aperture):.3f} mm'

        fig = make_phase_plot(wrp_phase=wave.get_wrapped_phase(aperture),
                              unwrp_phase=wave.get_unwrapped_phase(aperture)[0],
                              geometry_center=True,
                              linewidth=1,
                              unwrapped_ylims=(-100, 100),
                              unwrapped_phase_lbl=unwrapped_phase_lbl,
                              wrapped_phase_lbl=wrapped_phase_lbl)

        package_name = f'phase/phase_f{int(units.m2mm(np.around(focus, decimals=3)))}_' \
                       f'g{gaussian_width_param}_' \
                       f's{wave.area.coordinate_grid[0].shape[0]}'

        filename = saver.create_filename(wave, 'phase', z=z)
        saver.save_image(fig, package_name, filename)
        plt.close(fig)

    @staticmethod
    def save_intensity(wave: Wave, z: float, saver: Saver):
        """
        Сохраняет график для интенсивности
        :return:
        """
        fig = make_intensity_plot(intensity=wave.intensity)

        package_name = f'intensity/intensity_f{int(units.m2mm(np.around(wave.focal_len, decimals=3)))}_' \
                       f'g{wave.gaussian_width_param}_' \
                       f's{wave.area.coordinate_grid[0].shape[0]}'
        filename = saver.create_filename(wave, 'intensity', z=z)
        saver.save_image(fig, package_name, filename)

        plt.close(fig)

    @staticmethod
    def save_r_z(array_wave_array, array_aperture_array, z_array, matrixes, step, saver: SimpleSaver):
        """
        Сохраняет график для интенсивности
        :return:
        """
        fig = make_r_z_plot(**{'array_wave_array': array_wave_array,
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
