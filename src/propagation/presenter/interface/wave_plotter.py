import matplotlib.pyplot as plt
import numpy as np

from src.propagation.presenter.plotter.figure_maker import make_r_z_plot, make_phase_plot, make_intensity_plot
from src.propagation.model.areas.interface.aperture import Aperture
from src.propagation.presenter.interface.plotter import Plotter
from src.propagation.presenter.saver.mac_saver import MacSaver
from src.propagation.presenter.saver.saver import Saver
from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units


class WavePlotter(Plotter):
    """
    Построение графиков одной прогонки волны в пространстве
    """

    def __init__(self, wave: Wave, aperture: Aperture, distance: float, saver: Saver):
        """
        :param wave: снапшот волны - волна, распространившаяся на расстояние distance
        :param aperture: апертура для данной волны
        :param distance: координата распространения
        :param saver: класс, сохраняющий графики
        """
        self.__wave = wave
        self.__aperture = aperture
        self.__z = distance
        self.__saver = saver

    def save_phase(self):
        """
        Сохраняет график для фазы
        :return:
        """
        focus = self.__wave.focal_len
        gaussian_width_param = self.__wave.gaussian_width_param
        k = 2 * np.pi / self.__wave.wavelength

        unwrapped_phase_lbl = f'[{np.min(self.__wave.get_unwrapped_phase(self.__aperture)[0]):.2f}, ' \
                              f'{np.max(self.__wave.get_unwrapped_phase(self.__aperture)[0]):.2f}] rad; ' \
                              f'[{np.min(self.__wave.get_unwrapped_phase(self.__aperture)[0]) * 1e+6 / k:.1f}, ' \
                              f'{np.max(self.__wave.get_unwrapped_phase(self.__aperture)[0]) * 1e+6 / k:.1f}] um'

        wrapped_phase_lbl = f'z: {units.m2mm(self.__z):.1f} mm; R: {self.__wave.get_wavefront_radius(self.__aperture):.3f} mm'

        fig = make_phase_plot(wrp_phase=self.__wave.get_wrapped_phase(self.__aperture),
                              unwrp_phase=self.__wave.get_unwrapped_phase(self.__aperture)[0],
                              geometry_center=True,
                              linewidth=1,
                              unwrapped_ylims=(-100, 100),
                              unwrapped_phase_lbl=unwrapped_phase_lbl,
                              wrapped_phase_lbl=wrapped_phase_lbl)

        package_name = f'phase/phase_f{int(units.m2mm(np.around(focus, decimals=3)))}_' \
                       f'g{gaussian_width_param}_' \
                       f's{self.__wave.area.coordinate_grid[0].shape[0]}'

        filename = self.__saver.create_filename(self.__wave, 'phase', z=self.__z)
        self.__saver.save_image(fig, package_name, filename)
        plt.close(fig)

    def save_intensity(self):
        """
        Сохраняет график для интенсивности
        :return:
        """
        fig = make_intensity_plot(intensity=self.__wave.intensity)

        package_name = f'intensity/intensity_f{int(units.m2mm(np.around(self.__wave.focal_len, decimals=3)))}_' \
                       f'g{self.__wave.gaussian_width_param}_' \
                       f's{self.__wave.area.coordinate_grid[0].shape[0]}'
        filename = self.__saver.create_filename(self.__wave, 'intensity', z=self.__z)
        self.__saver.save_image(fig, package_name, filename)

        plt.close(fig)

    def save_r_z(self, array_wave_array, array_aperture_array, z_array, matrixes, step):
        """
        Сохраняет график для интенсивности
        :return:
        """
        saver = MacSaver()
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

    @property
    def wave(self):
        return self.__wave

    @wave.setter
    def wave(self, wave):
        self.__wave = wave

    @property
    def aperture(self):
        return self.__aperture

    @aperture.setter
    def aperture(self, aperture):
        self.__aperture = aperture

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, z):
        self.__z = z

    @property
    def saver(self):
        return self.__saver

    @saver.setter
    def saver(self, saver):
        self.__saver = saver
