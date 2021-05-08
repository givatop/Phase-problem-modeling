from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...areas.interface.aperture import Aperture
from ...waves.interface.wave import Wave
from ....utils.math import units
from ....utils.math.general import get_slice


def configuration(func):
    """
    Конфигуратор функций построения графиков свойствами графиков
    :param func: функция построения для конфигурирования
    :return:
    """

    def wrapper(*args, **kwargs):
        figsize = kwargs.get('figsize', [8.0, 6.0])
        dpi = kwargs.get('dpi', 300)
        grid = kwargs.get('grid', True)
        x_label = kwargs.get('xlabel', 'x')
        y_label = kwargs.get('ylabel', 'y')
        y_scale = kwargs.get('yscale', 'linear')
        facecolor = kwargs.get('facecolor', 'w')
        edgecolor = kwargs.get('edgecolor', 'k')
        linewidth = kwargs.get('linewidth', 1.5)

        fig, ax = plt.subplots(figsize=figsize,
                               dpi=dpi,
                               facecolor=facecolor,
                               edgecolor=edgecolor,
                               linewidth=linewidth)

        ax.grid(grid)
        plt.yscale(y_scale)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        func(fig=fig, ax=ax, *args, **kwargs)

    return wrapper


class Plotter(ABC):
    """
    Абстрактный класс строителя графиков
    """

    @abstractmethod
    def save_phase(self):
        """
        Сохраняет графики фаз
        :return:
        """
        pass

    @abstractmethod
    def save_intensity(self):
        """
        Сохраняет графики интенсивности
        :return:
        """
        pass

    @abstractmethod
    def save_aperture_bound(self, bound: float):
        """
        Сохраняет зависимости с пересечением скачка апертуры (с 0 на 1) с неразвернутой фазой волны
        :param bound: диапазон значений вблизи скачка апертуры (с 0 на 1)
        :return:
        """
        pass

    @abstractmethod
    def save_r_z(self):
        """
        Сохраняет графики зависимости радиуса волнового фронта от дистанции распространения волны
        :return:
        """
        pass

    @staticmethod
    def _make_aperture_bound_dependency(wave: Wave, aperture: Aperture, bound: float) -> (tuple, tuple):
        """
        Создаёт зависимость с пересечением скачка апертуры (с 0 на 1) с неразвернутой фазой волны
        :param wave: волна
        :param aperture: апертура
        :param bound: диапазон значений
        :return: два кортежа из массивов [x, y] значений неразвернутой фазы и апертуры в пределах скачка
        в указанном диапазоне (+- диапазон)
        """

        wrp_phase_x_slice_x, wrp_phase_x_slice_y = get_slice(
            wave.phase,
            wave.phase.shape[0] // 2,
            xslice=True
        )
        ap_x_slice_x, ap_x_slice_y = get_slice(
            aperture.aperture,
            aperture.aperture.shape[0] // 2,
            xslice=True
        )

        change_index = 0

        for i, v in enumerate(ap_x_slice_y):
            if v == 0:
                continue
            else:
                change_index = i
                break

        return (wrp_phase_x_slice_x[change_index - bound:change_index + bound],
                wrp_phase_x_slice_y[change_index - bound:change_index + bound]), \
               (ap_x_slice_x[change_index - bound:change_index + bound],
                ap_x_slice_y[change_index - bound:change_index + bound])

    @staticmethod
    def _make_r_z_dependency(wave: Wave, aperture: Aperture, z: float) -> (tuple, tuple):
        """
        Создаёт зависимость радиуса кривизны волнового фронта от расстояния
        :return:
        """
        return (z, wave.get_wavefront_radius(aperture)) + \
               (z, np.abs(z - units.m2mm(wave.focal_len)))
