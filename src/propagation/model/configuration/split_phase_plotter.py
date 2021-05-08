import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.propagation.model.configuration.configurator import axes_configurator, figure_configurator
from src.propagation.utils.math.general import get_slice


@figure_configurator
def make_phase_plot(**kwargs):
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2, ax3 = fig.add_subplot(2, 3, 1), fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)
    ax4, ax5, ax6 = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)

    make_wrp_phase_x_slice_ax(ax1, **kwargs)
    make_wrp_phase_y_slice_ax(ax2, **kwargs)
    make_unwrp_phase_x_slice_ax(ax4, **kwargs)
    make_unwrp_phase_y_slice_ax(ax5, **kwargs)
    make_wrp_phase_color_ax(ax3, **kwargs)
    make_unwrp_phase_color_ax(ax6, **kwargs)

    return fig


@axes_configurator
def make_wrp_phase_x_slice_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    wrapped_ymin, wrapped_ymax = kwargs.get('wrapped_ylims', [-np.pi, np.pi])
    linewidth = kwargs.get('linewidth', 1.5)
    grid_centering = kwargs.get('grid_centering', False)  # mm

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    wrp_phase_xslice_x, wrp_phase_xslice_y = get_slice(wrp_phase, x_max, xslice=True)

    # Центрирование координатной сетки
    if grid_centering:
        wrp_phase_xslice_x -= wrp_phase_xslice_x[wrp_phase_xslice_x.size // 2]  # plot()

    ax.title.set_text('x slice')
    ax.plot(wrp_phase_xslice_x, wrp_phase_xslice_y, linewidth=linewidth, label=f'x: {y_max}')
    ax.set_ylim([wrapped_ymin, wrapped_ymax])
    ax.legend()


@axes_configurator
def make_wrp_phase_y_slice_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    wrapped_ymin, wrapped_ymax = kwargs.get('wrapped_ylims', [-np.pi, np.pi])
    linewidth = kwargs.get('linewidth', 1.5)
    grid_centering = kwargs.get('grid_centering', False)  # mm

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    wrp_phase_yslice_x, wrp_phase_yslice_y = get_slice(wrp_phase, y_max, xslice=False)

    # Центрирование координатной сетки
    if grid_centering:
        wrp_phase_yslice_x -= wrp_phase_yslice_x[wrp_phase_yslice_x.size // 2]  # plot()

    ax.title.set_text('y slice')
    ax.plot(wrp_phase_yslice_x, wrp_phase_yslice_y, linewidth=linewidth, label=f'y: {x_max}')
    ax.set_ylim([wrapped_ymin, wrapped_ymax])
    ax.legend()


@axes_configurator
def make_unwrp_phase_x_slice_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    unwrapped_ymin, unwrapped_ymax = kwargs.get('unwrapped_ylims', [None, None])
    linewidth = kwargs.get('linewidth', 1.5)
    grid_centering = kwargs.get('grid_centering', False)  # mm
    xmin, xmax = kwargs.get('xlims', [None, None])

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    unwrp_phase_xslice_x, unwrp_phase_xslice_y = get_slice(unwrp_phase, x_max, xslice=True)

    # Центрирование координатной сетки
    if grid_centering:
        unwrp_phase_xslice_x -= unwrp_phase_xslice_x[unwrp_phase_xslice_x.size // 2]  # plot()

    ax.title.set_text('x slice')
    ax.plot(unwrp_phase_xslice_x, unwrp_phase_xslice_y, linewidth=linewidth, label=f'x: {y_max}')
    ax.set_ylim([unwrapped_ymin, unwrapped_ymax])
    ax.set_xlim([xmin, xmax])
    ax.legend()


@axes_configurator
def make_unwrp_phase_y_slice_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    unwrapped_ymin, unwrapped_ymax = kwargs.get('unwrapped_ylims', [None, None])
    linewidth = kwargs.get('linewidth', 1.5)
    grid_centering = kwargs.get('grid_centering', False)  # mm
    xmin, xmax = kwargs.get('xlims', [None, None])

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    unwrp_phase_yslice_x, unwrp_phase_yslice_y = get_slice(unwrp_phase, y_max, xslice=False)

    # Центрирование координатной сетки
    if grid_centering:
        unwrp_phase_yslice_x -= unwrp_phase_yslice_x[unwrp_phase_yslice_x.size // 2]  # plot()

    ax.title.set_text('y slice')
    ax.plot(unwrp_phase_yslice_x, unwrp_phase_yslice_y, linewidth=linewidth, label=f'y: {x_max}')
    ax.set_ylim([unwrapped_ymin, unwrapped_ymax])
    ax.set_xlim([xmin, xmax])
    ax.legend()


@axes_configurator
def make_wrp_phase_color_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    wrapped_phase_lbl = kwargs.get('wrapped_phase_lbl', '')  # mm
    grid_centering = kwargs.get('grid_centering', False)  # mm
    crosshair_halfwidth = kwargs.get('crosshair_halfwidth', 0)  # mm

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    # Центрирование координатной сетки
    if grid_centering:
        extent = [-width // 2, width // 2, height // 2, -height // 2]  # imshow()
    else:
        extent = [0, width, height, 0]

    # Перекрестные линии для визуального различения заданного центра
    if crosshair_halfwidth != 0:
        wrp_phase = wrp_phase.copy()
        wrp_mean = (wrp_phase.max() - wrp_phase.min()) / 2
        wrp_phase[y_max - crosshair_halfwidth:y_max + crosshair_halfwidth, :] = wrp_mean
        wrp_phase[:, x_max - crosshair_halfwidth:x_max + crosshair_halfwidth] = wrp_mean

    img = ax.imshow(wrp_phase, extent=extent, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax.title.set_text(f'Wrapped. {wrapped_phase_lbl}')
    ax.grid(False)


@axes_configurator
def make_unwrp_phase_color_ax(ax, **kwargs):
    wrp_phase = kwargs.get('wrp_phase')
    unwrp_phase = kwargs.get('unwrp_phase')
    geometry_center = kwargs.get('geometry_center')
    unwrapped_phase_lbl = kwargs.get('unwrapped_phase_lbl', '')  # mm
    grid_centering = kwargs.get('grid_centering', False)  # mm
    crosshair_halfwidth = kwargs.get('crosshair_halfwidth', 0)  # mm

    height, width = wrp_phase.shape

    # Поиск максимальных по модулю координат для заданного центра: геометрического или энергетического
    if geometry_center:
        y_max, x_max = [height // 2, width // 2]
    else:
        abs_max = np.argmax(unwrp_phase)
        if np.abs(np.min(unwrp_phase)) > np.abs(np.max(unwrp_phase)):  # Отрицательный энергетический центр
            abs_max = np.argmin(unwrp_phase)
        y_max, x_max = np.unravel_index(abs_max, unwrp_phase.shape)

    # Центрирование координатной сетки
    if grid_centering:
        extent = [-width // 2, width // 2, height // 2, -height // 2]  # imshow()
    else:
        extent = [0, width, height, 0]

    # Перекрестные линии для визуального различения заданного центра
    if crosshair_halfwidth != 0:
        wrp_phase, unwrp_phase = wrp_phase.copy(), unwrp_phase.copy()
        unwrp_mean = (unwrp_phase.max() - unwrp_phase.min()) / 2
        unwrp_phase[y_max - crosshair_halfwidth:y_max + crosshair_halfwidth, :] = unwrp_mean
        unwrp_phase[:, x_max - crosshair_halfwidth:x_max + crosshair_halfwidth] = unwrp_mean

    img = ax.imshow(unwrp_phase, extent=extent, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax.title.set_text(f'Unwrapped. {unwrapped_phase_lbl}')
    ax.grid(False)
