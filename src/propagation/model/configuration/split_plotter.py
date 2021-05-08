import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.propagation.utils.math.general import get_slice


def make_intensity_plot(intensity, geometry_center=False, **kwargs):
    """
    Строит график с исчерпывающей информацией об интенсивности волны
    :return:
    """
    title = kwargs.get('title', '')
    dpi = kwargs.get('dpi', 100)
    linewidth = kwargs.get('linewidth', 1.5)
    cmap = kwargs.get('cmap', 'jet')
    color_bar = kwargs.get('color_bar', True)
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'y')
    ymin, ymax = kwargs.get('ylims', [None, None])
    xmin, xmax = kwargs.get('xlims', [None, None])
    intensity_lbl = kwargs.get('intensity_lbl', '')  # mm

    fig1 = plt.figure(dpi=dpi, figsize=(16, 9))
    ax1, ax2, ax3 = fig1.add_subplot(1, 3, 1), fig1.add_subplot(1, 3, 2), fig1.add_subplot(1, 3, 3)

    make_intensity_x_slice_plot(ax1, intensity, geometry_center, linewidth=linewidth, ymin=ymin, ymax=ymax, xmin=xmin,
                                xmax=xmax)
    make_intensity_y_slice_plot(ax2, intensity, geometry_center, linewidth=linewidth, ymin=ymin, ymax=ymax, xmin=xmin,
                                xmax=xmax)
    make_intensity_color_map_plot(ax3, intensity, intensity_lbl=intensity_lbl, color_bar=color_bar, cmap=cmap)

    return fig1


def make_intensity_x_slice_plot(ax, intensity, geometry_center=False, **kwargs):
    linewidth = kwargs.get('linewidth', 1.5)
    ymin, ymax = kwargs.get('ylims', [None, None])
    xmin, xmax = kwargs.get('xlims', [None, None])

    if geometry_center:
        max_indexes = [intensity.shape[0] // 2, intensity.shape[1] // 2]
    else:
        max_indexes = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)

    intensity_xslice_x, intensity_xslice_y = get_slice(intensity, max_indexes[0], xslice=True)
    intensity_xslice_x -= intensity_xslice_x[intensity_xslice_x.size // 2]
    ax.plot(intensity_xslice_x, intensity_xslice_y, linewidth=linewidth, label=f'x: {max_indexes[1]}')
    ax.title.set_text('x slice')

    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    ax.legend()
    ax.grid(True)


def make_intensity_y_slice_plot(ax, intensity, geometry_center=False, **kwargs):
    linewidth = kwargs.get('linewidth', 1.5)
    ymin, ymax = kwargs.get('ylims', [None, None])
    xmin, xmax = kwargs.get('xlims', [None, None])

    if geometry_center:
        max_indexes = [intensity.shape[0] // 2, intensity.shape[1] // 2]
    else:
        max_indexes = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)

    intensity_yslice_x, intensity_yslice_y = get_slice(intensity, max_indexes[1], xslice=False)
    intensity_yslice_x -= intensity_yslice_x[intensity_yslice_x.size // 2]
    ax.plot(intensity_yslice_x, intensity_yslice_y, linewidth=linewidth, label=f'y: {max_indexes[0]}')
    ax.title.set_text('x slice')

    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    ax.legend()
    ax.grid(True)


def make_intensity_color_map_plot(ax, intensity, **kwargs):
    intensity_lbl = kwargs.get('intensity_lbl', '')  # mm
    color_bar = kwargs.get('color_bar', True)
    cmap = kwargs.get('cmap', 'jet')

    img = ax.imshow(intensity, cmap=cmap,
                    extent=[-intensity.shape[1] // 2, intensity.shape[1] // 2,
                            -intensity.shape[0] // 2, intensity.shape[0] // 2])

    if color_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)

    ax.title.set_text(f'Intensity. {intensity_lbl}')
