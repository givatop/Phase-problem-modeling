import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.propagation.utils.math.general import get_slice


def figure_configurator(func):
    def wrapper(intensity, geometry_center=False, **kwargs):
        figsize = kwargs.get('figsize', [16, 9])
        dpi = kwargs.get('dpi', 100)
        facecolor = kwargs.get('facecolor', 'w')
        edgecolor = kwargs.get('edgecolor', 'k')

        fig = func(intensity, geometry_center, **kwargs)

        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.set_facecolor(facecolor)
        fig.set_edgecolor(edgecolor)

        return fig

    return wrapper


def axes_configurator(func):
    def wrapper(ax, **kwargs):
        linewidth = kwargs.get('linewidth', 1.5)
        kwargs['linewidth'] = linewidth

        y_scale = kwargs.get('yscale', 'linear')
        x_scale = kwargs.get('xscale', 'linear')
        x_label = kwargs.get('xlabel', 'x')
        y_label = kwargs.get('ylabel', 'y')

        ax.set_xscale(x_scale)
        ax.set_xscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

        return func(ax, **kwargs)

    return wrapper


@figure_configurator
def make_intensity_plot(intensity, geometry_center=False):

    fig1 = plt.figure()
    ax1, ax2, ax3 = fig1.add_subplot(1, 3, 1), fig1.add_subplot(1, 3, 2), fig1.add_subplot(1, 3, 3)

    make_intensity_x_slice_plot(ax1, intensity=intensity, geometry_center=geometry_center)
    make_intensity_y_slice_plot(ax2, intensity=intensity, geometry_center=geometry_center)
    make_intensity_color_map_plot(ax3, intensity)

    return fig1


@axes_configurator
def make_intensity_x_slice_plot(ax, **kwargs):
    intensity = kwargs.get('intensity')
    geometry_center = kwargs.get('geometry_center')
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


@axes_configurator
def make_intensity_y_slice_plot(ax, **kwargs):
    intensity = kwargs.get('intensity')
    geometry_center = kwargs.get('geometry_center')
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
    ax.title.set_text('y slice')

    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    ax.legend()


def make_intensity_color_map_plot(ax, intensity, **kwargs):
    intensity_lbl = kwargs.get('intensity_lbl', '')  # mm
    cmap = kwargs.get('cmap', 'jet')

    img = ax.imshow(intensity, cmap=cmap,
                    extent=[-intensity.shape[1] // 2, intensity.shape[1] // 2,
                            -intensity.shape[0] // 2, intensity.shape[0] // 2])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax.title.set_text(f'Intensity. {intensity_lbl}')
