import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.propagation.model.configuration.configurator import figure_configurator, axes_configurator
from src.propagation.utils.math.general import get_slice


@figure_configurator
def make_intensity_plot(**kwargs):
    fig = plt.figure()
    ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)

    make_intensity_x_slice_ax(ax1, **kwargs)
    make_intensity_y_slice_ax(ax2, **kwargs)
    make_intensity_color_ax(ax3, **kwargs)

    return fig


@axes_configurator
def make_intensity_x_slice_ax(ax, **kwargs):
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
def make_intensity_y_slice_ax(ax, **kwargs):
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


@axes_configurator
def make_intensity_color_ax(ax, **kwargs):
    intensity = kwargs.get('intensity')
    intensity_lbl = kwargs.get('intensity_lbl', '')  # mm
    cmap = kwargs.get('cmap', 'jet')

    img = ax.imshow(intensity, cmap=cmap,
                    extent=[-intensity.shape[1] // 2, intensity.shape[1] // 2,
                            -intensity.shape[0] // 2, intensity.shape[0] // 2])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    ax.title.set_text(f'Intensity. {intensity_lbl}')
    ax.grid(False)
