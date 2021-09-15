import os
import sys
import argparse


from icecream import ic
import numpy as np
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
from src.propagation.utils.math.units import m2um, m2mm

parser = argparse.ArgumentParser(description='Propagate initial wave on desired distances')

parser.add_argument(
    '--mode',
    type=str,
    choices=['complex_amplitude', 'slice', 'error', 'tie_phase'],
    required=True,
    help='Режим'
)
parser.add_argument(
    '--file_path',
    type=str,
    required=True,
)
parser.add_argument(
    '--save_folder',
    type=str,
    required=True,
)
# Figure
parser.add_argument(
    '--dpi',
    type=int,
    default=100,
)
parser.add_argument(
    '--figsize',
    type=float,
    nargs=2,
    required=True,
)
parser.add_argument(
    '--figure_title',
    type=str,
    default='Figure',
)

# Graphic
parser.add_argument(
    '--cmap',
    type=str,
    default='gray'
)
parser.add_argument(
    '--px_size',
    type=float,
    required=True,
)
parser.add_argument(
    '--add_grid',
    type=bool,
    default=True,
)
parser.add_argument(
    '--intensity_title',
    type=str,
    default='I(x,y)',
)
parser.add_argument(
    '--phase_title',
    type=str,
    default='Phi(x,y)',
)
parser.add_argument(
    '--intensity_xlabel',
    type=str,
    default='x, mm',
)
parser.add_argument(
    '--phase_xlabel',
    type=str,
    default='x, mm',
)
parser.add_argument(
    '--intensity_ylabel',
    type=str,
    default='y, mm',
)
parser.add_argument(
    '--phase_ylabel',
    type=str,
    default='y, mm',
)
parser.add_argument(
    '--intensity_cbar_ylabel',
    type=str,
    default='a.u.',
)
parser.add_argument(
    '--phase_cbar_ylabel',
    type=str,
    default='rad',
)
# Output
parser.add_argument(
    '--show_plot',
    type=int,
    default=1,
)
parser.add_argument(
    '--save_plot',
    type=int,
    default=1,
)

args = parser.parse_args()

# Setup
mode = args.mode
filepath = args.file_path
save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Figure
dpi = args.dpi
figsize = args.figsize
figure_title = args.figure_title

# Graphic
cmap = args.cmap
px_size = args.px_size
add_grid = args.add_grid
intensity_title = args.intensity_title
phase_title = args.phase_title
intensity_xlabel = args.intensity_xlabel
phase_xlabel = args.phase_xlabel
intensity_ylabel = args.intensity_ylabel
phase_ylabel = args.phase_ylabel
intensity_cbar_ylabel = args.intensity_cbar_ylabel
phase_cbar_ylabel = args.phase_cbar_ylabel

# Print
ic(mode)
ic(filepath)
ic(save_folder)
ic(dpi)
ic(figsize)
ic(figure_title)
ic(cmap)
ic(px_size)
ic(add_grid)
ic(intensity_title)
ic(phase_title)
ic(intensity_xlabel)
ic(phase_xlabel)
ic(intensity_ylabel)
ic(phase_ylabel)
ic(intensity_cbar_ylabel)
ic(phase_cbar_ylabel)

# Initial
fig = plt.figure(dpi=dpi, figsize=figsize)
fig.suptitle(figure_title)
array = np.load(filepath)

# Grid
if array.ndim == 1:
    height, width = 1, array.shape[0]
elif array.ndim == 2:
    height, width = array.shape
else:
    ValueError(f'Unknown shape: {array.shape}')

x = np.arange(-width // 2, width // 2)
y = np.arange(-height // 2, height // 2)
X, Y = np.meshgrid(x, y)

extent = list(map(
    lambda size_in_px: m2mm(size_in_px * px_size),
    [-width // 2, width // 2, height // 2, -height // 2]
))
ic(width, height, extent)


if mode == 'complex_amplitude':
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    # Intensity
    intensity = np.abs(array) ** 2

    if array.ndim == 1:
        ax1.plot(x, intensity)
    else:
        img1 = ax1.imshow(intensity, extent=extent, cmap=cmap)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(img1, cax=cax1)
        cbar1.ax.set_ylabel(intensity_cbar_ylabel)

    ax1.grid(add_grid)
    ax1.title.set_text(intensity_title)
    ax1.set_xlabel(intensity_xlabel)
    ax1.set_ylabel(phase_ylabel)

    # Phase
    phase = unwrap_phase(np.angle(array))

    if array.ndim == 1:
        ax2.plot(x, phase)
    else:
        img2 = ax2.imshow(phase, extent=extent, cmap=cmap)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(img2, cax=cax2)
        cbar2.ax.set_ylabel(phase_cbar_ylabel)

    ax2.grid(add_grid)
    ax2.title.set_text(phase_title)
    ax2.set_xlabel(phase_xlabel)
    ax2.set_ylabel(phase_ylabel)


elif mode == 'tie_phase':
    ax = fig.gca()

    if array.ndim == 1:
        ax.plot(x, array)
    else:
        img = ax.imshow(array, extent=extent, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.set_ylabel(phase_cbar_ylabel)
    ax.grid(add_grid)
    ax.title.set_text(phase_title)
    ax.set_xlabel(phase_xlabel)
    ax.set_ylabel(phase_ylabel)


fig.tight_layout()

if args.save_plot:
    init_filename = os.path.splitext(os.path.basename(filepath))[0]
    filename = f'{init_filename} {mode}.png'
    save_path = os.path.join(save_folder, filename)
    fig.savefig(save_path)

if args.show_plot:
    plt.show()
