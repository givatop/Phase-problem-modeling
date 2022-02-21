import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.propagation.utils.math.units as units
from src.propagation.presenter.loader import load_file


plt.style.use('seaborn')
DPI = 150
FIGSIZE = [12, 6]
CMAP = 'jet'
show = 0


def create_cbar(ax, img, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel(label)
    return cbar


dzs = units.um2mm(np.arange(40, 1001, 40))

for dz in dzs:
    print(f'dz: {dz:.3f} mm')

    z1 = -dz / 2
    z2 = dz / 2

    # Загрузка файлов
    folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
             r'\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\2. Автофокусировка\z=0.000 propagation'

    path = os.path.join(folder, f'BMSTU -b 1024x1024.png')
    init_phase = load_file(path)
    init_phase -= np.min(init_phase)
    height, width = init_phase.shape

    path = os.path.join(folder, f'TIE i z={z1:.3f} i z={z2:.3f} dz={dz:.3f}mm.npy')
    tie_phase = np.load(path)
    tie_phase -= np.min(tie_phase)

    # Расчет ошибки
    diff = init_phase[512, 55:-55] - tie_phase[512, 55:-55]
    tie_phase += np.mean(diff)

    ap_i_ys, ap_i_yf = int(np.ceil(0.05 * height)), int(np.ceil(0.95 * height))
    ap_i_xs, ap_i_xf = int(np.ceil(0.05 * width)), int(np.ceil(0.95 * width))
    abs_error = np.abs(init_phase - tie_phase)[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]
    tie_phase = tie_phase[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]
    init_phase = init_phase[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]

    # Графики
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 dpi=DPI, figsize=FIGSIZE)

    img = ax1.imshow(tie_phase, cmap=CMAP, vmin=0, vmax=.8)
    create_cbar(ax1, img, label='rad')

    img = ax2.imshow(abs_error, cmap=CMAP, vmin=0, vmax=.3)
    create_cbar(ax2, img, label='rad')

    # Slices
    y_slice = 470
    color = 'magenta'
    ax3.plot(tie_phase[y_slice, :], color=color, label='TIE')
    ax3.plot(init_phase[y_slice, :], color='green', linestyle='dashed',
             label='Initial')

    ax4.plot(abs_error[y_slice, :], color=color)

    ax1.axhline(y=y_slice, color=color)
    ax2.axhline(y=y_slice, color=color)

    ax1.set_title('TIE phase')
    ax2.set_title(f'Absolute error: dz = {dz:.3f} mm')

    ax3.legend(loc='lower right')

    ax3.set_ylim([-0.2, 1])
    ax4.set_ylim([-0.01, 0.3])

    fig.tight_layout()

    if not show:
        filename = f'error dz = {dz:.3f} mm'
        path = os.path.join(folder, f'{filename}.png')
        fig.savefig(path)
        plt.close(fig)


if show:
    plt.show()
