import os
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import unwrap_phase

import src.propagation.utils.math.units as units
import src.propagation.utils.optic as optic
from src.propagation.presenter.loader import load_file
from src.propagation.utils.optic.propagation_methods import angular_spectrum_band_limited
from src.propagation.utils.tie import (
    FFTSolver1D,
    FFTSolver2D,
    BoundaryConditions,
)


def create_cbar(ax, img, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel(label)
    return cbar


# DISTANCE = units.um2m(10)
for DISTANCE in [
    units.um2m(10),
    units.um2m(100),
    units.um2m(1000),
    units.um2m(10000),
]:
    WAVELENGTH = units.nm2m(500)
    PX_SIZE = units.um2m(5)
    DPI = 200

    folder = r"\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа" + \
             r"\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\1. Изображения"
    filename = 'BMSTU -b 1024x1024.png'
    path = os.path.join(folder, filename)


    phase = load_file(path)
    intensity = np.ones(phase.shape)

    height, width = intensity.shape
    x = np.arange(-width / 2, width / 2) * PX_SIZE
    y = np.arange(-height / 2, height / 2) * PX_SIZE
    X, Y = np.meshgrid(x, y)

    aperture = optic.rect_2d(X, Y, wx=0.9 * width * PX_SIZE, wy=0.9 * height * PX_SIZE)
    phase, intensity = phase * aperture, intensity * aperture

    field = np.sqrt(intensity) * np.exp(1j * phase) * aperture


    # Propagation
    minus_distance = -DISTANCE / 2
    plus_distance = DISTANCE / 2

    field_plus_z = angular_spectrum_band_limited(field, plus_distance, WAVELENGTH, PX_SIZE)
    field_minus_z = angular_spectrum_band_limited(field, minus_distance, WAVELENGTH, PX_SIZE)

    intensity_plus_z, phase_plus_z = np.abs(field_plus_z) ** 2, unwrap_phase(np.angle(field_plus_z) * aperture)
    intensity_minus_z, phase_minus_z = np.abs(field_minus_z) ** 2, unwrap_phase(np.angle(field_minus_z) * aperture)


    # TIE
    solver = FFTSolver2D([intensity_minus_z, intensity_plus_z], DISTANCE,
                         WAVELENGTH, PX_SIZE, ref_intensity=intensity_minus_z)
    
    phase_z_tie = solver.solve(threshold=0.1)


    # Error
    diff = phase[512, 55:-55] - phase_z_tie[512, 55:-55]
    phase_z_tie += np.mean(diff)

    ap_i_ys, ap_i_yf = int(np.ceil(0.05 * height)), int(np.ceil(0.95 * height))
    ap_i_xs, ap_i_xf = int(np.ceil(0.05 * width)), int(np.ceil(0.95 * width))
    abs_error = np.abs(phase - phase_z_tie)[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]
    phase_z_tie = phase_z_tie[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]
    phase = phase[ap_i_ys:ap_i_yf, ap_i_xs:ap_i_xf]


    # Propagated field plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, dpi=DPI)
    i_vmin, i_vmax = None, None
    p_vmin, p_vmax = None, None

    img = ax1.imshow(intensity_minus_z, vmin=i_vmin, vmax=i_vmax)
    create_cbar(ax1, img, label='a.u.')

    img = ax2.imshow(phase_minus_z, vmin=p_vmin, vmax=p_vmax)
    create_cbar(ax2, img, label='rad')

    img = ax3.imshow(intensity_plus_z, vmin=i_vmin, vmax=i_vmax)
    create_cbar(ax3, img, label='a.u.')

    img = ax4.imshow(phase_plus_z, vmin=p_vmin, vmax=p_vmax)
    create_cbar(ax4, img, label='rad')

    ax1.set_title(f'z = {units.m2mm(minus_distance):.3f} mm')
    ax3.set_title(f'z = {units.m2mm(plus_distance):.3f} mm')

    fig.tight_layout()
    path = os.path.join(folder, f'field dz = {units.m2mm(DISTANCE):.3f} mm.png')
    fig.savefig(path)


    # TIE
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, dpi=DPI)
    i_vmin, i_vmax = None, None
    p_vmin, p_vmax = 0.0, 0.8

    img = ax1.imshow(phase_z_tie, vmin=p_vmin, vmax=p_vmax)
    create_cbar(ax1, img, label='rad')

    img = ax2.imshow(abs_error, vmin=p_vmin, vmax=p_vmax)
    create_cbar(ax2, img, label='rad')

    # Slices
    y_slice = 512
    color = 'orange'
    ax3.plot(phase_z_tie[y_slice, :], color=color, label='TIE phase')
    ax3.plot(phase[y_slice, :], color='green', linestyle='dashed',
             label='Initial phase')

    ax4.plot(abs_error[y_slice, :], color=color)

    ax1.axhline(y=y_slice, color=color)
    ax2.axhline(y=y_slice, color=color)

    ax1.set_title('TIE phase')
    ax2.set_title(f'Absolute error: dz = {units.m2mm(DISTANCE):.3f} mm')

    ax3.grid()
    ax4.grid()

    ax3.legend()

    ax3.set_ylim([0.1, 1.1])
    ax4.set_ylim([0.0, 0.8])

    fig.tight_layout()
    filename = f'error dz = {units.m2mm(DISTANCE):.3f} mm'
    path = os.path.join(folder, f'{filename}.png')
    fig.savefig(path)


    # Save NDARRAYs
    path = os.path.join(folder, f'i z={units.m2mm(minus_distance):.3f}mm.npy')
    np.save(path, intensity_minus_z)

    path = os.path.join(folder, f'i z={units.m2mm(plus_distance):.3f}mm.npy')
    np.save(path, intensity_plus_z)

    path = os.path.join(folder, f'phi z=0.000mm.npy')
    np.save(path, phase)

    path = os.path.join(folder, f'TIE z1={units.m2mm(minus_distance):.3f} '
                                f'z2={units.m2mm(plus_distance):.3f}mm '
                                f'dz={units.m2mm(DISTANCE):.3f}mm.npy')
    np.save(path, phase_z_tie)

    path = os.path.join(folder, f'TIE ERROR z1={units.m2mm(minus_distance):.3f} '
                                f'z2={units.m2mm(plus_distance):.3f}mm '
                                f'dz={units.m2mm(DISTANCE):.3f}mm.npy')
    np.save(path, abs_error)


# plt.show()
