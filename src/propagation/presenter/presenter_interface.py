import numpy as np
import matplotlib.pyplot as plt


from ..presenter.plotter import (
    make_intensity_plot,
    make_phase_plot,
    make_r_z_plot
)

from ..presenter.saver import (
    save_plot,
    save_npy,
    save_text
)


def save_intensity_plot(
        folder_name: str,
        filename: str,
        intensity: np.ndarray
):
    intensity_plot_figure = make_intensity_plot(intensity)
    save_plot(folder_name, filename, intensity_plot_figure, 'intensity')
    plt.close(intensity_plot_figure)


def save_phase_plot(
        folder_name: str,
        filename: str,
        wrapped_phase: np.ndarray,
        unwrapped_phase: np.ndarray,
        wavelength: float,
        wavefront_radius: float,
        z: float
):
    phase_plot_figure = make_phase_plot(wrapped_phase, unwrapped_phase, wavelength, wavefront_radius, z)
    save_plot(folder_name, filename, phase_plot_figure, 'phase')


def save_intensity_npy(
        folder_name: str,
        filename: str,
        intensity: np.ndarray
):
    save_npy(folder_name, filename, intensity, 'intensity')


def save_phase_npy(
        folder_name: str,
        filename: str,
        phase: np.ndarray
):
    save_npy(folder_name, filename, phase, 'phase')


def save_r_z_plot(
        folder_name: str,
        filename: str,
        waves_array: np.ndarray,  # 2d array
        apertures_array: np.ndarray,  # 2d array
        z_array: np.ndarray,
        matrixes: list,
        step: float
):
    r_z_plot_figure = make_r_z_plot(waves_array, apertures_array, z_array, matrixes, step)
    save_plot(folder_name, filename, r_z_plot_figure)


def save_r_z_metadata(
        folder_name: str,
        filename: str,
        r: float,
        z: float
):
    save_text(folder_name, filename, z, r)
