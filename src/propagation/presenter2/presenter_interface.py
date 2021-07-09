import numpy as np
import matplotlib.pyplot as plt

from ..presenter2.plotter import (
    make_intensity_plot,
    make_phase_plot,
    make_r_z_plot
)

from ..presenter2.saver import (
    save_plot,
    save_npy,
    save_text
)


def save_intensity_plot(
        folder_name: str,
        filename: str,
        intensity: np.ndarray
):
    intensity_plot_figure = make_intensity_plot()
    save_plot(folder_name, filename, intensity_plot_figure, 'intensity')
    plt.close(intensity_plot_figure)


def save_phase_plot(
        folder_name: str,
        filename: str,
        wrapped_phase: np.ndarray,
        unwrapped_phase: np.ndarray
):
    phase_plot_figure = make_phase_plot()
    save_plot(folder_name, filename, phase_plot_figure, 'phase')
    plt.close(phase_plot_figure)


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
        wave_z_dictionary: dict,
        matrix_shape_aperture_dictionary: dict,
        step: float
):
    r_z_plot_figure = make_r_z_plot()
    save_plot(folder_name, filename, r_z_plot_figure)
    plt.close(r_z_plot_figure)


def save_r_z_metadata(
        folder_name: str,
        filename: str,
        r: float,
        z: float
):
    save_text(folder_name, filename, z, r)
