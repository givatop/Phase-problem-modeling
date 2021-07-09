import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from os import path

from ..model.waves.interface.wave import Wave
from ..utils.math.units import m2mm

TEXT_FOLDER_NAME = ''
R_Z_FOLDER_NAME = ''
INTENSITY_PNG_FOLDER_NAME = 'intensity png'
INTENSITY_NPY_FOLDER_NAME = 'intensity npy'
PHASE_PNG_FOLDER_NAME = 'phase png'
PHASE_NPY_FOLDER_NAME = 'phase npy'


def save_plot(
        folder_name: str,
        filename: str,
        plot_figure: Figure,
        how: str = ''
):
    if how == 'intensity':
        package_name = INTENSITY_PNG_FOLDER_NAME
    elif how == 'phase':
        package_name = PHASE_PNG_FOLDER_NAME
    else:
        package_name = R_Z_FOLDER_NAME

    if not path.exists(f"./../data/{folder_name}/{package_name}/"):
        os.makedirs(f"./../data/{folder_name}/{package_name}")

    filepath = os.getcwd() + f"/../data/{folder_name}/{package_name}/{filename}"

    plot_figure.savefig(filepath)


def save_npy(
        folder_name: str,
        filename: str,
        npy_matrix: np.ndarray,
        how: str
):
    package_name = INTENSITY_NPY_FOLDER_NAME if how == 'intensity' else PHASE_NPY_FOLDER_NAME

    if not path.exists(f"./../data/{folder_name}/{package_name}/"):
        os.makedirs(f"./../data/{folder_name}/{package_name}")

    filepath = os.getcwd() + f"/../data/{folder_name}/{package_name}/{filename}"

    np.save(filepath, npy_matrix)


def save_text(
        folder_name: str,
        filename: str,
        z: float,
        r: float
):
    if not path.exists(f"./../data/{folder_name}/{TEXT_FOLDER_NAME}/"):
        os.makedirs(f"./../data/{folder_name}/{TEXT_FOLDER_NAME}")

    filepath = os.getcwd() + f"/../data/{folder_name}/{TEXT_FOLDER_NAME}/{filename}.txt"

    text = f'{z:.3f} {r:.3f}'

    with open(filepath, 'a') as file:
        if isinstance(text, str):
            file.write(text + '\n')


def create_filename(
        z: float,
        extension: str = 'png'
) -> str:
    return f'z_{m2mm(z):.3f}mm.{extension}'


def create_folder_name(
        start_point: float,
        stop_point: float,
        step: float,
        wave: Wave
) -> str:
    return f'z_{m2mm(start_point)}-{m2mm(stop_point)}-{m2mm(step)} ' \
           f'f_{m2mm(wave.focal_len)} ' \
           f'w_{wave.gaussian_width_param} ' \
           f'{wave.grid.width}x{wave.grid.height}'

