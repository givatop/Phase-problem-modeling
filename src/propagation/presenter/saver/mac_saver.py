from typing import Union
from matplotlib.figure import Figure
import numpy as np
import os
from os import path

from src.propagation.presenter.saver.saver import Saver
from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units


class MacSaver(Saver):
    """
    Сохранение файлов в MacBook Матвей
    """

    def save_image(self, fig: Union[Figure, np.ndarray], package_name: str, filename: str):

        if not path.exists(f"./../../data/images/{package_name}/"):
            os.makedirs(f"./../../data/images/{package_name}")

        filepath = os.getcwd() + f"/../../data/images/{package_name}/{filename}"

        if isinstance(fig, Figure):
            fig.savefig(filepath)
        elif isinstance(fig, np.ndarray):
            np.save(filepath, fig)

    @staticmethod
    def create_filename(wave: Wave, method: str, z=False, it=False) -> str:
        return f'{method}_' \
               f'f{int(units.m2mm(np.around(wave.focal_len, decimals=3)))}_' \
               f'g{wave.gaussian_width_param}_' \
               f's{wave.area.coordinate_grid[0].shape[0]}_' + \
               f'{str(int(units.m2mm(z))) + "_" if z else "0_"}' + \
               f'{f"{it}.png" if it else ".png"}'

    @staticmethod
    def create_folder_name(method: str, wave=False) -> str:
        plot_package = {'i': 'intensity',
                        'p': 'phase',
                        'r': 'r(z)',
                        'b': 'bound'}[method]
        # package_name = f'{method if wave else method}'
        return plot_package
