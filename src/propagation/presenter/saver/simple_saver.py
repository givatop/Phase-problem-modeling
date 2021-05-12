from typing import Union
from matplotlib.figure import Figure
import numpy as np
import os
from os import path

from src.propagation.presenter.saver.saver import Saver
from src.propagation.model.waves.interface.wave import Wave
from src.propagation.utils.math import units


class SimpleSaver(Saver):
    """
    Сохранение файлов в MacBook Матвей
    """
    def __init__(self, folder_name: str):
        super().__init__(folder_name)

    def save_image(self, fig: Union[Figure, np.ndarray], package_name: str, filename: str):

        # todo add path.join
        if not path.exists(f"./../../data/{self.folder_name}/{package_name}/"):
            os.makedirs(f"./../../data/{self.folder_name}/{package_name}")
        else:
            pass
            # todo логика, если директория уже существует

        filepath = os.getcwd() + f"/../../data/{self.folder_name}/{package_name}/{filename}"

        if isinstance(fig, Figure):
            fig.savefig(filepath)
        elif isinstance(fig, np.ndarray):
            np.save(filepath, fig)

    @staticmethod
    def create_filename(z: float, extension: str = 'png') -> str:
        return f'z_{units.m2mm(z):.3f}mm.{extension}'

    @staticmethod
    def create_folder_name(method: str, wave=False) -> str:
        plot_package = {'i': 'intensity',
                        'p': 'phase',
                        'r': 'r(z)',
                        'b': 'bound'}[method]
        # package_name = f'{method if wave else method}'
        return plot_package
