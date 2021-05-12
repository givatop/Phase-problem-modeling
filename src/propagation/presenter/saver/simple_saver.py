import os
import numpy as np

from os import path
from typing import Union
from matplotlib.figure import Figure

from src.propagation.presenter.saver.saver import Saver
from src.propagation.utils.math import units


class SimpleSaver(Saver):
    """
    Сохранение файлов в MacBook Матвей
    """
    def __init__(self, folder_name: str):
        super().__init__(folder_name)

    def save_image(self, image: Union[Figure, np.ndarray], package_name: str, filename: str):

        # todo add path.join
        if not path.exists(f"./../../data/{self.folder_name}/{package_name}/"):
            os.makedirs(f"./../../data/{self.folder_name}/{package_name}")
        else:
            pass
            # todo логика, если директория уже существует

        filepath = os.getcwd() + f"/../../data/{self.folder_name}/{package_name}/{filename}"

        if isinstance(image, Figure):
            image.savefig(filepath)
        elif isinstance(image, np.ndarray):
            np.save(filepath, image)

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
