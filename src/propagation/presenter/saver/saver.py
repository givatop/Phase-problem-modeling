import numpy as np

from typing import Union, Dict
from abc import ABC, abstractmethod
from matplotlib.figure import Figure


class Saver(ABC):
    """
    Интерфейс сохранения файлов в папку
    """
    @abstractmethod
    def __init__(self, folder_name: str):
        self.folder_name = folder_name

    @abstractmethod
    def save_image(self, image: Union[Figure, np.ndarray], package_name: str, filename: str):
        """
        Сохраняет кратинку
        :return:
        """
        pass

    @abstractmethod
    def save_text(self, text: Union[str, Dict], package_name: str, filename: str):
        """
        Сохраняет текст в файл
        :param text:
        :param package_name:
        :param filename:
        """
        pass

    @staticmethod
    @abstractmethod
    def create_filename(z: float, extension: str = 'png') -> str:
        """
        Создаёт имя файла по указанным параметрам типа:
        :param z:
        :param extension: png or npy
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def create_folder_name(method: str) -> str:
        """
        Создаёт имя пакаета по указанному типу графика
        :param method: тип графика
        :return:
        """
        pass
