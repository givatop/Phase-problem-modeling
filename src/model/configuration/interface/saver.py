from abc import ABC, abstractmethod
from matplotlib.figure import Figure


# интерфейс для сохранения файлов в папку
class Saver(ABC):

    @abstractmethod
    def save_image(self, fig: Figure, package_name: str, filename: str):
        """
        Сохраняет кратинку
        :return:
        """
        pass