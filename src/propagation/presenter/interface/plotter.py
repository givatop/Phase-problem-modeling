from abc import ABC, abstractmethod


class Plotter(ABC):
    """
    Абстрактный класс строителя графиков
    """

    @abstractmethod
    def save_phase(self):
        """
        Сохраняет графики фаз
        :return:
        """
        pass

    @abstractmethod
    def save_intensity(self):
        """
        Сохраняет графики интенсивности
        :return:
        """
        pass

    @abstractmethod
    def save_r_z(self):
        """
        Сохраняет графики зависимости радиуса волнового фронта от дистанции распространения волны
        :return:
        """
        pass
