from ..areas.grid import CartesianGrid
from ...model.waves.interface.wave import Wave
from ...utils.optic.propagation_methods import angular_spectrum_propagation


class ArbitraryWave(Wave):
    """ Произвольная волна """

    def __init__(self, intensity, phase, grid: CartesianGrid, wavelength: float):
        """

        :param intensity:
        :param phase:
        :param grid: двухмерная координатная сетка расчёта распределения поля
        :param wavelength: длина волны [м]
        """
        field = (intensity, phase)
        super().__init__(field, grid, wavelength)

    def propagate_on_distance(self, z: float, method=angular_spectrum_propagation, **kwargs):
        method(self, z, **kwargs)
