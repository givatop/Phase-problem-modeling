
from src.propagation.utils.optic.field import circ
from .grid import PolarGrid

from ...utils.math.units import px2m


class Aperture:
    """ Апертура, circ """

    def __init__(self, polar_grid: PolarGrid, aperture_diameter: float):
        """
        Создаёт апертуру (circ) на основе сетки в полярных координатах
        :param polar_grid: сетка в полярных координатах
        :param aperture_diameter: диаметр апертуры [px]
        """
        aperture_diameter = px2m(aperture_diameter, px_size_m=polar_grid.pixel_size)  # [м]
        self._aperture_diameter = aperture_diameter
        self._radial_area = polar_grid
        self._aperture = circ(polar_grid.polar_grid, w=aperture_diameter)

    @property
    def aperture_diameter(self):
        return self._aperture_diameter

    @aperture_diameter.setter
    def aperture_diameter(self, aperture_diameter):
        self._aperture_diameter = px2m(aperture_diameter, px_size_m=self._radial_area.pixel_size)  # [м]
        self._aperture = circ(self.radial_area.polar_grid, w=self._aperture_diameter)

    @property
    def radial_area(self):
        return self._radial_area

    @property
    def aperture(self):
        return self._aperture
