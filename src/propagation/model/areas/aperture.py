
from src.propagation.utils.optic.field import circ
from .grid import RadialCoordinateGrid

from ...utils.math import units


class Aperture:
    """ Апертура, circ """

    def __init__(self, radial_area: RadialCoordinateGrid, aperture_diameter: float):
        """
        Создаёт круглую апертуру на основе сетки сферических координат
        :param radial_area: сетки сферических координат
        :param aperture_diameter: диаметр апертуры [px]
        """
        aperture_diameter = units.px2m(aperture_diameter, px_size_m=radial_area.pixel_size)  # [м]
        self.__aperture_diameter = aperture_diameter
        self.__radial_area = radial_area
        self.__aperture = circ(radial_area.coordinate_grid, w=aperture_diameter)

    @property
    def aperture_diameter(self):
        return self.__aperture_diameter

    @aperture_diameter.setter
    def aperture_diameter(self, aperture_diameter):
        self.__aperture_diameter = units.px2m(aperture_diameter, px_size_m=self.__radial_area.pixel_size)  # [м]
        self.__aperture = circ(self.radial_area.coordinate_grid, w=self.__aperture_diameter)

    @property
    def radial_area(self):
        return self.__radial_area

    @property
    def aperture(self):
        return self.__aperture
