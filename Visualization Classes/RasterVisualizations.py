'''
infraGit
photo-lab-3\Reuma
16, Jan, 2017 
'''

import platform

import numpy as np
from numpy import pi, sqrt, arctan, arctan2, sin, cos

from RasterData import RasterData

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')


class RasterVisualization:

    @classmethod
    def hillshade(cls, array, azimuth=315, angle_altitude=45):
        """
        Generates hillshade from array used by azimuth and angle altitude

        :param array: DEM
        :param azimuth: degrees, default 315 deg
        :param angle_altitude: degrees, defualt 45 deg

        :return: a shaded relief of a raster

        """
        if isinstance(array, RasterData):
            array = array.data
        x, y = np.gradient(array)
        slope = pi / 2. - arctan(sqrt(x * x + y * y))
        aspect = arctan2(-x, y)
        azimuthrad = azimuth * pi / 180.
        altituderad = angle_altitude * pi / 180.

        shaded = sin(altituderad) * sin(slope) + \
                 cos(altituderad) * cos(slope) * cos(azimuthrad - aspect)
        return 255 * (shaded + 1) / 2


if __name__ == '__main__':
    pass