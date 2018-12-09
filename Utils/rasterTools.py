import matplotlib.pyplot as plt
import numpy as np
import richdem as rd

from RasterData import RasterData


class RasterTools:
    """
    General tools for raster computations.
    """

    @classmethod
    def slope_richdem(cls, raster, verbose=False, method='slope_riserun'):
        """
        Compute a slope map of a DEM via `richDEM`

        :param raster: DEM
        :param verbose: flag to show a figure of the result or not (default False)
        :param method: the method according to which the slope is computed:
            - 'slope_riserun'
            - 'slope_precentage'
            - 'slope_degrees'
            - 'slope_radians'

        Further information in `richdem <https://richdem.readthedocs.io/en/latest/python_api.html>`_.

        :type raster: RasterData, np.array
        :type verbose: bool

        :return: slope map

        :rtype: np.array
        """
        if isinstance(raster, RasterData):
            raster = rd.rdarray(raster.data, no_data=raster.voidData)
        elif isinstance(raster, np.ndarray):
            raster = rd.rdarray(raster, no_data=-9999)

        slope = rd.TerrainAttribute(raster, attrib=method)

        if verbose:
            plt.imshow(slope)
        return slope

    @classmethod
    def aspect_richdem(cls, raster, verbose=False):
        """
        Compute the aspect of a DEM via `richDEM`

        :param raster: DEM
        :param verbose: flag to show a figure of the result or not (default False)

        Further information in `richdem <https://richdem.readthedocs.io/en/latest/python_api.html>`_.

        :type raster: (RasterData, np.array)
        :type verbose: bool

        :return: aspect map

        :rtype: np.array
        """
        if isinstance(raster, RasterData):
            raster = rd.rdarray(raster.data, no_data=raster.voidData)
        elif isinstance(raster, np.ndarray):
            raster = rd.rdarray(raster, no_data=-9999)

        aspect = rd.TerrainAttribute(raster, attrib='aspect')

        if verbose:
            plt.imshow(aspect)
        return aspect

    @classmethod
    def curvature_richdem(cls, raster, verbose=False):
        """
        Compute the curvature of a DEM via `richDEM`

        :param raster: DEM
        :param verbose: flag to show a figure of the result or not (default False)

        Further information in `richdem <https://richdem.readthedocs.io/en/latest/python_api.html>`_.

        :type raster: RasterData, np.array
        :type verbose: bool

        :return: curvature map

        :rtype: np.array
        """
        if isinstance(raster, RasterData):
            raster = rd.rdarray(raster.data, no_data=raster.voidData)
        elif isinstance(raster, np.ndarray):
            raster = rd.rdarray(raster, no_data=-9999)

        curvature = rd.TerrainAttribute(raster, attrib='curvature')

        if verbose:
            plt.imshow(curvature)
        return curvature

    @classmethod
    def plane_curvature_richdem(cls, raster, verbose=False):
        """
        Compute the curvature of a DEM via `richDEM`

        :param raster: DEM
        :param verbose: flag to show a figure of the result or not (default False)

        Further information in `richdem <https://richdem.readthedocs.io/en/latest/python_api.html>`_.

        :type raster: RasterData, np.array
        :type verbose: bool

        :return: plane curvature map

        :rtype: np.array
        """
        if isinstance(raster, RasterData):
            raster = rd.rdarray(raster.data, no_data=raster.voidData)
        elif isinstance(raster, np.ndarray):
            raster = rd.rdarray(raster, no_data=-9999)

        curvature = rd.TerrainAttribute(raster, attrib='planform_curvature')

        if verbose:
            plt.imshow(curvature)

        return curvature

    @classmethod
    def profile_curvature_richdem(cls, raster, verbose=False):
        """
        Compute the curvature of a DEM via `richDEM`

        :param raster: DEM
        :param verbose: flag to show a figure of the result or not (default False)

        Further information in `richdem <https://richdem.readthedocs.io/en/latest/python_api.html>`_.

        :type raster: RasterData, np.array
        :type verbose: bool

        :return: profile curvature map

        :rtype: np.array
        """
        if isinstance(raster, RasterData):
            raster = rd.rdarray(raster.data, no_data=raster.voidData)
        elif isinstance(raster, np.ndarray):
            raster = rd.rdarray(raster, no_data=-9999)

        curvature = rd.TerrainAttribute(raster, attrib='profile_curvature')

        if verbose:
            plt.imshow(curvature)
        return curvature
