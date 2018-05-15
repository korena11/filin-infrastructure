'''
infraGit
photo-lab-3\Reuma
16, Jan, 2017 
'''

import numpy as np
from scipy import interpolate


class RasterFactory:
    """
        Creates raster view from point set based on a certain property (e.g. range, intesity, etc.)
        The panoramic view is stored as a PanoramProperty object
    """
    def CreateRaster(cls, points, _property='Z', gridSpacing= 0.05, interpolationMethod = 'linear'):
        '''

        :param points: The point set to create the raster from can be either a PointSet, PointSubSet or any other BaseProperty-derived objects
        :param gridSpacing: the spacing between grid
        :param interpolationMethod: method of interpolation. {'linear', 'nearest', 'cubic'}
        :param _property: The property to create the panorama according to. Now only applicable with range

        :return: raster property
        '''

        #:TODO: check points is PointSet. else.... do something

        # Bounding box
        values = np.array(points.Size)
        minX = min(points.X)
        maxX = max(points.X)
        minY = min(points.Y)
        maxY = max(points.Y)

        if _property == 'Z':
            values = points.Z
        else: #TODO: applicability with other properties, 'intensity', 'color', 'segmentation', 'normals', 'range'
            pass


        grid_x, grid_y = np.mgrid[minX:gridSpacing:maxX, minY:gridSpacing:maxY]
        xyPoints = points.ToNumpy()[:,:1]
        raster = interpolate.griddata(xyPoints, values, (grid_x, grid_y), method=interpolationMethod)

        return RasterProperty(points, raster, gridSpacing, extent={'North': maxY, 'South': minY, 'West': minX, 'East': maxX})




if __name__ == '__main__':
    pass