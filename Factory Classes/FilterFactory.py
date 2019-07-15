from numba import jit
from numpy import sqrt, arctan, ones, uint8, pi, min, max, logical_and, nonzero

from PointSet import PointSet
from Properties.Neighborhood.NeighborsProperty import NeighborsProperty
from Properties.SegmentationProperty import SegmentationProperty


class FilterFactory:
    """
    Filter PointSet using different methods
    """

    @staticmethod
    def SlopeBasedMorphologicFilter(points, searchRadius, slopeThreshold):
        """
        Slope Based Morphological Filter
        
        :param points
        :param searchRadius: search radius in meters
        :param slopeThreshold: maximum slope angle allowed, given in degrees

        :type points: PointSet
        :type searchRadius: float
        :type slopeThreshold: float

        :return: segmented points, with segment 0 contains the terrain points

        :rtype: SegmentationProperty

        """
        pntData = points.ToNumpy()
        groundPointsIndices = FilterFactory.__slopeBasedMorphologicFilter(pntData, searchRadius, slopeThreshold)

        # 0 - terrain, 1 - cover
        return FilterFactory.__CreateSegmentationProperty(points, groundPointsIndices)        

    @staticmethod
    def FilterByBoundingBox(points, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):
        """
        Filtering based on a defined boundary box
        

        :param points
        :param xmin, xmax: ymin, ymax, zmin, zmax:

        :type points: PointSet
        :type xmin, xmax: ymin, ymax, zmin, zmax: float

        :return: segmented data, segment 0 contains the non-filtered points

        :rtype: SegmentationProperty
            
        """
        
        # Defining values of unsent parameters
        if (xmin == None):
            xmin = min(points.X())
            
        if (xmax == None):
            xmax = max(points.X())
            
        if (ymin == None):
            ymin = min(points.Y())
            
        if (ymax == None):
            ymax = max(points.Y())
            
        if (zmin == None):
            zmin = min(points.Z())
            
        if (zmax == None):
            zmax = max(points.Z())
            
        # Finding the indices of all the points inside the defined bounds
        insidePointIndices = nonzero(logical_and(logical_and(logical_and(points.X() >= xmin, points.X() <= xmax),
                                                              logical_and(points.Y() >= ymin, points.Y() <= ymax)),
                                                  logical_and(points.Z() >= zmin, points.Z() <= zmax)))
        
        return FilterFactory.__CreateSegmentationProperty(points, insidePointIndices)
    
    @staticmethod
    def FilterBySphericalCoordinates(sphCoorProp, minAzimuth = 0, maxAzimuth = 360,
                                     minElevationAngle = -90, maxElevationAngle = 90, minRange = 0, maxRange = None):
        """
        Filtering based on spherical coordinates values
        """
        
        if (sphCoorProp == None):
            return None
            
        # If maxRange was not provided, defining as the highest range
        if (maxRange == None):
            maxRange = max(sphCoorProp.Ranges())
        
        # Finding the indices of all the points inside the defined bounds
        insidePointIndices = nonzero(logical_and(logical_and(logical_and(sphCoorProp.Azimuths() >= minAzimuth, 
                                                                          sphCoorProp.Azimuths() <= maxAzimuth),
                                                              logical_and(sphCoorProp.ElevationAngles() >= minElevationAngle, 
                                                                          sphCoorProp.ElevationAngles() <= maxElevationAngle)),
                                                  logical_and(sphCoorProp.Ranges() >= minRange, 
                                                              sphCoorProp.Ranges() <= maxRange)))

        return FilterFactory.__CreateSegmentationProperty(sphCoorProp.Points(), insidePointIndices)

    @staticmethod
    def SmoothPointSet_MLS(pointset, radius, polynomial_order, polynomial_fit):
        r"""
        Smoothing with pcl's Moving Least Squares (MLS) for data smoothing and imporved normal estimation

        :param pointset: PointSet  to smooth
        :param radius:  radius that is to be used for determining the k-nearest neighbors used for fitting.
        :param polynomial_order: Set the order of the polynomial to be fit.
        :param polynomial_fit: Set the surface and normal are approximated using a polynomial, or only via tangent estimation.

        :type pointset: PointSet
        :type radius: float
        :type polynomial_fit: bool
        :type polynomial_order: int

        :return:

        .. warning::
           NOT WORKING. PCL-PYTHON CANNOT BE IMPORTED. WAS NOT DEBUGGED
        """
        import pcl
        import numpy as np

        p = pcl.PointCloud()
        p.from_array(pointset.ToNumpy(), dtype=np.float32)
        mls = p.make_moving_least_squares()
        mls.set_search_radius(radius)
        mls.set_polynomial_order(polynomial_order)
        mls.set_polynomial_fit(polynomial_fit)
        smoothed_p = mls.reconstruct()

        return PointSet(smoothed_p.to_array())

    @staticmethod
    def smooth_simple(neighbors_property):
        """
        Smoothing by replacement of each point with its neighbors average value

        :param neighbors_property: the neighborhood property for averaging

        :type neighbors_property: NeighborsProperty

        :return: new smoothed pointset

        :rtype: PointSet
        """
        import numpy as np
        from tqdm import tqdm
        smoothed_pcl_list = list(
            map(lambda neighborhood: np.mean(neighborhood.neighbors.ToNumpy(), axis=0),
                tqdm(neighbors_property, total=neighbors_property.Size, leave=True, position=0)))
        # create a class according to the neighbors' points class and populate it with the smoothed points
        smoothed_pcl = type(neighbors_property.Points).__new__(type(neighbors_property.Points))
        smoothed_pcl.__init__(np.asarray(smoothed_pcl_list))
        smoothed_neigborhood = NeighborsProperty(smoothed_pcl)
        smoothed_neigborhood.setNeighborhood(range(neighbors_property.Size),
                                             neighbors_property.getNeighborhood(range(neighbors_property.Size)))

        return smoothed_neigborhood

    @staticmethod
    @jit
    def __slopeBasedMorphologicFilter(pntData, searchRadius, slopeThreshold):
        """
        Runs slope based morphological filter via jit

        :param pntData:
        :param searchRadius:
        :param slopeThreshold:

        :return:

        """
        numPoints = len(pntData)
        groundPointsIndices = []

        slopeThreshold = slopeThreshold * pi / 180

        for i in range(numPoints):
            isGround = True
            for j in range(numPoints):
                dist = sqrt((pntData[i, 0] - pntData[j, 0]) ** 2 + (pntData[i, 1] - pntData[j, 1]) ** 2)
                if (dist < searchRadius and pntData[i, 2] > pntData[j, 2]):
                    slope = arctan((pntData[i, 2] - pntData[j, 2]) / dist)
                    if (slope > slopeThreshold):
                        isGround = False
                        break

            if (isGround):
                groundPointsIndices.append(i)

        return groundPointsIndices

    @staticmethod
    def __CreateSegmentationProperty(points, indices):
        numPoints = points.Size()
        segments = ones((numPoints), dtype=uint8)
        segments[indices] = 0

        return SegmentationProperty(points, segments)


if __name__ == '__main__':
    pass
    #     pointSetList = []
    #     IOFactory.ReadXYZ('..\\Sample Data\\DU9_2.xyz', pointSetList)
    #
    # #    filterFactory = FilterFactory()
    #     terrainSubSet = FilterFactory.SlopeBasedMorphologicFilter(pointSetList[0], 1.0, 25).GetSegment(0)
    #
    #     Visualization.RenderPointSet(terrainSubSet, 'color', color=(0.5, 0, 0))
    #     Visualization.Show()
