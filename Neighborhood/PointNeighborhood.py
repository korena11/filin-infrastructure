import numpy as np
from scipy.spatial import Delaunay


class PointNeighborhood:
    def __init__(self, r, nn, num, idx, dist):
        self.r = r  # Radius of neighborhood
        self.nn = nn  # Max number of neighborhood poinets set
        self.num = num  # Number of points in neighborhood
        self.idx = idx  # Neighborhood points indices
        self.dist = dist  # Distance of neighborhood points from center point

        self.localRotatedNeighbors = None

    def GetRadius(self):
        return self.r

    def GetMaxNN(self):
        return self.nn

    def GetNumberOfNeighbors(self):
        return self.num

    def GetNeighborhoodIndices(self):
        return self.idx

    def GetLocalRotatedNeighborhood(self):
        return self.localRotatedNeighbors

    def SetLocalRotatedNeighbors(self, neighborsArray):
        self.localRotatedNeighbors = neighborsArray.copy()

    def VisualizeNeighborhoodTriangulation(self):
        """
        .. warning::

            Not working
            
        :return:
        """
        flag = 0
        idx = np.arange(len(self.localRotatedNeighbors))

        # simplices returns points IDs for each triangle
        triangulation = (Delaunay(self.localRotatedNeighbors[:, 0:2])).simplices

        tri = np.where(triangulation == 0)[0]  # Keep triangles that have the first point in them
