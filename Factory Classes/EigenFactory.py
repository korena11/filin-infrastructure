'''
infraGit
photo-lab-3\Reuma
23, Nov, 2017 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from RasterData import RasterData
from EigenProperty import EigenProperty
from MyTools import computeImageDerivatives

class EigenFactory:

    @staticmethod
    def eigen_PCA(points, rad = None):
        """
        compute eigenvalues and eigenvectors

        :param points: ndarray nx3 points with origin in pnt
        """

        if rad != None:
            sigma = rad / 3
            w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
                -(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2) / (2 * sigma ** 2))
            if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
                w = np.ones(points[:, 0].shape)
        else:
            w = np.ones(points[:, 0].shape)

        pT = np.vstack((np.expand_dims(w * points[:, 0], 0), np.expand_dims(w * points[:, 1], 0),
                        np.expand_dims(w * points[:, 2], 0)))

        C = np.dot(pT, points) / np.sum(w)  # points.shape[0]  # covariance matrix of pointset
        eigVal, eigVec = np.linalg.eig(C)

        return EigenProperty(points, eigenValues = eigVal, eigenVectors = eigVec)

    @staticmethod
    def eigen_Hessian(data, winsize, resolution=1, **kwargs):
        """
        Compute the eigenvalues and eigenvectors from the Hessian Matrix
        :param winsize - the window size for filtering
        :param resolution - filter resolution
        :return: eigenProperty

        """
        #TODO apply for PointSet data

        if isinstance(data, RasterData):
            dx, dy, dxx, dyy, dxy = computeImageDerivatives(data.data, order =2,  ksize = winsize, resolution=resolution)

            # Eigenvalues computation (numerically) - (eq. 3 - 36, Amit Baruch dissertation)
            # instead of constructing a matrix for each pixel
            b = - dyy - dxx
            c = dxx * dyy - dxy**2
            eigMax = np.real((- b + np.sqrt(b**2 - 4 * c))/ 2)
            eigMin = np.real((- b - np.sqrt(b**2 - 4 * c)) / 2)

            return EigenProperty(data, eigenValues = np.array([eigMin, eigMax]))




if __name__ == '__main__':
    pass