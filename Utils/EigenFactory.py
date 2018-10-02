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
from MyTools import computeImageDerivatives
from PointSet import PointSet


class EigenFactory:

    @staticmethod
    def eigen_PCA(points, rad=None, pt_index=-1):
        """
        Compute eigenvalues and eigenvectors about a point.

        :param points: the point set
        :param rad: the radius according to which weights are being computes (if None, then unit weight is used)
        :param pt_index: the index of the point about which the PCA (or tensor) is computed. If about the centroid, use
           (-1). Defualt: (-1)

        :type points: PointSet
        """
        if pt_index == -1:
            ref_point = np.mean(points.ToNumpy())
        else:
            ref_point = points.GetPoint(pt_index)

        deltas = points.ToNumpy() - ref_point

        if rad != None:
            sigma = rad / 3
            w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
                -(deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2) / (2 * sigma ** 2))
            if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
                w = np.ones(points[:, 0].shape)
        else:
            w = np.ones(points.Size)

        pT = np.vstack((np.expand_dims(w * deltas[:, 0], 0), np.expand_dims(w * deltas[:, 1], 0),
                        np.expand_dims(w * deltas[:, 2], 0)))

        C = np.dot(pT, deltas) / np.sum(w)  # points.shape[0]  # covariance matrix of the pointset about the point
        eigVal, eigVec = np.linalg.eigh(C)

        return (eigVal, eigVec)

    @staticmethod
    def eigen_Hessian(data, winsize, resolution=1, **kwargs):
        r"""
        Compute the eigenvalues and eigenvectors from the Hessian Matrix

        Eigenvalues computation (numerically) - (eq. 3-36, Amit Baruch dissertation) instead of
        constructing a matrix for each pixel.

        The Hessian is defined as:

        .. math::

           \mathbf{H}= \begin{bmatrix}
           {{Z}_{xx}} & {{Z}_{xy}}  \\
           {{Z}_{yx}} & {{Z}_{yy}}  \\
           \end{bmatrix}

        Substituting it in :math:`\text{det}\left(\mathbf{H}-\lambda \mathbf{I}` leads to:

        .. math::

           {\lambda }^{2}-\lambda \left( {{Z}_{xx}}+{{Z}_{yy}} \right)+\left( {{Z}_{xx}}{{Z}_{yy}}-Z_{xy}^{2}
           \right)=0

        Solving results in the maximum and minimum eigenvalues:

        .. math::

           {\lambda }_{\min ,\max }=\frac{{{Z}_{xx}}+{{Z}_{yy}}\pm \sqrt{{{\left( {{Z}_{xx}}-{{Z}_{yy}} \right)}^{2}}
           +4\cdot Z_{xy}^{2}}}{2}

        .. warning::
           This function does not compute eigenvectors, *only* minimal and maximal eigen values.

        :param winsize - the window size for filtering
        :param resolution - filter resolution

        :return: eigen values and eigen vectors
        :rtype: tuple

        """
        # TODO apply for PointSet data

        if isinstance(data, RasterData):
            dx, dy, dxx, dyy, dxy = computeImageDerivatives(data.data, order=2, ksize=winsize, resolution=resolution)

            # Eigenvalues computation (numerically) - (eq. 3 - 36, Amit Baruch dissertation)
            # instead of constructing a matrix for each pixel
            b = - dyy - dxx
            c = dxx * dyy - dxy ** 2
            eigMax = np.real((- b + np.sqrt(b ** 2 - 4 * c)) / 2)
            eigMin = np.real((- b - np.sqrt(b ** 2 - 4 * c)) / 2)

            return np.array([eigMin, eigMax]), None
