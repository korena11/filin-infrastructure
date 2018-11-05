import numpy as np

import RotationUtils
from PointSet import PointSet
from SaliencyProperty import SaliencyProperty
from TensorProperty import TensorProperty


class SaliencyFactory(object):

    @staticmethod
    def pointwise_tensor_saliency(tensor_property, principal_components_number=3, weights=1):
        r"""
        Compute saliency according to PCA (tensors) similarity.

        According to :cite:`Guo.etal2018`, with the following steps:

        1. Perform Cholesky factorization
        2. Compute sigma sets for each point (or patch)

            .. math::

                S_i = \{ \alpha {\bf M}_1, ..., \alpha {\bf M}_d,...,-\alpha {\bf M}_1,..., -\alpha {\bf M}_d\}

            with :math:`\alpha = \sqrt{d}` and :math:`d` the covariance dimension
        3. Compute the average sigma set for the entire point cloud

            .. math::

                S_{AVG} = \frac{1}{N}\sum_{i=1}^N S_i

            with :math:`N` number of points in the point cloud

        4. Rotate each :math:`S_i` to the :math:`k`-th eigenvector of the covariance matrix (:math:`S_i^{R_k}`)
        5. Compute the distance of each rotated sigma set to the average sigma set:

            .. math::

                G_i = \sum_k \left|S_i^{R_k} - S_{AVG}\right|

        :param tensor_property: the patches or neighbors of all points
        :param principal_components_number: the number of principal components to use. Default: 3.
        :param weights: weights for the k-th principal component. Default: all equal 1

        :type tensor_property: TensorProperty
        :type principal_components_number: int
        :type weights: np.array


        :return: saliency property for the cloud
        :rtype: SaliencyProperty
        """

        S = []  # collection of sigma sets
        Srk = []

        if weights == 1:
            # if there are no specific weights for k -- all equal 1.
            weights = np.ones(principal_components_number)

        for tensor in tensor_property.GetAllPointsTensors():
            S.append(SaliencyFactory.__computeSigmaSet(tensor))
            Si = np.array(S[-1])
            SiRk = []

            for k in range(principal_components_number):
                # for each tensor rotate the sigma set according to the k-th principal component
                Rk = RotationUtils.Rotation_2Vectors(tensor.eigenvectors[k], np.array([1, 0, 0]))
                SiRk.append(Rk.dot(Si))

            Srk.append(SiRk)

        Sarray = np.array(S)
        Srkarray = np.array(Srk)
        Savg = Sarray.mean(axis=1)
        G = np.sum(weights * np.sum(np.abs(Srkarray - Savg), axis=1), axis=1)
        return SaliencyProperty(tensor_property.Points, G)

    @staticmethod
    def range_saliency(points, threshold):
        """
        Compute saliency as a function of the distance from a common plane. Points above a specific threshold their saliency will be set to zero.

        :param points: a point cloud
        :param threshold: the threshold above which the points will be set to zero

        :type points: PointSet
        :type threshold: float

        :return: saliency property

        :rtype: SaliencyProperty
        """

        # Compute the common plane
        n, d = SaliencyFactory.__commonPlane_approx(points)
        xyz = points.ToNumpy()[:, :3]

        dists = np.abs(n.dot(xyz.T) + d) / (np.linalg.norm(n))
        # dists*= 0.01
        # dists[dists > threshold * 0.01] = -0.01

        return SaliencyProperty(points, dists)

    @staticmethod
    def __commonPlane(points):
        """
        Compute the common plane. Finds approximation using Gauss-Markov and then finds the real plane using conditional adjustment with variables

        :param points:

        :type points: PointSet

        :return: the adjusted parameter and the residual in each dimension

        :rtype: tuple
        """
        normal_approx, d = SaliencyFactory.__commonPlane_approx(points)
        sigma = 10
        x0 = np.hstack((normal_approx, d))

        n = points.Size  # number of points
        xyz = points.ToNumpy()[:, :3]
        u = 4  # number of parameters (a b c d)

        # Design matrices
        A = np.ones((n, u))
        A[:, :3] = xyz

        while sigma > 0.1:
            a = np.diag(np.ones((n, 3 * n)) * x0[0])
            b = np.diag(np.ones((n, 3 * n)) * x0[1], 1)
            c = np.diag(np.ones((n, 3 * n)) * x0[2], 2)
            B = a + b + c

            M = B.dot(B.T)
            N = A.dot(np.linalg.inv(M).dot(A))

            w = A.dot(xyz.T)
            u = A.T.dot(np.linalg.inv(M).dot(w))
            dx = -np.linalg.solve(N, u)
            x0 += dx

            v = -B.T.dot(np.linalg.inv(M).dot(A.dot(dx) + w))
            sigma = v.T.dot(v) / (n - u)
        return x0, v

    @staticmethod
    def __commonPlane_approx(points):
        r"""
        Compute the approximation for a common plane via least squares s.t. :math:`||{\bf n}||=1`. 

        The model for three points:

        .. math::
            \tilde{{\bf x}} = {\bf x} - {\bf \bar{x}}
            {\bf n}\tilde{\bf x} = 0

        with :math:`{\bf{\bar{x}}` the centroid of the points

        The normal is the eigenvector that relates to the smallest eigenvalue

        :param points: point cloud

        :type points: PointSet

        :return: the approximated parameters (normal vector + d)

        :rtype: tuple
        """
        import MyTools as mt
        n = points.Size

        # find the centroid
        xyz = points.ToNumpy()[:, :3]
        x_bar = np.mean(xyz, axis=0)
        x_ = xyz - x_bar

        # Design matrix
        A = x_.copy()
        N = A.T.dot(A)

        # the solution is the eigenvector that relates to the smallest eigenvalue
        eigval, eigvec = mt.eig(N)  # returned by order

        d = np.mean(eigvec[0, :].dot(xyz.T))

        return eigvec[0, :], d



    @staticmethod
    def __computeSigmaSet(tensor):
        """
        Compute the sigma set of a tensor

        :param tensor: the tensor for which the sigma set will be computed

        :type tensor: Tensor

        :return: sigma set

        :rtype: np.array
        """
        M = np.linalg.cholesky(tensor.covariance_matrix)
        d = tensor.covariance_matrix.dim
        alpha = np.sqrt(d)

        # sigma set:
        return np.hstack((alpha * M, -alpha * M))
