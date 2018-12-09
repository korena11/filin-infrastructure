import numpy as np

from PointSet import PointSet
from SaliencyProperty import SaliencyProperty
from TensorFactory import TensorFactory
from TensorProperty import TensorProperty


class SaliencyFactory(object):

    @staticmethod
    def pointwise_tensor_saliency(tensor_property, principal_components_number=3, weights=1):
        r"""
        Compute saliency according to PCA (tensors) similarity.

        According to :cite:`Guo.etal2018`, with the following steps:

        Given :math:`d \times d` covariance matrix for each points (or patches).

        1. Perform Cholesky factorization for each covariance matrix:

            .. math::

                C_{(d\times d)}={\bf MM}^T

        2. Compute sigma sets for each point (or patch)

            .. math::

                S_{i_{(d\times 2d)}} = \{ \alpha {\bf M}_1, ..., \alpha {\bf M}_d,...,-\alpha {\bf M}_1,..., -\alpha {\bf M}_d\}

            with :math:`\alpha = \sqrt{d}` and :math:`d` the covariance dimension

        3. Compute the average sigma set for the entire point cloud

            .. math::

                S_{AVG_{(2d^2\times 1)}} = \frac{1}{N}\sum_{i=1}^N S_i

            with :math:`N` number of points in the point cloud

        4. Compute Principal components around :math:`S_{AVG}`. The result will be :math:`\lambda_{(2d^2)}` eigenvalues and :math:`\bar{\bf{e}}_{(2d^2\times 1)}` eigenvectors.

        5. Rotate each :math:`S_i` by the :math:`k`-th eigenvector of the covariance matrix (:math:`S_i\cdot{\bar{\bf{e}}_k}`)
        6. Compute the distance of each rotated sigma set to the average sigma set:

            .. math::

                G_i = \sum_k \left|S_i\cdot{\bar{\bf{e}}_k}\right|

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

        if isinstance(weights, int) and weights == 1:
            # if there are no specific weights for k -- all equal 1.
            weights = np.ones(principal_components_number)

        for tensor in tensor_property.GetAllPointsTensors():
            S.append(SaliencyFactory.__computeSigmaSet(tensor))


        Sarray = np.array(S)
        # compute tensor around S_AVG
        s_tensor, S_ref = TensorFactory.tensorGeneral(Sarray)

        # Rotate according to the k-lowest
        eigenvectors = s_tensor.eigenvectors[principal_components_number, :]
        S_eigenvectors = np.diag(Sarray.dot(eigenvectors.T))
        G = np.sum(weights * np.abs(S_eigenvectors))


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
        dists[dists > threshold * 0.01] = -0.01

        return SaliencyProperty(points, dists)

    @staticmethod
    def __commonPlane_approx(points):
        r"""
        Compute the approximation for a common plane via least squares s.t. :math:`||{\bf n}||=1`. 

        The model for three points:

        .. math::
            \tilde{{\bf x}} = {\bf x} - {\bf \bar{x}}
            {\bf n}\tilde{\bf x} = 0

        with :math:`{\bf{\bar{x}}}` the centroid of the points

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
        print(tensor.covariance_matrix)
        M = np.linalg.cholesky(tensor.covariance_matrix)
        d = tensor.covariance_matrix.ndim
        alpha = np.sqrt(d)

        # sigma set:
        return np.hstack((alpha * M, -alpha * M))
