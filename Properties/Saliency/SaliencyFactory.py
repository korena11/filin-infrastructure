import numpy as np

import RotationUtils
from SaliencyProperty import SaliencyProperty


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

        :type tensor_property: Tensors.TensorProperty
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
