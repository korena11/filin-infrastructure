import cv2
import numpy as np

import MyTools as mt
from PointSet import PointSet
from SaliencyProperty import SaliencyProperty
from TensorFactory import TensorFactory
from TensorProperty import TensorProperty


class SaliencyFactory(object):

    @classmethod
    def pointwise_pca_saliency(cls, tensor_property, principal_components_number=3, weights=1, verbose=False):
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
        :param verbose: print running comments (Default: False)

        :type tensor_property: TensorProperty
        :type principal_components_number: int
        :type weights: np.array
        :type verbose: bool

        :return: saliency property for the cloud

        :rtype: SaliencyProperty

        **Usage example**

        .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
            :lines: 13-21
            :emphasize-lines: 4
            :linenos:

        """

        S = []  # collection of sigma sets

        if isinstance(weights, int) and weights == 1:
            # if there are no specific weights for k -- all equal 1.
            weights = np.ones(principal_components_number)
        counter = 0
        print('>>> Compute sigma-sets for all tensors')
        for tensor in tensor_property.GetAllPointsTensors():

            if verbose:
                counter += 1
                print(counter)
            if tensor is None:
                S.append(np.zeros((principal_components_number * 2 * principal_components_number)))
            else:
                S.append(SaliencyFactory.__computeSigmaSet(tensor, verbose=verbose))

        Sarray = np.stack(S, axis=0)
        # compute tensor around S_AVG
        s_tensor, S_ref = TensorFactory.tensorGeneral(Sarray)

        # Rotate according to the k-lowest
        print('>>> Compute eigenvectors and projection on the average sigma-set')
        eigenvectors = s_tensor.eigenvectors[:principal_components_number, :]
        S_eigenvectors = (Sarray.dot(eigenvectors.T))
        G = np.sum(weights * np.abs(S_eigenvectors), axis=1)
        if verbose:
            print('G', G)
            print('S_eigenvectors', S_eigenvectors)

        return SaliencyProperty(tensor_property.Points, G)

    @classmethod
    def curvature_saliency(cls, neighbors_property, normals_property, curvature_property, curvature_attribute,
                           weight_distance=1, weight_normals=1, verbose=False):
        r"""
        Computes saliency in each point according to difference in curvature and normal, as a function of the distance

        For each point, the angle between the normals is computed, as well as the difference in curvature magnitude:

        .. math::
            \begin{eqnarray}
            d\kappa_{ij} = |\kappa_i - \kappa_j| \\
            d{\bf N}_{ij} = {\bf N}_i \cdot {\bf }_j
            \end{eqnarray}

        The saliency of the point is the sum of:

        .. math::
            s = \sum{d\kappa \cdot w_d e^{-d} \cdot w_{dN} e^{-d{\bf N} + 1}}

        :param curvature_property: curvature property computed in advance
        :param curvature_attribute: the attribute according to which the curvature is measured.
        :param normals_property: normals property computed in advance
        :param neighbors_property: the neighborhood property of the point cloud.
        :param weight_distance: weights for distance element. Default: 1.
        :param weight_normals: weights for normal element. Default: 1.
        :param verbose: print running messages. Default: False

        :type curvature_property: CurvatureProperty.CurvatureProperty or np.ndarray
        :type normals_property: NormalsProperty.NormalsProperty
        :type curvature_attribute: str
        :type neighbors_property: NeighborsProperty.NeighborsProperty
        :type weight_normals: float
        :type weight_distance: float
        :type verbose: bool

        :return: saliency values for each point

        :rtype: SaliencyProperty
        """
        import CurvatureProperty
        from warnings import warn

        tensor_saliency = []

        for neighborhood in neighbors_property:
            # get all the current values of curvature and normals. The first is the point to which the
            # computation is made
            if isinstance(curvature_property, CurvatureProperty.CurvatureProperty):
                current_curvatures = curvature_property.__getattribute__(curvature_attribute)[
                    neighborhood.neighborhoodIndices]
            elif isinstance(curvature_property, np.ndarray):
                current_curvatures = curvature_property[neighborhood.neighborhoodIndices]
            else:
                warn(
                    'curvature_property has to be either array or CurvatureProperty. Add condition if needed otherwise')
                return 1

            current_normals = normals_property[neighborhood.neighborhoodIndices, :]

            # difference in curvature
            dk = current_curvatures[1:, :] - current_curvatures[0, :]

            # distances influence
            dist_element = weight_distance * np.exp(-neighborhood.distances[1:])

            # normal influence
            dn = current_normals[1:, :].dot(current_normals[0, :])
            normal_element = weight_normals * np.exp(-(dn + 1))

            tensor_saliency.append(np.sum(dk * dist_element * normal_element))

        return SaliencyProperty(neighbors_property.Points, np.asarray(tensor_saliency))




    # ------------------- Saliency on Panorama Property ---------------------
    @classmethod
    def panorama_frequency(cls, panorama_property, filters, sigma_sent=True, feature='pixel_val'):
        """
        Compute frequency saliency map for a panorama property (its image).

        The saliency is computed according to either the pixel value in the panorama, LAB or normals.
        Where the distance for saliency is computed between blurred and unblurred images :cite:`Achanta.etal2009`.

        :param panorama_property: the property according to which the saliency is computed
        :param filters: list of filters (either sigmas or kernel sizes) to run with the DoG filter.
        :param sigma_sent: sigma sizes are in ``filters`` (true) as opposed to kernel sizes (false). (default: True)
        :param feature: according to which property the saliency is computed, can be:

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

        :type panorama_property: PanoramaProperty.PanoramaProperty
        :type filters: list
        :type sigma_sent: bool
        :type feature: str

        .. note::
           Other features can be added.



        :return: saliency image

        :rtype: numpy.array

        **Usage example**

        .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
           :lines: 25-32
           :emphasize-lines: 5
           :linenos:

        .. code-block:: python

            sigma = 2.5
            s1 = Saliency.Factory.panorama_frequency(color_panorama,
            filters = [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma], feature = 'LAB')

            s2 = Saliency.Factory.panorama_frequency(normals_panorama, filters= [3, 5, 7],
            feature = 'pixel_val')

        """

        image = panorama_property.PanoramaImage.astype(np.float32)

        # if the image feature is CIELAB, the image should be transformed to CIELab
        if feature == 'LAB':
            # image should have three dimensions in order to be transformed to LAB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # compute difference of gaussians according to kernel or sigma sizes
        if sigma_sent:
            blurred_images = [mt.DoG_filter(image, sigma1=filter1, sigma2=filter2) for filter1, filter2 in
                              zip(filters[:-1], filters[1:])]
        else:
            blurred_images = [mt.DoG_filter(image, ksize1=filter1, ksize2=filter2) for filter1, filter2 in
                              zip(filters[:-1], filters[1:])]

        # compute difference between blurred images
        image_blurred = [image - blurred for blurred in blurred_images]
        s = np.array(image_blurred)

        if feature == 'LAB':
            saliency_map = np.linalg.norm(s, axis=3)

        else:  # other features can be added, only this should be changed to "elif feature == 'pixel_val'"
            saliency_map = np.abs(s)

        return np.mean(saliency_map, axis=0)

    @classmethod
    def panorama_contrast(cls, panorama_property, region_size, feature='pixel_val'):
        """
        Compute local saliency map (contrast based) for a panorama property (its image).

        The saliency is computed according to either the pixel value in the panorama, LAB. The saliency is based on
        distances between regions :cite:`Achanta.etal2008`.


        :param panorama_property: the property according to which the saliency is computed
        :param region_size: the region (R1) size that does not change throughout
        :param feature: according to which property the saliency is computed, can be:

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

            .. note::

                Other features can be added.

        :type panorama_property: PanoramaProperty.PanoramaProperty
        :type filters: int
        :type feature: str

        :return: saliency image

        :rtype: numpy.array

        **Usage example**

        .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
           :lines: 34-40
           :emphasize-lines: 4
           :linenos:
        """

        image = panorama_property.PanoramaImage.astype(np.float32)

        # if the image feature is CIELAB, the image should be transformed to CIELab
        if feature == 'LAB':
            # image should have three dimensions in order to be transformed to LAB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # 1. Creating the kernels
        k_R1 = np.ones((region_size, region_size)) * (1 / region_size ** 2)

        row_size, column_size = image.shape[:2]
        k_R2 = [np.ones((int(ksize), int(ksize))) / ksize ** 2 for ksize in [row_size / 2, row_size / 4, row_size / 8]]

        # 2. Create convoluted map according to region1
        map_R1 = cv2.filter2D(image, -1, k_R1)
        maps_R2 = [cv2.filter2D(image, -1, ksize) for ksize in k_R2]

        s = np.array([map_R1 - map_R2 for map_R2 in maps_R2])

        # compute distances on difference images
        if feature == 'LAB':
            saliency_map = np.linalg.norm(s, axis=3)

        else:  # other features can be added, only this should be changed to "elif feature == 'pixel_val'"
            saliency_map = np.abs(s)

        return np.mean(saliency_map, axis=0)

    @classmethod
    def panorama_context(cls, panorama_property, scale_r, scales_number=4, feature='pixel_val',
                         kpatches=64, constant=3, verbose=False):
        r"""
        Compute context saliency (position based) for panorama property (its image)

        .. warning::

            This saliency measure is not implemented correctly and should be reviewed and rewritten

        Saliency is computed based on the distances between regions and their positions :cite:`Goferman.etal2012`.

        :param panorama_property: the property according to which the saliency is computed
        :param scale_r: scale for multiscale saliency, according to which the neighboring scales are defined:

            .. math::
                R_q=\left\{ r, \frac{1}{2}r, \frac{1}{4}r,...\right\}

            according scales_number

        :param scales_number: the number of scales that should be computed. default 4.
        :param feature: according to which property the saliency is computed, can be:

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

            .. note::

                Other features can be added.

        :param kpatches: the number of minimum distance patches. default: 64
        :param threshold: threshold for the color distance. default: 0.2
        :param constant: a constant; default: 3 (paper implementation)
        :param verbose: print inter-running results

        :type panorama_property: PanoramaProperty.PanoramaProperty
        :type scale_r: float
        :type scales_number: int
        :type feature: str
        :type kpatches: int
        :type constant: int
        :type verbose: bool

        :return: saliency map

        **Usage example**

        .. code-block:: python

            s3 = panorama_context(panorama, r_scale = 2, scales_number = 4, feature = 'pixel_val',
            kpatches=128, constant=3)
            s3[s3 < 1.e-5] = 0

       .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
          :lines: 42-48
          :emphasize-lines: 5
          :linenos:
        """
        image = panorama_property.PanoramaImage.astype(np.float32)

        # if the image feature is CIELAB, the image should be transformed to CIELab
        if feature == 'LAB':
            # image should have three dimensions in order to be transformed to LAB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        ksizes = [np.array(scale_r) * (2 ** (-n)) for n in np.arange(scales_number).astype('f')]
        s = [cls.__contextAware(image, ksize.astype('int'), feature,
                                kpatches=kpatches,
                                c=constant,
                                verbose=verbose)
             for ksize in ksizes]
        saliency_map = np.array(s)
        return np.mean(saliency_map, axis=0)

    # ------------------ Private Functions ------------------------

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
    def __computeSigmaSet(tensor, verbose=False):
        """
        Compute the sigma set of a tensor

        :param tensor: the tensor for which the sigma set will be computed
        :param verbose: print running comments (default: False)

        :type tensor: Tensor
        :type verbose: bool

        :return: sigma set

        :rtype: np.array
        """

        if verbose:
            print(tensor.covariance_matrix)
        M = np.linalg.cholesky(tensor.covariance_matrix)

        d = tensor.covariance_matrix.ndim
        alpha = np.sqrt(d)

        # sigma set:
        return np.hstack(((alpha * M).flatten(), (-alpha * M).flatten()))

    @classmethod
    def __contextAware(cls, image, ksize, image_feature, kpatches=64, c=3, verbose=False):
        """
        Saliency computed according to :cite:`Goferman.etal2012` for one scale.

           .. warning::

            This saliency measure is not implemented correctly and should be reviewed and rewritten

        :param image: image on which the saliency is computed
        :param ksize: the sizes of the patches
        :param image_feature: the feature according to which the saliency is computed.

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

            .. note::

                Other features can be added.

        :param kpatches: the number of minimum distance patches. default: 64
        :param c: a constant; c=3 in the paper's implementation. default: 3
        :param verbose: print debugging prints. defualt: False

        :type image: numpy.array
        :type ksize: numpy.array
        :type image_feature: str
        :type kpatches: int
        :type c: int
        :type verbose: bool

        :return: saliency map of the given image

        :rtype: numpy.array

        """
        import warnings

        if type(kpatches) != int:
            warnings.warn('k should be integer, using kpatches=64 instead.', RuntimeWarning)
            kpatches = 64

        # 1. Creating the kernels
        patch = np.ones((ksize, ksize)) / ksize ** 2
        averaged_image = cv2.filter2D(image, -1, patch)

        m, n = image.shape[:2]

        ind_list = np.arange(0, m * n, np.int(ksize / 2), dtype='float')
        saliency = np.zeros((m, n))

        if image_feature == 'pixel_val':
            averaged_image = cv2.normalize(averaged_image, averaged_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_32F)

        # 2. distance between colors of each patch
        for k in ind_list:
            if np.isnan(k):
                continue
            else:

                k = k.astype('int')
                i_current = np.int(k / n)
                j_current = np.int(k % n)
                dcolor, i, j = cls.__dcolor(averaged_image[i_current, j_current], averaged_image, image_feature,
                                            kpatches=kpatches, verbose=verbose)

                d_pos = np.sqrt((i_current - i) ** 2 + (j_current - j) ** 2)
                d = dcolor / (1 + c * d_pos)
                s = 1 - np.exp(-np.mean(d))

                # remove indices of patches that were already found as similar and set their saliency value to the
                # same as the one that was already found
                saliency[i, j] = s
                saliency[i_current, j_current] = s

        return saliency

    @classmethod
    def __dcolor(cls, p_i, image, image_feature, kpatches, verbose=False):
        '''
        computes the most similar patch to a patch i and their indices

           .. warning::

            This  measure is not implemented correctly and should be reviewed and rewritten

        :param p_i: the patch to which the comparison is made
        :param vector: all other patches
        :param image_feature: the feature according to which the dcolor is computed

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

            .. note::

                Other features can be added.
        :param kpatches: the number of minimum distance patches,  dtype = int

        :type p_i: numpy.array
        :type image: numpy.array
        :type image_feature: str
        :type kpatches: int
        :type verbose: bool

        :return: a vector of K most similar dcolors; a vector of K most similar indices

        '''

        dcolors = np.zeros(image.shape[:2])

        m, n = image.shape[:2]
        if image_feature == 'LAB':
            dist = np.zeros(image.shape)
            dist[:, :, 0] = np.sqrt((p_i[0] - image[:, :, 0]) ** 2)
            dist[:, :, 1] = np.sqrt((p_i[1] - image[:, :, 1]) ** 2)
            dist[:, :, 2] = np.sqrt((p_i[2] - image[:, :, 2]) ** 2)
            dcolors = np.linalg.norm(dist, axis=2)

        elif image_feature == 'pixel_val':
            dcolors = np.sqrt((p_i - image) ** 2)

        K_closest = np.argsort(dcolors, axis=None)[:kpatches]
        i = K_closest / n
        j = K_closest % n

        if verbose:
            print(dcolors[i, j])
        return dcolors[i, j], i, j
