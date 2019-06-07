import cv2
import numpy as np
from tqdm import tqdm, trange

import MyTools as mt
from BaseProperty import BaseProperty
from PointSet import PointSet
from SaliencyProperty import SaliencyProperty
from TensorFactory import TensorFactory
from TensorProperty import TensorProperty

EPS = 1e-10

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
                           weight_distance_sigma=0.05, alpha=0.05, noise_size=0.01,
                           verbose=False):
        r"""
        Computes saliency in each point according to difference in curvature and normal, as a function of the distance

        For each point, the angle between the normals is computed, as well as the difference in curvature magnitude:

        .. math::
            \begin{eqnarray}
            d\kappa_{ij} = |\kappa_i - \kappa_j| \\
            d{\bf N}_{ij} = {\bf N}_i \cdot {\bf N}_j
            \end{eqnarray}

        The distances are weighted as Gaussians: the closest and farthest get low weights.

        The saliency of the point is the sum:

        .. math::
            s = \sum{ w_d e^{-d} + d\kappa \cdot w_{dN} e^{-(d{\bf N} + 1)}}

        :param curvature_property: curvature property computed in advance
        :param curvature_attribute: the attribute according to which the curvature is measured.
        :param normals_property: normals property computed in advance
        :param neighbors_property: the neighborhood property of the point cloud.
        :param weight_distance_sigma: sigma of the Gaussian accoring to which the distances are weighted. Also serves as the sigma for point association. Default: 0.05
        :param verbose: print running messages. Default: False
        :param noise_size: maximal std of the normals deviations for a point to be considered as vegetation. Default: 0.001

        :type curvature_property: CurvatureProperty.CurvatureProperty or np.ndarray
        :type normals_property: NormalsProperty.NormalsProperty
        :type curvature_attribute: str
        :type neighbors_property: NeighborsProperty.NeighborsProperty
        :type weight_distance_sigma: float
        :type noise_size: float
        :type verbose: bool

        :return: saliency values for each point

        :rtype: SaliencyProperty
        """

        from warnings import warn

        # epsilon = stats.norm.ppf(1 - alpha / 2) * noise_size

        tensor_saliency = []
        j = 0
        kstd = []
        kmean = []
        for neighborhood, i in zip(neighbors_property, trange(neighbors_property.Size,
                                                              desc='Curvature Saliency for each neighborhood')):
            if neighborhood.numberOfNeighbors < 3:
                tensor_saliency.append(0)
                continue

            # get all the current values of curvature and normals. The first is the point to which the
            # computation is made
            try:
                current_curvatures = curvature_property.__getattribute__(curvature_attribute)[
                    neighborhood.neighborhoodIndices]
            except:
                try:
                    current_curvatures = curvature_property[neighborhood.neighborhoodIndices]
                except TypeError:
                    warn(
                    'curvature_property has to be either array or CurvatureProperty. Add condition if needed otherwise')
                    return 1

            current_normals = normals_property.getPointNormal(neighborhood.neighborhoodIndices)

            # difference in curvature
            dk = np.abs(current_curvatures[1:] - current_curvatures[0]) / (neighborhood.numberOfNeighbors - 1)
            # dk[np.where(np.abs(dk) < epsilon)] = 0
            dk_normed = (dk - dk.min()) / (dk.max() - dk.min() + EPS)
            # dk = current_curvatures[0]

            kstd.append(np.std(dk))
            kmean.append(np.mean(dk))

            # distances influence
            dist_element = 1 / (2 * np.pi * weight_distance_sigma ** 2) * \
                           np.exp(-neighborhood.distances[1:] ** 2 / (2 * weight_distance_sigma ** 2))
            dist_element_normed = (dist_element - dist_element.min()) / (dist_element.max() - dist_element.min() + EPS)

            # normal influence
            # dn = current_normals[1:, :].dot(current_normals[0, :])
            dn = (np.linalg.norm(current_normals[0, :] - current_normals[1:, :], axis=1)) / (
                        neighborhood.numberOfNeighbors - 1)
            if verbose:
                print('normal mean {a}, normal std {b}'.format(a=dn.mean(), b=dn.std()))

            if dn.std() > noise_size:
                dn = 0
                dk_normed = 0

            tensor_saliency.append(np.sum(dist_element_normed * (dn + dk_normed)))

        print('kmean {a} kstd {b}'.format(a=np.asarray(kmean).mean(), b=np.asarray(kstd).mean()))
        return SaliencyProperty(neighbors_property.Points, tensor_saliency)

    @classmethod
    def multiscale_saliency(cls, saliencies, percentiles=75):
        """
        Aggregates the saliencies computed by multiple scales

        :param saliencies: list of the SaliencyProperty holding the saliency as computed in a specific scale
        :param percentiles: which percentile to take throughout the scales. If different for each scale, should be a list

        :type saliencies: list
        :type percentiles: int, list

        :return: Aggregated saliency property as computed in multiple scales.

        :rtype: SaliencyProperty

        """
        multi_scale_values = np.zeros((saliencies[0].Size, 1))
        if isinstance(percentiles, int):
            percentile_ = np.ones(len(saliencies))
            percentiles *= percentile_

        elif len(percentiles) == 1:
            percentile_ = np.ones(len(saliencies))
            percentiles *= percentile_

        for saliency, percentile in zip(saliencies, percentiles):
            svals = np.asarray(saliency.getPointSaliency())

            # normalize the values of the saliencies to the range of 0-1
            saliency_normed = (svals - svals.min()) / (svals.max() - svals.min() + EPS)

            # take only n-th percentile
            percentile_n = np.percentile(saliency_normed, percentile)
            print('takes the {} percentile'.format(str(percentile)))
            saliency_normed[saliency_normed < percentile_n] = 0

            multi_scale_values += saliency_normed[:, None]

        return SaliencyProperty(saliencies[0].Points, multi_scale_values)


    # -------------------------- Hierarchical Method for Saliency computation -------------------
    @classmethod
    def hierarchical_saliency(cls, neighborhoods, normals, num_bins=8, low_level_percentage=1, sigma=0.05,
                              association_percentage=20, high_level_percentage=10, verbose=True):
        r"""
        Compute saliency by hierarchical method.

        Based on :cite:`Shtrom.etal2013` paper.

        The procedure is divided into three main components:

        #. Low-level saliency, which is composed by Fast Point Feature Histogram (FPFH) and the low-level distinctness accordingly.

        #. Point Association, and

        #.  High-level saliency.

        :param neighborhoods: the neighborhoods computed for the queried point set
        :param normals: the computed normals of the point cloud
        :param num_bins: the number of bins for feature histogram construction
        :param low_level_percentage: threshold for minimal chi-square distance for the low-level distinctness computation (percentage). Default: 1\%
        :param association_percentage: percentage of focus points for point association (percentage). Default: 20\%
        :param high_level_percentage: percentage of points to use for high level distinctness (percentage). Default: 10\%
        :param sigma: for point association
        :param verbose: print inter-running messages.

        :type neighborhoods: NeighborsProperty.NeighborsProperty
        :type normals: NormalsProperty.NormalsProperty
        :type num_bins: int
        :type low_level_percentage: float
        :type association_percentage: float
        :type high_level_percentage: float
        :type sigma: float
        :type verbose: bool

        :return: saliency property with the values of the saliency

        :rtype: SaliencyProperty

        .. seealso::
            :meth:`SaliencyFactory._SaliencyFactory__SPFH`, :meth:`SaliencyFactory._SaliencyFactory__FPFH`, :meth:`SaliencyFactory._SaliencyFactory__low_level_distinctness`,
            :meth:`SaliencyFactory._SaliencyFactory__point_association`, :meth:`SaliencyFactory._SaliencyFactory__high_level_distinctness`
        """
        from MyTools import chi2_distance
        import itertools
        from functools import partial
        import open3d as o3d

        pointset = neighborhoods.Points
        search_param = o3d.KDTreeSearchParamHybrid(radius=neighborhoods.average_neighborhood_radius(),
                                                   max_nn=neighborhoods.average_neighborhood_size())
        k = 0
        # 1. For each point compute the simplified point feature histogram

        spfh_point = list(map(partial(cls.__SPFH, normals=normals, num_bins=num_bins, verbose=False),
                              tqdm(neighborhoods, total=neighborhoods.Size, desc='SPFH...')))
        spfh_property = SPFH_property(pointset, spfh_point)
        # for neighborhood, i in zip(neighborhoods, tqdm(range(neighborhoods.Size), desc='SPFH for all point cloud')):
        #     spfh_point = cls.__SPFH(neighborhood, normals, num_bins, verbose=False)
        #     spfh_property.set_spfh(neighborhood.center_point_idx, spfh_point)

        # 2. For each SPFH compute the FPFH
        # fpfh = []
        fpfh = list(map(partial(cls.__FPFH, spfh_property=spfh_property),
                        tqdm(spfh_property, total=spfh_property.Size, position=0, desc='FPFH... ')))

        # 1 + 2. Compute FPFH open3d
        # fpfh = SaliencyFactory.FPFH_open3d(pointset, search_param).T
        # fpfh = fpfh.tolist()
        # 3. Low level distinctness,
        current_hist = []
        pointset_distances = []

        for i, j in zip(range(pointset.Size),
                        trange(pointset.Size, desc='chi square', position=0)):

            tmp_fpfh = fpfh.copy()
            current_hist = tmp_fpfh.pop(i)

            # 3.1 Compute the Chi-square distance of each histogram to each histogram
            chi_square_dist = chi2_distance(current_hist, np.asarray(tmp_fpfh))
            pointset_distances.append(chi_square_dist)

            # 3.2 Compute low-level dissimilarity of only points that their histogram is close to the current point
        pointset_distances = np.asarray(pointset_distances)
        dmin = low_level_percentage / 100 * pointset_distances.max(axis=1)
        D_low = list(
            map(cls.__low_level_distinctness, tqdm(pointset, total=pointset.Size, desc='low level distinctness',
                                                   position=0), itertools.repeat(pointset), pointset_distances, dmin))
        D_low = np.asarray(D_low)
        sorted_idx = np.argsort(D_low)  # Sort the low level distinctness according to magnitude

        # 4. Point association
        A_low = []

        # Use the only percentage of the highest low level distinctness to compute the high-level one
        num_points_assoc = int(association_percentage / 100 * pointset.Size)
        focus_points_idx = sorted_idx[-num_points_assoc:]
        focus_points = pointset.GetPoint(focus_points_idx)
        A_low = list(map(cls.__point_association, tqdm(pointset, total=pointset.Size,
                                                       desc='point association', position=0),
                         itertools.repeat(focus_points), itertools.repeat(focus_points_idx),
                         itertools.repeat(D_low), itertools.repeat(sigma)))

        # 5. High level distinctness
        # Use the only percentage of the highest low level distinctness to compute the high-level one
        num_points = int(high_level_percentage / 100 * pointset.Size)
        valid_idx = sorted_idx[-num_points:]
        valid_pts = pointset.GetPoint(valid_idx)
        D_high = list(map(cls.__high_level_distinctness, tqdm(pointset, total=pointset.Size,
                                                              desc='high-level distinctness', position=0),
                          itertools.repeat(valid_pts), itertools.repeat(valid_idx), pointset_distances))

        A_low = np.asarray(A_low)
        D_high = np.asarray(D_high)

        return SaliencyProperty(pointset, 0.5 * (D_low + A_low) + 0.5 * D_high)

    @classmethod
    def __high_level_distinctness(cls, point, valid_pts, valid_idx, chi_dist):
        r"""
        Compute the high-level distinctness of a point within a pointset

        #. High-level dissimilarity measure is computed by
            .. math::
                d_H(p_i,p_j) = D_{\chi^2}(p_i,p_j) \cdot \log\left(1+||p_i-p_j||\right)

        #. The high-level distinctness is computed according to a percentage of the points with the highest low-level distinctness, and is defined by:
            .. math::
                D_{high}(p_i) = 1-\exp\left(-\frac{1}{N}\sum d_H(p_i,p_j)\right)

            with N the number of points that correspond to the percentage of the highest low level distinctness

        :param point: the 3d coordinates of the current point
        :param valid_pts: \% of the points with the highest low-level distinctness
        :param valid_idx: indices of the valid points

        :type point: np.ndarray 3x1
        :type valid_pts: np.ndarray nx3
        :type valid_idx: list, np.ndarray, int
        :type chi_dist: np.array

        :return: the high-level distinctness of point :math:`p_i`.

        :rtype: float
        """

        # 1. Compute the high-level dissimilarity
        pi_pj = np.linalg.norm(point - valid_pts, axis=1)
        pi_pj /= pi_pj.max()
        dh = chi_dist[valid_idx] * np.log(1 + pi_pj)

        # 2. Compute the high-level distinctness
        return 1 - np.exp(-dh.mean())

    @classmethod
    def __low_level_distinctness(cls, point, pointset, chi_dist, dmin):
        r"""
        Compute the lowlevel distinctness for a point.

        #. Dissimilarity measure is computed by

            .. math::
                d_L(p_i, p_j) = \frac{D_{\chi^2}(p_i, p_j)}{1+||p_i-p_j||}

        #.  For points with :math:`D_{\chi^2}(p_i,p_j)<d_{\min}, \forall j` the low-level distinctness
        of point :math:`p_i` is computed by

           .. math::
               D_{low}(p_i) = 1-\exp\left(-\frac{1}{N}\sum d_L(p_i,p_j) \right)

        :param point: the 3d coordinates of the point to which the low-level distinctness is computed
        :param pointset: the points to which the computed distinctness is related
        :param chi_dist: the computed :math:`D_{\chi^2}` for point :math:`p_i` with all other points
        :param dmin: the threshold distance for distinctness measure

        :type point: np.ndarray 3x1
        :type pointset: BallTreePointSet.BallTreePointSet, PointSetOpen3D.PointSetOpen3D
        :type chi_dist: np.array
        :type dmin: float

        :return: the distinctness value of point :math:`p_i`.

        :rtype: float
        """

        valid_idx = np.argwhere(chi_dist <= dmin).flatten()

        if valid_idx.shape[0] < 1:
            return 0

        valid_pts = pointset.GetPoint(valid_idx)
        pi_pj = np.linalg.norm(point - valid_pts, axis=1)
        dl = chi_dist[valid_idx] / (1 + pi_pj)

        # 3.3 Compute the low-level distinctness
        return 1 - np.exp(-dl.mean())

    @classmethod
    def __point_association(cls, point, focus_points, focus_points_idx, distinctness, sigma=0.05):
        r"""
        Associate points that neighbor high-distinctness points with their distinction.

        A fraction (according to percentage) of points that have high low-level distinctness are chosen as focus points.
        :math:`p_{f_i}` is the closest focus point to :math:`p_i` and :math:`D_{foci}(p_i)` is the low-level
        distinctness of :math:`p_{f_i}`, then:

        .. math::
            A_{low}(p_i) = D_{foci}(p_i) \cdot \exp\left(-\frac{||p_{f_i} - p_i||^2}{2\sigma^2}\right)

        :param point: the current point to which the association is computed
        :param focus_points: a list of the focus points
        :param focus_points_idx: the id of the focus points
        :param distinctness: low-level distinctness computed for the cloud
        :param num_points: the number of points to consider as focus points
        :param sigma: a parameter for decay

        :type point: np.ndarray 3x1
        :type focus_points: np.ndarray nx3
        :type focus_points_idx: np.ndarray, list, int
        :type distinctness: np.ndarray
        :type sigma: float

        :return: the association value for each point

        :rtype: np.ndarray
        """

        # 1. Find the distances between current point and the focus points
        distances = np.linalg.norm(point - focus_points, axis=1)

        # 2. The point association is of the closest foci point
        foci_point_id = focus_points_idx[np.argmin(distances)]
        D_foci = distinctness[foci_point_id]

        return D_foci * np.exp(- distances.min() ** 2 / (2 * sigma ** 2))

    @classmethod
    def __SPFH(cls, neighborhood, normals, num_bins=11, verbose=False):
        r"""
        Compute the Simplified Point Feature Histogram.

        #. Find the Darboux frame for each point :math:`u =  {\bf N}, \, v = (p_t-p)\times u, \, w = u\times v` with :math:`p_t` the current point and :math:`{\bf N}` the normal at that point.

        #. The SPFH (simplified point feature histogram) is computed by:

            .. math::

                \begin{array}
                \alpha = v \cdot {\bf N_t} \\
                \phi = u \cdot \frac{p_t-p}{||p_t-p||} \\
                \theta = \arctan(w\cdot {\bf N_t}, u\cdot {\bf N_t})
                \end{array}

        with :math:`p_t` a point in the neighborhood and :math:`{\bf N_t}` the normal of each neighbor.

        :param neighborhood: the current neighborhood to which the SPFH is computed
        :param normals: the normals of the point cloud
        :param num_bins: number of bins in the histogram. Default: 11 (according to :cite:`Shtrom.etal2013` -- 11)
        :param verbose: print inter-running messages. Default: False

        :type neighborhood: PointNeighborhood.PointNeighborhood
        :type normals: NormalsProperty.NormalsProperty
        :type num_bins: int
        :type verbose: bool

        :return: array of :math:`\alpha, \phi, \theta`.

        :rtype: np.array

        """
        # Construct the size of the descriptor
        bins = np.linspace(0, 2 * np.pi, num_bins + 1)

        # If the neighborhood is too small (less than two points)
        if neighborhood.neighbors.Size <= 3:
            hist_alpha = (np.zeros(bins.shape[0] - 1), bins)
            hist_phi = (np.zeros(bins.shape[0] - 1), bins)
            hist_theta = (np.zeros(bins.shape[0] - 1), bins)
            return SPFH_point(neighborhood, hist_alpha, hist_phi, hist_theta)

        # Prevent division in 0
        idx = np.nonzero(neighborhood.distances != 0)
        neighbrhood_normals = normals.getPointNormal(neighborhood.neighborhoodIndices[idx])

        # 1. Build the Darboux frame with each point in the neighborhood
        u = normals.getPointNormal(neighborhood.center_point_idx)
        v = np.cross(neighborhood.neighbors_vectors(), u)
        w = np.cross(u, v)

        # 2. Build feature vector
        alpha = np.diag(v.dot(neighbrhood_normals.T))
        alpha.setflags(write=True)
        phi = u.dot(neighborhood.neighbors_vectors().T)

        u_nt = u.dot(neighbrhood_normals.T)
        w_nt = np.diag(w.dot(neighbrhood_normals.T))

        theta = np.arctan(w_nt / u_nt)

        # Require only positive angles
        alpha[np.where(alpha < 0)] = alpha[np.where(alpha < 0)] + 2 * np.pi
        phi[np.where(phi < 0)] = phi[np.where(phi < 0)] + 2 * np.pi
        theta[np.where(theta < 0)] = theta[np.where(theta < 0)] + 2 * np.pi

        hist_alpha = np.histogram(alpha, bins)
        hist_phi = np.histogram(phi, bins)
        hist_theta = np.histogram(theta, bins)

        tmp_alpha = hist_alpha[0] / neighborhood.numberOfNeighbors
        tmp_phi = hist_phi[0] / neighborhood.numberOfNeighbors
        tmp_theta = hist_theta[0] / neighborhood.numberOfNeighbors

        hist_alpha = (tmp_alpha, bins)
        hist_phi = (tmp_phi, bins)
        hist_theta = (tmp_theta, bins)

        if verbose:
            from matplotlib import pyplot as plt
            figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.set_title('alpha')
            ax2.set_title('phi')
            ax3.set_title('theta')
            ax1.hist(alpha, bins)
            ax2.hist(phi, bins)
            ax3.hist(theta, bins)

        # 4. Construct the auxilary class SPFH_point
        return SPFH_point(neighborhood, hist_alpha, hist_phi, hist_theta)

    @classmethod
    def __FPFH(cls, spfh_current, spfh_property):
        r"""
        Compute Fast Point Feature Histogram (FPFH)

        For each point in spfh_propety, the FPFH is computed by

            .. math::
                FPFH(p) = SPFH(p) + \frac{1}{K} \sum_{k=1}^K \frac{SPFH(p_k)}{||p-p_k||}

        with :math:`K` the number of neighbors of p.

        :param spfh_property: the auxiliary property holding all the SPFH computed for the point cloud

        :return: a feature vector for each point

        :rtype: np.ndarray

        """
        if spfh_current.neighborhood.numberOfNeighbors < 2:
            return spfh_current.getFeatureVector()

        spfh_neighbors = spfh_property.getPointSPFH(spfh_current.neighborhood.neighborhoodIndices[1:])

        spfh_neighbors_normalized = list(
            map(lambda spfh_neighbor, distance_neighbor: spfh_neighbor.getFeatureVector() / distance_neighbor,
                spfh_neighbors, spfh_current.neighborhood.distances[1:]))

        # [spfh_neighbor.getFeatureVector() / distance_neighbor for spfh_neighbor,
        #                                                                                   distance_neighbor in
        #                         ]
        return spfh_current.getFeatureVector() + np.mean(np.asarray(spfh_neighbors_normalized), axis=0)

    @classmethod
    def FPFH_open3d(self, pointset_open3d, knn_search_param):
        """
        Compute the FPFH via open3d.

        .. note::
            This is implemented as proposed in :cite:`Rusu.etal2009` and not modified to :cite:`Shtrom.etal2013`.
            This means that the SPFH is not defined as absolute values of the angles, but as the angles themselves.

        :param pointset_open3d: the point cloud as PointSetOpen3D. if a PointSet is received it is change to PointSetOpen3D.
        :param knn_search_param: the type of search: knn, rnn or hybrid.

        :type pointset_open3d: PointSet, PointSetOpen3D.PointSetOpen3D
        :type knn_search_param: class

        :return: feature vector FPFH

        :rtype: np.array

        ** Usage example **

        .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
           :lines: 50-60
           :emphasize-lines: 10
           :linenos:

        """

        import open3d as o3d
        from PointSetOpen3D import PointSetOpen3D

        if isinstance(pointset_open3d, PointSetOpen3D):
            fpfh = o3d.registration.compute_fpfh_feature(pointset_open3d.data, knn_search_param)
        else:
            p3d = PointSetOpen3D(pointset_open3d)
            fpfh = o3d.registration.compute_fpfh_feature(p3d.data, knn_search_param)

        return fpfh.data




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


class SPFH_property(BaseProperty):
    """
    An auxiliary property class for SPFH computation
    """
    spfh = None

    def __init__(self, pointset, spfhs=None):
        super(SPFH_property, self).__init__(pointset)
        if spfhs is None:
            self.spfh = np.empty(pointset.Size, SPFH_point)
        else:
            self.spfh = np.asarray(spfhs)

    def set_spfh(self, idx, spfh_point):
        """
        Set the spfh of each point

        :param idx: indices of the point(s) to set
        :param spfh_point: the SPFH point
        :return:
        """
        self.spfh[idx] = spfh_point

    def __next__(self):
        self.current += 1
        try:
            return self.getPointSPFH(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def getPointSPFH(self, idx):
        """

        :param idx:
        :return:

        :rtype: SPFH_point
        """
        return self.spfh[idx]


class SPFH_point(object):
    """
    An auxiliary class for SPFH computation
    """
    alpha_histogram = None
    phi_histogram = None
    theta_histogram = None
    neighborhood = None

    def __init__(self, neighborhood, alpha_hist, phi_hist, theta_hist):
        self.neighborhood = neighborhood
        self.hist_bins = alpha_hist[1]
        self.alpha_histogram = alpha_hist[0]
        self.phi_histogram = phi_hist[0]
        self.theta_histogram = theta_hist[0]

    def getFeatureVector(self):
        return np.hstack((self.alpha_histogram, self.phi_histogram, self.theta_histogram))
