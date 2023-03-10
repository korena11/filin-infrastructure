import cv2
import numpy as np
from tqdm import tqdm, trange

import Utils.MyTools as mt
from DataClasses.PointSet import PointSet
from DataClasses.RasterData import RasterData
from Properties.BaseProperty import BaseProperty
from Properties.Saliency.SaliencyProperty import SaliencyProperty
from Properties.Tensors.TensorFactory import TensorFactory
from Properties.Tensors.TensorProperty import TensorProperty, Tensor

EPS = 1e-10

class SaliencyFactory(object):

    @staticmethod
    def pointwise_pca_saliency(tensor_property, principal_components_number=3, weights=1, verbose=False):
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

    # ----------------- DIRECTIONAL SALIENCY -----------------------
    @staticmethod
    def directional_saliency(neighbors_property, normals_property, curvature_property, curvature_attribute, weighting_func, neighborhood_property_curvature=None,
                             sigma_normal=0.01, sigma_curvature= 0.1, curvature_weight=.5, alpha=0.05,
                             verbose=False, **kwargs):
        """

        :param neighbors_property: the neighborhood property of the point cloud.
        :param normals_property: normals at the neighborhood
        :param curvature_property: curvature property computed in advance
        :param curvature_attribute: the attribute according to which the curvature is measured.
        :param weighting_func: the weighting function (from WeightingFuncions module, as an example)
        :param neighborhood_property_curvature: the neighborhood property of the point cloud for curvature saliency. If None, both normals and curvature will be using the same neighborhood
        :param sigma_normal: maximal std of the normals deviations for a point to be considered as vegetation. Default: 0.01
        :param sigma_curvature: std of the curvature deviations for a surface texture. Default: 0.1
        :param alpha: for the hypothesis testing. Default: 0.05
        :param curvature_weight: the weight of the curvature in the saliency computation
        :param verbose: print running messages. Default: False
        :param stopat: a point to stop at for debugging, show its neighborhood, and start from that point at the visualization of all points' neighborhood

        :type curvature_property: CurvatureProperty.CurvatureProperty or np.ndarray
        :type normals_property: np.array
        :type curvature_attribute: str
        :type weighting_func: func
        :type neighborhood_property_curvature: NeighborsProperty.NeighborsProperty
        :type neighbors_property: NeighborsProperty.NeighborsProperty
        :type sigma_normal: float
        :type sigma_curvature: float
        :type verbose: bool

        :return: saliency values for each point

        :rtype: SaliencyProperty
        """

        from warnings import warn
        from scipy import stats
        from Properties.Neighborhood.NeighborsProperty import NeighborsProperty
        from Properties.Neighborhood.PointNeighborhood import PointNeighborhood
        from Properties.Normals.NormalsProperty import NormalsProperty
        from DataClasses.PointSubSet import PointSubSet
        from VisualizationClasses.VisualizationO3D import VisualizationO3D

        normal_weight = 1 - curvature_weight
        tensor_saliency = []
        if 'stopat' in kwargs:
            stopat = kwargs.pop('stopat')
        else:
            stopat = 0
        j = 0
        sign_dk = []
        kstd = []
        kmean = []
        dn_ = []
        dk_ = []
        w_neighborhood = NeighborsProperty(neighbors_property.Points)
        if neighborhood_property_curvature is None:
            neighborhood_property_curvature = neighbors_property

        for neighborhood,  i in zip(neighbors_property, trange(neighbors_property.Size, desc='Directional Saliency for each neighborhood', position=0)):
            # print(i)
            if verbose:
                if i==stopat:
                    print('!')
                    
            neighborhood.weightNeighborhood(weighting_func, rho=kwargs['rho'], sigma=kwargs['sigma'])

            if verbose:
                w_neighborhood.setNeighborhood(i, neighborhood)
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

            if neighborhood.numberOfNeighbors < 4:
                dn_.append(0)
                dk_.append(0)
                sign_dk.append(0)
            else:
                if verbose:
                    if i == stopat:
                        vis = VisualizationO3D()
                        print('\n current normals {}'.format(current_normals * neighborhood.weights[:, None]))
                        print('\n current kappas {}'.format(current_curvatures * neighborhood.weights[:, None]))
                        neighborhood.weights[0] = 1
                        tmp_normals = NormalsProperty(neighborhood.neighbors, current_normals * neighborhood.weights[:, None])
                        tmp_normals.__setattr__('k', np.round(current_curvatures * neighborhood.weights))
                        vis.visualize_property(tmp_normals)
                number_active = int(np.sum(neighborhood.weights > 0.01)) # how many of the neighbors take part in the computation
                chi_statistic = stats.chi2.ppf(alpha, number_active)  # X^2-distribution
                dn = SaliencyFactory.__normal_saliency(neighborhood, current_normals, chi_statistic, number_active, sigma_normal, verbose=verbose)
                dn_.append(dn)

                dk = SaliencyFactory.__curvature_saliency(neighborhood, current_curvatures,chi_statistic, number_active, sigma_curvature,  verbose=verbose)
                dk_.append(dk)

                if dk == 0:
                    sign_dk.append(0)
                else:
                    sign_dk.append(np.sign(current_curvatures[0]))
            if verbose:
                print('\n {}: dn {}, dk {}'.format(i, dn_[-1], dk_[-1]))

        dn = mt.scale_values(np.asarray(dn_))
        dk = mt.scale_values(np.asarray(dk_))
        # dn = np.asarray(dn_)
        # dk = np.asarray(dk_)
        # dn = 1 - np.exp(-dn)
        # dk = 1 - np.exp(-dk)
        sign_dk = np.asarray(sign_dk)

        tensor_saliency = (dn * normal_weight + dk * curvature_weight)
        saliency = SaliencyProperty(neighbors_property.Points, tensor_saliency)
        saliency.__setattr__('dn', dn)
        saliency.__setattr__('dk', dk )
        if verbose:
            import matplotlib.pyplot as plt
            vis = VisualizationO3D()
            vis.visualize_neighborhoods(w_neighborhood, saliency, stopat)
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].set_title('Histogram |dk|')
            ax[0, 0].hist(dk[dk>0.001])
            ax[0, 1].set_title('Histogram dk')
            ax[0, 1].hist(dk[np.abs(dk)>0.001] * sign_dk[dk>0.001])
            ax[1, 0].set_title('Histogram dn')
            ax[1, 0].hist(dn[dn>0.001])
            ax[1, 1].set_title('Histogram saliency')
            ax[1, 1].hist(tensor_saliency[np.abs(tensor_saliency)>0.001])


            # ax[1,1].text(0.2,.4, 'min dk {0:.3f}, min dn {0:.3f} \n max dk {0:.3f}, max dn {0:.3f}'.format
            # (round(dk.min()), round(dn.min()), round(dk.max()), round(dn.max())), fontsize=15)
            # ax[1,1].axis('off')

        return saliency

    @staticmethod
    def __normal_saliency(neighborhood, current_normals, chi2, number_effective, sigma_expected=0.01, verbose=False):
        r"""
        Checks the saliency of a point based on  normals property

        .. math::
                d{\bf N}_{ij} = {\bf N}_i \cdot {\bf N}_j

        The distances are weighted as Gaussians (with shift rho): the closest and farthest get low weights.
        
        The significance of the normal saliency is checked using the :math:`chi^2`-test, i.e.,
        
        .. math::
            H_0: \sigma_{\bf N}^2 \leq \hat{\sigma}_0^2 
            H_1: \sigma_{\bf N}^2 > \hat{\sigma}_0^2
        
        So that if 
        
        .. math::
            \frac{\nu \cdot \sigma_{\bf N}^2}{\hat{\sigma}_0^2} < \chi^2_{\nu, \alpha}
        
        We reject the hypothesis. 
               
        :param neighborhood: the neighborhood of a point
        :param current_normals: normals array of the current point
        :param chi2: :math:`\chi^2` from the table (i.e., :math:`\chi^2_{\nu, \alpha}`)
        :param number_effective: effective number of neighbors that take part in the compuatation
        :param sigma_expected: maximal std of the normals deviations (i.e., :math:`\hat{\sigma}_0^2). Default: 0.01
        :param verbose: print running messages. Default: False

        :type current_normals: np.array
        :type neighborhood: PointNeighborhood.PointNeighborhood
        :type chi2: float
        :type number_effective: int
        :type sigma_expected: float
        :type verbose: bool

        :return: array of saliency value of a point in a neighborhood according to normals

        :rtype: np.array
        """
        # normal influence
        dn = 1 - current_normals[1:, :].dot(current_normals[0, :])
        # dn = (np.linalg.norm(current_normals[0, :] - current_normals[1:, :], axis=1)) / (
        #         neighborhood.numberOfNeighbors)
        # dn = mt.scale_values(dn)
        num_zeros = np.sum(np.abs(dn[neighborhood.weights[1:] > 0.01]) < 0.016)

        chi2_statistic = (neighborhood.numberOfNeighbors - 1) * dn.std() / sigma_expected

        # if there are more than 60% that are less than 5 deg difference
        if num_zeros > 0.6 * number_effective:
            dn = 0
        elif chi2_statistic < chi2:
            dn = 0
        else:
            dn = np.sum(np.abs(dn) * neighborhood.weights[1:])/np.sum(neighborhood.weights[1:])
        return dn

    @staticmethod
    def __curvature_saliency(neighborhood, current_curvatures,  chi2, number_effective, sigma_expected=0.01, verbose=False):
        r"""
        Computes saliency in each point according to difference in curvature.

        For each point, the difference in curvature between a close point's surrounding  and its farther area are tested

        .. math::
            d\kappa_{ij} = |\kappa_i - \kappa_j|
            
          The significance of the curvature saliency is checked using the :math:`chi^2`-test, i.e.,
        
        .. math::
            H_0: \sigma_{\kappa}^2 \leq \hat{\sigma}_0^2 
            H_1: \sigma_{\kappa}^2 > \hat{\sigma}_0^2
        
        So that if 
        
        .. math::
            \frac{\nu \cdot \sigma_{\kappa}^2}{\hat{\sigma}_0^2} < \chi^2_{\nu, \alpha}
        
        We reject the hypothesis. 

        :param neighborhood: the point neighborhood.
        :param current_curvatures: curvatures of the points in neighborhood
        :param chi2: :math:`\chi^2` from the table (i.e., :math:`\chi^2_{\nu, \alpha}`)
        :param number_effective: effective number of neighbors that take part in the compuatation
        :param sigma_expected: maximal std of the normals deviations (i.e., :math:`\hat{\sigma}_0^2). Default: 0.01
        :param verbose: print running messages. Default: False

        :type neighborhood: PointNeighborhood.PointNeighborhood
        :type current_curvatures: np.ndarray
        :type number_effective: int
        :type chi2: float
        :type noise_size: float
        :type verbose: bool

        :return: array of saliency value of a point in a neighborhood according to curvature variations

        :rtype: np.array
        """

        # difference in curvature
        dk = current_curvatures[1:] - current_curvatures[0]
        num_zeros = np.sum(np.abs(dk[neighborhood.weights[1:] > 0.01]) <= 0.04)
        chi2_statistic = (neighborhood.numberOfNeighbors - 1) * dk.std() / sigma_expected

        if num_zeros > 0.6 * number_effective:
            dk = 0
        elif chi2_statistic < chi2:
            dk = 0
        else:
            dk = np.abs(np.sum(dk * neighborhood.weights[1:])/np.sum(neighborhood.weights[1:]))
        return dk

    @staticmethod
    def multiscale_saliency(saliencies, percentiles=75):
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
            # take only n-th percentile
            percentile_n = np.percentile(svals, percentile)
            print('takes the {} percentile'.format(str(percentile)))
            svals[svals < percentile_n] = 0

            # normalize the values of the saliencies to the range of 0-1
            saliency_normed = (svals - svals.min()) / (svals.max() - svals.min() + EPS)

            multi_scale_values += saliency_normed[:, None]

        return SaliencyProperty(saliencies[0].Points, multi_scale_values)

    # -------------------------- Hierarchical Method for Saliency computation -------------------
    @classmethod
    def hierarchical_saliency(cls, normals, search_param, low_level_percentage=1, sigma=0.05,
                              association_percentage=20, high_level_percentage=10, verbose=True, chi_filename=None):
        r"""
        Compute saliency by hierarchical method.

        Based on :cite:`Shtrom.etal2013` paper.

        The procedure is divided into three main components:

        #. Low-level saliency, which is composed by Fast Point Feature Histogram (FPFH) and the low-level distinctness accordingly.

        #. Point Association, and

        #.  High-level saliency.

        :param search_param: search parameter from open3d. Can be  o3d.geometry.KDTreeSearchParamKNN(knn),
        o3d.geometry.KDTreeSearchParamRadius(radius) or o3d.geometry.KDTreeSearchParamHybrid(radius, knn)
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
        import open3d as o3d
        from functools import partial

        pointset = normals.Points
        # search_param = o3d.KDTreeSearchParamHybrid(radius=neighborhood_radius,
        #                                            max_nn=neighbornood_maxnn)
        k = 0

        # 1 + 2. Compute FPFH open3d
        fpfh = SaliencyFactory.FPFH_open3d(pointset, normals, search_param).T

        # # 3 + 4. Compute low and high level distinctness
        # D_low, D_high = list(map(partial(cls.__point_hierarchical_saliency,
        #                                  pointset=pointset, low_level_percentage=low_level_percentage, high_level_percentage=high_level_percentage),
        #                          tqdm(np.arange(pointset.Size), total=pointset.Size, desc='low and high level distinctness'), fpfh))

        # 3.1 Compute the Chi-square distance of each histogram to each histogram
        from functools import partial
        import pickle

        try:
            pointset_distances = pickle.load(open(chi_filename, 'rb'))
        except:
            pointset_distances = list(map(partial(chi2_distance, histB=fpfh), tqdm(fpfh, desc='chi square')))
            if chi_filename is not None:
                pickle.dump(pointset_distances, open(chi_filename + '.p', 'wb'))

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
        dh = chi_dist[valid_idx, None] * np.log(1 + pi_pj)

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
        pi_pj = np.linalg.norm(point - valid_pts[1:], axis=1)
        dl = chi_dist[valid_idx[1:]] / (1 + pi_pj)

        # 3.3 Compute the low-level distinctness
        Dl = 1 - np.exp(-dl.mean())
        if np.isnan(Dl):
            Dl = 0
        return Dl

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
    def FPFH_open3d(self, pointset_open3d, normalsproperty, knn_search_param):
        """
        Compute the FPFH via open3d.

        .. note::
            This is implemented as proposed in :cite:`Rusu.etal2009` and not modified to :cite:`Shtrom.etal2013`.
            This means that the SPFH is not defined as absolute values of the angles, but as the angles themselves.

        :param pointset_open3d: the point cloud as PointSetOpen3D. if a PointSet is received it is change to PointSetOpen3D.
        :param knn_search_param: the type of search: knn, rnn or hybrid.
        :param normalsproperty: the normals property of the point cloud

        :type pointset_open3d: PointSet, PointSetOpen3D.PointSetOpen3D
        :type knn_search_param: class
        :type normalsproperty: NormalsProperty.NormalsPropoerty

        :return: feature vector FPFH

        :rtype: np.array

        **Usage example**

        .. literalinclude:: ../../../../Properties/Saliency/test_saliencyFactory.py
           :lines: 50-60
           :emphasize-lines: 10
           :linenos:

        """

        import open3d as o3d
        from DataClasses.PointSetOpen3D import PointSetOpen3D

        if not isinstance(pointset_open3d, PointSetOpen3D):
            p3d = PointSetOpen3D(pointset_open3d)

        else:
            p3d = pointset_open3d

        if not p3d.data.has_normals():
            p3d.data.normals = o3d.Vector3dVector(normalsproperty.Normals)

        fpfh = o3d.registration.compute_fpfh_feature(p3d.data, knn_search_param)
        return fpfh.data


    # ------------------- Saliency on Panorama Property or rasters ---------------------
    @classmethod
    def directional_saliency_by_raster(cls, raster, curvature_raster, normals_tuple, win_size=(3,3), width=1, curvature_weight=.5):
        r"""
        Compute the directional saliency with a moving window on raster. *Returns raster*

        :param raster: the raster data on which the saliency is computed
        :param curvature_raster: computed curvature, as a (m,n) raster
        :param normals_tuple: computed normals as a tuple of (nx, ny, nz) as rasters.
        :param win_size: the window sizes in which the saliency is checked. Must be odd number. Default: (3,3)
        :param width: the width of the band to compare. Must be odd number. For example:

        window :math:`5\times 5` with width 1    | window :math:`5\times 5` with width 3
        -----------------------------------------|--------------------------------------
                0 0 1  0 0                        |  0 1   1 1 0
                0 0 1  0 0                        |  0 1   1 1 0
                0 0 -4 0 0                        |  0 0 -12 0 0
                0 0 1  0 0                        |  0 1   1 1 0
                0 0 1  0 0                        |  0 1   1 1 0
        -----------------------------------------|----------------------------------------

        :param curvature_weight: weighting parameter for saliency computation

        :type raster: RasterData
        :type curvature_raster: np.array
        :type normals_tuple: tuple
        :type win_size: tuple
        :type width: int
        :type curvature_weight: float

        :return: raster with enhanced saliency values according to their difference from specific directions. Note that the returned saliency is normalized between (0,1)

        .. seealso::
            :func:`SaliencyFactory.directional_saliency`

        """
        import warnings
        from scipy.signal import convolve2d

        if win_size[0] % 2 ==0 or win_size[1] %2 ==0 or width %2==0:
            warnings.warn('Window must be odd number in both axes')
            raise WindowsError

        # 1. Construct windows
        horizontal = np.zeros(win_size)
        vertical = np.zeros(win_size)
        diag_NW = np.zeros(win_size)

        n = np.floor(win_size[0] /2).astype('int')
        m = np.floor(win_size[1] /2).astype('int')

        # horizontal and vertical
        horizontal[(m-np.floor(width/2)).astype('int') : (m+np.ceil(width/2)).astype('int'), :] = 1
        vertical[:, (n-np.floor(width/2)).astype('int') : (n+np.ceil(width/2)).astype('int')] = 1

        # set  middle points as minus the sum of the filled pixels
        horizontal[m, n] = -np.sum(horizontal) +1
        vertical[m, n] = -np.sum(vertical) +1

        # diagonal windows
        for i in np.arange(-np.floor(width / 2).astype('int'), np.ceil(width / 2).astype('int')):
            diag_NW += np.diagflat(np.ones((1, np.min((win_size[0], win_size[1]))) - np.abs(i)), i)

        # set middle point as zero and flip to get the NE-SW diagonal
        diag_NW[m,n] = -np.sum(diag_NW) + 1
        diag_NE = np.fliplr(diag_NW) + 1

        windows = [horizontal, vertical, diag_NW, diag_NE]
        rasters = [curvature_raster * curvature_weight]
        normals = []
        for normal in normals_tuple:
            n = normal * (1-curvature_weight)
            rasters.append(n)

        # rasters = rasters + normals # combine lists
        convolved = np.zeros(curvature_raster.shape)

        # 3. convolve all windows with all rasters
        for raster in rasters:
            for kernel in windows:
                convolved += (convolve2d(raster, kernel, mode='same'))
        return SaliencyProperty(raster, mt.scale_values(convolved, 0 ,1))
        #

    @classmethod
    def saliency_raster_gaussian(cls, curvature_property, dn, win_size, rho, sigma, curvature_weight=.5, curvature_attribute ='mean'):
        """
        Computes directional saliency with laplacian

        :param curvature_property: computed curvature, as a (m,n) raster
        :param dn: dn as computed point-wise (but as raster)
        :param win_size: the window sizes in which the saliency is checked (in meters).
        :param rho: the distance of the highest weight
        :param sigma: the Gaussian sigma
        :param curvature_weight: weighting parameter for saliency computation
        :param curvature_attribute: the curvature type which will be used for the computation. Default: mean

        :type curvature_property: CurvatureProperty.CurvatureProperty
        :type dn: np.ndarray
        :type win_size: int
        :type rho: float
        :type sigma: float
        :type curvature_weight: float

        :return: raster with enhanced saliency values according to their difference from specific directions. Note that the returned saliency is normalized between (0,1)
        """
        from scipy.ndimage import convolve
        from Properties.Curvature.CurvatureProperty import CurvatureProperty
        resolution = curvature_property._BaseProperty__dataset.resolution[0]

        dn[np.where(np.isnan(dn))] = 0
        if win_size % 2 ==0:
            center_pix = int(win_size/(2 * resolution))
        else:
            center_pix = int((win_size - 1)/(2 * resolution))

        # create distance map
        x = np.arange(0, win_size, resolution)
        y = np.arange(0, win_size, resolution)
        xx, yy = np.meshgrid(x,y)
        xx -= center_pix * resolution
        yy -= center_pix * resolution
        dist_map = np.sqrt((xx)**2 + (yy)**2)

        # create weights map according to phi and sigma
        window = 1/(np.sqrt(2*np.pi) * sigma) * np.exp(- (dist_map - rho)**2 / (2*sigma**2))
        window[center_pix, center_pix] = -1

        convolved = np.abs(convolve(curvature_property.__getattribute__(curvature_attribute), window)) * curvature_weight + dn * (1 - curvature_weight)

        return SaliencyProperty(curvature_property._BaseProperty__dataset, mt.scale_values(convolved, 0,1))

    @classmethod
    def panorama_frequency(cls, panorama_property, filters, sigma_sent=True, feature='pixel_val'):
        """
        Compute frequency saliency map for a panorama property (its image).

        The saliency is computed according to either the pixel value in the panorama, LAB or normals.
        Where the distance for saliency is computed between blurred and unblurred images :cite:`Achanta.etal2009`.

        :param panorama_property: the property according to which the saliency is computed
        :param filters: list of filters (either sigmas or kernel sizes) to run with the DoG filter.
        :param sigma_sent: sigma sizes are in ''filters'' (true) as opposed to kernel sizes (false). (default: True)
        :param feature: according to which property the saliency is computed, can be:

            - 'pixel_val' - the value of the pixel itself
            - 'LAB' - a feature vector using CIElab color space

        :type panorama_property: PanoramaProperty.PanoramaProperty, np.array, DataClasses.RasterData.RasterData
        :type filters: list
        :type sigma_sent: bool
        :type feature: str


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

         .. note::
           Other features can be added.
        """
        from Properties.Panoramas.PanoramaProperty import PanoramaProperty
        from DataClasses.RasterData import RasterData
        if isinstance(panorama_property, PanoramaProperty):
            image = panorama_property.panoramaImage.astype(np.float32)
        elif isinstance(panorama_property, RasterData):
            image = panorama_property.data.astype(np.float32)
        else:
            image = panorama_property.astype(np.float32)

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

        :type panorama_property: PanoramaProperty.PanoramaProperty, np.array, DataClasses.RasterData.RasterData
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
        from Properties.Panoramas.PanoramaProperty import PanoramaProperty
        from DataClasses.RasterData import RasterData

        if isinstance(panorama_property, PanoramaProperty):
            image = panorama_property.panoramaImage.astype(np.float32)
        elif isinstance(panorama_property, RasterData):
            image = panorama_property.data.astype(np.float32)
        else:
            image = panorama_property.astype(np.float32)

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

        :param panorama_property: the property according to which the saliency is computed, or a raster image
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

        :type panorama_property: PanoramaProperty.PanoramaProperty, np.array, DataClasses.RasterData.RasterData
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
          :lines: 42-49
          :emphasize-lines: 5
          :linenos:
        """
        from Properties.Panoramas.PanoramaProperty import PanoramaProperty
        from DataClasses.RasterData import RasterData

        if isinstance(panorama_property, PanoramaProperty):
            image = panorama_property.panoramaImage.astype(np.float32)
        elif isinstance(panorama_property, RasterData):
            image = panorama_property.data.astype(np.float32)
        else:
            image = panorama_property.astype(np.float32)

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
             for ksize in ksizes if ksize.astype('int') !=0 ]
        saliency_map = np.array(s)
        return np.mean(saliency_map, axis=0)


    @classmethod
    def ballTree_saliency(cls,pointset, scale, neighborhood_properties, curvature_attribute, leaf_size=10,
                          weight_distance=1, weight_normals=1, verbose=False):
        """
        Compute curvature based saliency using ball-tree and k-nearest-neighbors.

        The saliency for every set of points within a ball-tree cell is computed as one for all points.

        .. seealso::
            :meth:`SaliencyFactory.curvature_saliency`

        :param pointset: a point cloud of any sort
        :param scale: the scale of the minimal phenomena (size of the smallest ball-tree cell)
        :param neighborhood_properties: k nearest neighbors
        :param curvature_attribute: the attribute according to which the curvature is measured.

        :param leaf_size: the minimum number of points in the ball tree leaves. Default: 10
        :param weight_distance: weights for distance element. Default: 1.
        :param weight_normals: weights for normal element. Default: 1.
        :param verbose: print running messages. Default: False

        :type pointset: np.ndarray, PointSet.PointSet, BallTreePointSet.BallTreePointSet, PointSetOpen3D.PointSetOpen3D
        :type scale: float
        :type neighborhood_properties: int or float
        :type leaf_size: int
        :type weight_normals: float
        :type weight_distance: float
        :type verbose: bool

        :return: curvature property, normals property, saliency property

        :rtype: CurvatureProperty, NormalsProperty, SaliencyProperty


        .. warning::
           This method is old, not debugged, and returns more than SaliencyProperty.

        .. TODO: Should be updated so that the curvature and normals are inputs and not computed inside. Should also check if this is not redundant
        """

        from CurvatureFactory import CurvatureFactory, CurvatureProperty
        from NeighborsFactory import NeighborsFactory
        from BallTreePointSet import BallTreePointSet
        from PointSubSet import PointSubSet
        from TensorFactory import TensorFactory, TensorProperty
        from NormalsFactory import NormalsFactory, NormalsProperty

        # 1. Build the BallTree
        if isinstance(pointset, BallTreePointSet):
            balltree = pointset

        else:
            balltree = BallTreePointSet(pointset, leaf_size=leaf_size)

        # 2. Choose vertices according to minimal scale
        nodes = balltree.getSmallestNodesOfSize(scale, 'leaves')

        # 3. Build tensors out of each node's points, and build their neighborhoods
        tensors = []
        new_cloud = []
        if verbose:
            i = 0

        for node in nodes:
            if verbose:
                i += 1
                if i == 1192:
                    print('hello')
            tmp_subset = PointSubSet(pointset, balltree.getPointsOfNode(node))
            tensors.append(TensorFactory.tensorFromPoints(tmp_subset, -1))
            new_cloud.append(tensors[-1].reference_point)

        new_bt = BallTreePointSet(np.asarray(new_cloud))  # the new cloud of the centers of mass of each tensor
        tensors_property = TensorProperty(new_bt, np.asarray(tensors))
        neighbors = NeighborsFactory.balltreePointSet_knn(new_bt, neighborhood_properties)  # neighborhood construction

        # 4. Curvature for each tensor
        tensor_curvature = CurvatureFactory.tensorProperty_3parameters(tensors_property)

        # 5. Normal for each tensor
        tensor_normals = NormalsFactory.normals_from_tensors(tensors_property)

        # 6.  Saliency for each tensor
        tensor_saliency = cls.directional_saliency(neighbors, tensor_normals, tensor_curvature,
                                                               curvature_attribute=curvature_attribute, verbose=verbose)

        # 7. Assign saliency value for each point in the tensor
        pcl_saliency = SaliencyProperty(pointset)
        pcl_curvature = CurvatureProperty(pointset)
        pcl_normals = NormalsProperty(pointset)

        for node_, saliency, curvature, normal in zip(nodes, tensor_saliency, tensor_curvature, tensor_normals):
            idx = balltree.getPointsOfNode(node_)
            pcl_saliency.setPointSaliency(idx, saliency)
            pcl_curvature.setPointCurvature(idx, curvature)
            pcl_normals.setPointNormal(idx, normal)

        return pcl_saliency, pcl_curvature, pcl_normals


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
        # import Utils.MyTools as mt
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
        saliency = np.zeros((m, n))

        if np.int(ksize/2) == 0:
            return saliency

        ind_list = np.arange(0, m * n, np.int(ksize / 2), dtype='float')

        if image_feature == 'pixel_val':
            averaged_image = cv2.normalize(averaged_image, averaged_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_32F)

        # 2. distance between colors of each patch
        for k in tqdm(ind_list, total=ind_list.shape[0], desc='distance between colors of each patch'):
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
        i = (K_closest / n).astype('int')
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
