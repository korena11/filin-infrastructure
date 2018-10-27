'''
|today|

.. sectionauthor:: Reuma


Implementation of multiple kinds of saliencies a given image:

**Distance based**

1. :cite:`Achanta.etal2009`
2. :cite:`Achanta.etal2008`
3. :cite:`Goferman.etal2012`
4. :cite:`Guo.etal2018`

**Bayesian saliency**

5. :cite:`Xie.etal2013`

**PointCloud**

6. :cite:`Guo.etal2018`


'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as LA
import warnings

import cv2
import MyTools as mt
import LatexUtils as lu


def bayesian(image, **kwargs):
    """
    Saliency computed according to :cite:`Xie.etal2013`


    :param image:
    :param numSegments: number of segments in the superpixel segmentation. defualt: 200
    :param sigma: sigma for the segmentation (???). default: 5
    :param beta: position weight within the feature vector. default: 0.5
    :param sigma_dev: sigma for image derivatives. default: 2.5
    :param rho: small constant. default: 0.5
    :param verbose: print or show inter-running debug results

    :return:
    """
    from skimage.segmentation import slic
    from skimage.segmentation import mark_boundaries

    inputs = {'numSegments': 200,
              'sigma': 5,
              'verbose': True,
              'beta': 0.5,
              'sigma_dev': 2.5,
              'rho': 0.5}

    m, n = image.shape[:2]

    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    x = np.arange(n)
    y = np.arange(m)

    # position maps
    xx, yy = np.meshgrid(x, y)

    inputs.update(kwargs)

    verbose = inputs['verbose']
    numSegments = inputs['numSegments']
    beta = inputs['beta']
    rho = inputs['rho']
    # 1. Superpixel clustering - apply SLIC and extract (approximately) the supplied number of segments
    segments = slic(image, n_segments=numSegments, sigma=inputs['sigma'])

    # show the output of SLIC
    # if verbose:
    # plt.imshow(mark_boundaries(image, segments))
    # plt.axis("off")
    # plt.show()

    # 2. Feature vector for each pixel
    img_x, img_y, img_xx, img_yy, img_xy = mt.computeImageDerivatives(image_gray, order=2,
                                                                      sigma=inputs['sigma_dev'])

    s = np.stack((image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2],
                  img_x, img_y, img_xx, img_yy,
                  beta * xx, beta * yy), axis=0)

    # 3. each superpixel is represented by the average feature vecotr of its pixels and the variance covariance
    #    matrix between its features
    labels = np.unique(segments)
    M = []
    u = []
    for label in labels:
        segment_pix = s[:, segments == label]
        M_n = np.cov(segment_pix) + 1e-7 * np.eye(9)

        if verbose:
            print(mt.is_pos_semidef(M_n))
        #    print lu.matrix_latex(M_n)
        M.append(M_n)
        u.append(segment_pix)
    # 4. Dissimilarity measure
    # solve real symmetric eigenvalus problem
    d = np.array([__dissimilarity(M1, M2, verbose=False) for M1 in M for M2 in M])

    # 5. The Laplacian
    W = np.exp(-d * rho).reshape((labels.shape[0], labels.shape[0]))
    H = np.diag(np.sum(W, axis=0))
    L = W - H

    if verbose:
        from skimage.color import label2rgb
        image_label_overlay = label2rgb(segments, image=image)
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(mark_boundaries(image, segments))
        axes[0, 1].imshow(image_label_overlay, interpolation='nearest')
        axes[1, 0].imshow(W, cmap='gray')
        plt.show()

    print('hello')


def distance_based(image, **kwargs):
    '''
    Computes saliency map according to distance methods

    :param image: the image to which the saliency map is computed
    :param feature: according to which feature vector the saliency is computed:

                    - 'pixel_val' - the value of the pixel itself
                    - 'LAB' - a feature vector using CIElab color space
                    - 'normals' - a feature vector using point's normals

    :param method: The method according to which the distance is computed:

                   - *'frequency'* - the distance between blurred and unblurred image :cite:`Achanta.etal2009`

                   .. code-block:: python

                       s1 = distance_based(img, filter_sigma = [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma],
                        feature = 'normals')

                   - *'local'* - the distance between regions :cite:`Achanta.etal2008`

                   .. code-block:: python

                       s2 = distance_based(img, filter_size = 5, method = 'local', feature = 'normals')

                   - *'context'* - distance between regions and position :cite:`Goferman.etal2012`

                       :param scales_number: the number of scales that should be computed. default 3.
                       :param kpatches: the number of minimum distance patches,  dtype = int. default: 64
                       :param c: a constant; default: 3 (paper implementation)

                   .. code-block:: python

                       s3 = distance_based(img, filter_size = 150, method = 'context', feature = 'pixel_val', verbose = False,
                        scales_number = 4)
                       s3[s3 < 1.e-5] = 0

    :param verbose: Print debugging prints. boolean. Default True
    :param dist_type: the distance measure:

                        - 'L1' ** SHOULD BE ADDED **
                        - 'Euclidean'
                        - 'Mahalonobis' - ** SHOULD BE ADDED **

    :param filter_sigma: (for global method) size of the filter(s)
    :param region_size: (for local and context aware methods) size of the region(s)/patch(es)

    :param normals: given normals computation for saliency based on normals image feature 

    :return: saliency map

    '''

    inputs = {'feature': 'pixel_val',
              'method': 'frequency',
              'dist_type': 'Euclidean',
              'filter_sigma': 0,
              'filter_size': 0,
              'scales_number': 3,
              'verbose': True}
    inputs.update(kwargs)

    verbose = inputs['verbose']
    img_feature = inputs['feature']
    sigma_flag = True
    s = None
    image = image.astype(np.float32)
    if inputs['filter_sigma'] != 0:
        filters = inputs['filter_sigma']
        sigma_flag = True  # whether sigma or size were sent
    elif inputs['filter_size'] != 0:
        filters = inputs['filter_size']
        sigma_flag = False
    else:
        raise RuntimeError('No filter size or sigma is given')

    # if the image feature is CIELAB, the image should be transformed to CIELab
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if img_feature == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # if the image feature is normals, either the normals are computed in advance and are given to the function or
    # they are computed here.
    if img_feature == 'normals':
        if 'normals' in kwargs:
            normals_image = kwargs['normals']
        else:
            img_x, img_y = mt.computeImageDerivatives(image, 1)
            normals_image = np.stack((img_x, img_y), axis=2)

    # -------- Frequency tuned saliency - Achanta et al 2009 ---------------
    if inputs['method'] == 'frequency' and img_feature != 'normals':
        s = __frequencyTuned(image, sigma_flag, filters)
    elif inputs['method'] == 'frequency' and img_feature == 'normals':
        s = __frequencyTuned(normals_image, sigma_flag, filters)

    # ---------Salient region detection - Achanta et al., 2008 ---------------
    elif inputs['method'] == 'local' and img_feature != 'normals':
        s = __regionConstrast(image, filters)
    elif inputs['method'] == 'local' and img_feature == 'normals':
        s = __regionConstrast(normals_image, filters)

    elif inputs['method'] == 'context':
        scales_number = kwargs.get('scales_number', 3)
        context_input = {'kaptches': 64,
                         'thresh': 0.2,
                         'c': 3}
        context_input.update(kwargs)

        ksizes = [filters * 2 ** (-n) for n in np.arange(scales_number)]
        s = [__contextAware(image, ksize, img_feature,
                            kpatches=context_input['kaptches'],
                            thresh=context_input['thresh'],
                            c=context_input['c'],
                            verbose=verbose)
             for ksize in ksizes]
        saliency_map = np.array(s)

    if img_feature == 'LAB' and inputs['method'] != 'context':
        saliency_map = np.linalg.norm(s, axis=3)
    elif img_feature == 'normals' and inputs['method'] != 'context':
        saliency_map = np.linalg.norm(s, axis=3)
    elif img_feature == 'pixel_val' and inputs['method'] != 'context':
        saliency_map = np.abs(s)

    return np.mean(saliency_map, axis=0)


def __frequencyTuned(image, sigma_flag, filters):
    """
    Saliency computed according to :cite:`Achanta.etal2009`

    :param image: image on which the saliency is computed
    :param sigma_flag: whether the sigma or the kernel size are used for blurring
    :param image_feature: the feature according to which the saliency is computed.

    :return: saliency map of the given image
    """
    if sigma_flag:
        blurred_images = [mt.DoG_filter(image, sigma1=filter1, sigma2=filter2) for filter1, filter2 in
                          zip(filters[:-1], filters[1:])]
    else:
        blurred_images = [mt.DoG_filter(image, ksize1=filter1, ksize2=filter2) for filter1, filter2 in
                          zip(filters[:-1], filters[1:])]

    image_blurred = [image - blurred for blurred in blurred_images]
    return np.array(image_blurred)


def __regionConstrast(image, region1_size):
    """
    Saliency computed according to :cite:`Achanta.etal2008`

    :param image: image on which the saliency is computed
    :param region1_size: R1 size -- does not change throughout
    :param image_feature: the feature according to which the saliency is computed.

    :return: saliency map of the given image
    """

    # 1. Creating the kernels
    k_R1 = np.ones((region1_size, region1_size)) * 1 / region1_size ** 2

    row_size, column_size = image.shape[:2]
    k_R2 = [np.ones((ksize, ksize)) / ksize ** 2 for ksize in [row_size / 2, row_size / 4, row_size / 8]]

    # 2. Create convoluted map according to region1
    map_R1 = cv2.filter2D(image, -1, k_R1)
    maps_R2 = [cv2.filter2D(image, -1, ksize) for ksize in k_R2]

    return np.array([map_R1 - map_R2 for map_R2 in maps_R2])


def __contextAware(image, ksize, image_feature, **kwargs):
    """
    Saliency computed according to :cite:`Goferman.etal2012`

    only one scale.

    :param image: image on which the saliency is computed
    :param ksizes: the sizes of the patches
    :param image_feature: the feature according to which the saliency is computed.
    :param kpatches: the number of minimum distance patches,  dtype = int
    :param c: a constant; c=3 in the paper's implementation
    :param verbose: print debugging prints

    :return: saliency map of the given image

    """
    K = kwargs.get('kpatches', 64)
    c = kwargs.get('c', 3.)
    verbose = kwargs.get('verbose', True)

    if type(K) != int:
        warnings.warn('K should be interger, using K=64 instead.', RuntimeWarning)
        K = 64

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
            if k % n == 0:
                if verbose:
                    print(k)
            k = k.astype('int')
            i_current = np.int(k / n)
            j_current = np.int(k % n)
            dcolor, i, j = __dcolor(averaged_image[i_current, j_current], averaged_image, image_feature, kpatches=K,
                                    verbose=False)

            d_pos = np.sqrt((i_current - i) ** 2 + (j_current - j) ** 2)
            d = dcolor / (1 + c * d_pos)
            s = 1 - np.exp(-np.mean(d))

            # remove indices of patches that were already found as similar and set their saliency value to the
            # same as the one that was already found
            saliency[i, j] = s
            saliency[i_current, j_current] = s

    return saliency


def __dcolor(p_i, image, image_feature, **kwargs):
    '''
    computes the most similar patch to a patch i and their indices

    :param p_i: the patch to which the comparison is made
    :param vector: all other patches
    :param image_feature: the feature according to which the dcolor is computed
    :param kpatches: the number of minimum distance patches,  dtype = int

    :return: a vector of K most similar dcolors; a vector of K most similar indices

    '''

    K = kwargs.get('kpatches', 64)
    thresh = kwargs.get('thresh', 50)
    verbose = kwargs.get('verbose', True)
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

    K_closest = np.argsort(dcolors, axis=None)[:K]
    i = K_closest / n
    j = K_closest % n

    if verbose:
        print(dcolors[i, j])
    return dcolors[i, j], i, j


def __dissimilarity(A, B, **kwargs):
    """
    Computes the dissimilarity measure for two covariance matrices according to

    .. math::

        d(A,B) = \sqrt{\sum{ln^2\lambda{A,B}}}

    :param A, B: covariance matrices between which the dissimilarity is computed
    :param verbose: print intre running

    :return: dissimilarity measure between A and B

    """
    verbose = kwargs.get('verbose', False)
    if verbose:
        print(lu.matrix_latex(A))
        print(lu.matrix_latex(B))

    #  solve e.values for: A x = lambda B x
    eigs = LA.eigvals(A, B)
    eigvals = np.real(eigs)
    eigvals[np.isnan(eigvals)] = 0.

    d = np.sqrt(np.sum(np.log(eigvals) ** 2))
    return d


if __name__ == '__main__':
    # --- initializations
    img = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/ownCloud/Data/Images/mult1.png'), cv2.COLOR_BGR2RGB)
    sigma = 2.5

    print('-' * 20, 'frequency tuned', '-' * 20)
    s1 = distance_based(img, filter_sigma=[sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma],
                        feature='normals')

    print('-' * 20, 'local method', '-' * 20)
    s2 = distance_based(img, filter_size=5, method='local', feature='normals')
    print('-' * 20, 'context aware', '-' * 20)
    s3 = distance_based(img, filter_size=150, method='context', feature='pixel_val', verbose=False,
                        scales_number=4)
    s3[s3 < 1.e-5] = 0

    #    bayesian(img)
    fig = plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.imshow(img), ax.set_title('original')

    ax = plt.subplot(2, 2, 2)
    ax.imshow(s1, cmap='gray'), ax.set_title('normals, frequency tuned')

    ax = plt.subplot(2, 2, 3)
    ax.imshow(s2, cmap='gray'), ax.set_title('normals, salient region')
    ax = plt.subplot(2, 2, 4)
    ax.imshow(s3, cmap='gray'), ax.set_title('pixel value, context aware')

    plt.show()
