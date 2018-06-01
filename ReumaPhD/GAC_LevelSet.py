'''
infraGit
photo-lab-3\Reuma
27, Feb, 2017 
'''

import platform


if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from . import Saliency as sl
from matplotlib import pyplot as plt
from . import MyTools as mt
import numpy as np
import cv2
from skimage import measure


def chooseLargestContours(contours, labelProp, minArea):
    '''
    leaves only contours with area larger than minArea
    :param contours: list of contours
    :param labelProp: properties of labeled area
    :param minArea: minimal area needed
    :return: list of "large" contours
    '''
    contours_filtered = []

    for prop, c in zip(labelProp, contours):
        if prop.area >= minArea and prop.area <= 10e6:
            contours_filtered.append(c)

    return contours_filtered


def drawContours(phi, img, ax, **kwargs):
    """
    Draws the contours of a specific iteration
    :param phi: the potential function
    :param img: the image on which the contours will be drawn
    :return: the figure
    """

    ax.cla()
    ax.axis("off")
    ax.set_ylim([img.shape[0], 0])
    ax.set_xlim([0, img.shape[1]])

    phi_binary = phi.copy()

    # segmenting and presenting the found areas
    phi_binary[np.where(phi > 0)] = 0
    phi_binary[np.where(phi < 0)] = 1
    phi_binary = np.uint8(phi_binary)

    contours = measure.find_contours(phi, 0.)
    blob_labels = measure.label(phi_binary, background = 0)
    label_props = measure.regionprops(blob_labels)

  #  contours = chooseLargestContours(contours, label_props, 1)

    mt.imshow(img)

    for c in contours:
        c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
        ax.plot(c[:, 0], c[:, 1], '-r')

    return ax

def compute_vt(vectorField, edge_map, kappa, **kwargs):
    """
    Computes the vector field derivative in each direction according to
    vt = g(|\nabla f|)\nabla^2 * v - h(|\nabla f|)*(v - \nabla f)

    :param vectorField: usually created based on an edge map; nxmx2 (for x and y directions)
    :param edge_map
    :param kappa: curvature map

    ---optionals: ----
    :param eps: an epsilon for division
    gradient computation parameters
    :param gradientType
    :param ksize
    :param sigma

    :return: vt, nxmx2 (for x and y directions)
    """
    eps = kwargs.get('eps', 1e-5)
    # compute the derivatives of the edge map at each direction
    nabla_edge = mt.computeImageGradient(edge_map, **kwargs)

    # compute the laplacian of the vector field
    laplacian_vx = cv2.Laplacian(vectorField[:,:,0], cv2.CV_64F)
    laplacian_vy = cv2.Laplacian(vectorField[:,:,1], cv2.CV_64F)

    #compute the functions that are part of the vt computation
    g = np.exp(-nabla_edge/(kappa+eps)**2)
    h = 1 - g

    vx_t = g * laplacian_vx - h * (vectorField[:,:,0] - edge_map)
    vy_t = g * laplacian_vy - h * (vectorField[:, :, 1] - edge_map)

    return np.stack((vx_t, vy_t), axis = 2)


def compute_vb(img, phi, **kwargs):
    """
    Computes the band velocity, according to Li et al., 2006.
    :param img: the image upon which the contour is searched
    :param phi: the level set function
    :param band_width: the width of the contour. default: 5
    :param threshold: for when the velocity equals zero. default: 0.5
    :param stepsize. default 0.05

    :return: vb
    """
    band_width = kwargs.get('band_width', 5.)
    threshold = kwargs.get('threshold', 0.5)
    tau = kwargs.get('stepsize', 0.05)

    # Define the areas for R and R'
    R = (phi > 0) * (phi <= band_width)
    R_ = (phi >= -band_width) * (phi < 0)

    SR = np.zeros(img.shape)
    SR_ = np.zeros(img.shape)

    SR[R] = np.mean(img[R])
    SR_[R_] = np.mean(img[R_])
    vb = 1 - (SR - SR_) / (np.linalg.norm(SR + SR_))
    vb[vb < threshold] = 0
    vb *= tau
    vb[np.where(phi != 0)] = 0
    return vb

def geometricActiveContours(img, phi, **kwargs):
    """
    :param img: the image upon which the active contours is implemented.
    :param phi: the initial level-set function

    --Optionals--
    :param edges: an edge function according to which the GVF element is computed
    :param alpha: internal force; can be either constant, weights or function (such as edge function
                  g = 1/(1+|\nabla G(I) * I|).
                  default: 'edgeFunction'
    :param GAC_inputs - dictionary with:
            eps, stepsize - initializations for the GAC (default: 1. for all)

    Function g :
    :param g_ksize: kernel size for g derivatives, default: 3.

    Image gradient computation dictionary
    :param img_grad - dictionary with:
            'type': 'L1', 'L2' or 'LoG', default: L2.
            The kernel size can be decided either by sigma or by ksize (or both). The default is by sigma 2.5
            'sigma': sigma to be used, default: 2.5.
            'ksize': kernel size to be used, default: 0.

    Blurring dictionary
    :param blur - dictionary with:
            'ksize': kernel size for blurring, default: (0,0).
            'sigma': sigma size for blurring, default: 2.5

    region dictionary
    :param region - dictionary with:
            'region' - a map between [-1,1] where the most interesting areas are 1 and the least are -1
            'weights' - a weight for the region element; how much it influences the development

    Second Level set function - for open contours
    :param open_contour - Whether the contour searched is open or not. Boolean (default: false)
    :param psi - the function with which the first level-set function intersects with in order to define the end-points
            of the contour.

    :return:
    """

    # -------Initializations--------
    edgeMap = kwargs.get('edges', None)
    open_contour = kwargs.get('open_contour', False)
    GAC_inputs = {'alpha': None, 'stepsize': 1.}

    if open_contour:
        psi = kwargs['psi']

    gradient_inputs = {'gradientType': 'L2',
                       'sigma': 2.5,
                       'ksize': 0}
    g_ksize = kwargs.get('g_ksize', 3)

    blurring_inputs = {'ksize': (0,0),
                       'sigma': 2.5}

    region_inputs = {'region': 0,
                     'weight': 1}
    # updating inputs:
    if 'GAC_inputs' in list(kwargs.keys()):
        GAC_inputs.update(kwargs['GAC_inputs'])

    if 'img_grad' in list(kwargs.keys()):
        gradient_inputs.update(kwargs['img_grad'])

    if 'blur' in list(kwargs.keys()):
        blurring_inputs.update(kwargs['blur'])

    if 'region' in list(kwargs.keys()):
        region_inputs.update(kwargs['region'])

    alpha = GAC_inputs['alpha']
    eps = GAC_inputs['epsilon']
    stepsize = GAC_inputs['stepsize']

    blur_ksize = blurring_inputs['ksize']
    blur_sigma = blurring_inputs['sigma']
    g_x = g_y = 0

    if alpha is not None:
        g_x, g_y = mt.computeImageDerivatives(alpha, 1, ksize = g_ksize)
        alpha = cv2.GaussianBlur(g, blur_ksize, blur_sigma)
    else:
        alpha = 0

    if edgeMap is not None:
        v_x, v_y = mt.computeImageDerivatives(edgeMap, 1, ksize = g_ksize)
        v = np.stack((v_x, v_y), axis = 2)
    else:
        v = np.zeros((img.shape[0], img.shape[1], 2))
        edgeMap = np.zeros((img.shape[0], img.shape[1]))

    fig, ax = plt.subplots()

    # FFMpegWriter = manimation.writers['ffmpeg']
    # metadata = dict(title = 'Movie Test', artist = 'Matplotlib',
    #                 comment = 'Movie support!')
    # writer = FFMpegWriter(fps = 25, metadata = metadata)

    #with writer.saving(fig, 'mult1_double1.mp4', 300):
    ax2 = plt.figure(2)
    for i in range(0,200):

        # phi derivatives
        phi_x, phi_y, phi_xx, phi_yy, phi_xy = mt.computeImageDerivatives(phi, 2, ksize = gradient_inputs['ksize'],
                                                                          sigma = gradient_inputs['sigma'])
        # |\nabla \phi|
        norm_nabla_phi = np.sqrt(phi_x ** 2 + phi_y ** 2 + eps)

        if open_contour:
            nabla_psi = mt.computeImageGradient(psi, ksize = gradient_inputs['ksize'], gradientType = 'L1')

        # level set curvature: kappa=div(\nabla \phi / |\nabla \phi|)
        kappa = ((phi_xx * phi_y ** 2 + phi_yy * phi_x ** 2 - 2 * phi_xy * phi_x * phi_y) / norm_nabla_phi**3)

        # -----external elements--------
        # geodesic flow
        Fext_nabla_phi = g_x * phi_x + g_y * phi_y

        # gvf
        vt = compute_vt(v, edgeMap, kappa, eps = eps, **gradient_inputs)
        v += vt
        Fext_gvf = v[:,:,0] * phi_x + v[:,:,1] * phi_y

        # total external forces
        Fext = Fext_gvf - Fext_nabla_phi
        #-------------

        # band velocity
        if open_contour:
            vb = compute_vb(img, phi, stepsize = stepsize)
        else:
            vb = np.zeros(img)

        v_phi = (alpha * kappa + region_inputs['weight'] * region_inputs['region'] + 5 * vb)
        phi_t = v_phi * norm_nabla_phi - Fext

        phi -= cv2.GaussianBlur(phi_t, blur_ksize, blur_sigma) * stepsize
        if open_contour:
            psi_t = v_phi * nabla_psi
            psi -= cv2.GaussianBlur(psi_t, blur_ksize, blur_sigma) * stepsize
            psi_copy = psi.copy()
            psi_copy[np.where(psi > 0)] = 1
            psi_copy[np.where(psi < 0)] = -1
            plt.figure(2), plt.imshow(psi)
            plt.figure(3)
            #  psi_copy= np.uint8(psi_copy)
            ax = drawContours(phi * psi_copy, img, ax)
        else:
            ax = drawContours(phi, img, ax)
   #     writer.grab_frame()
        plt.pause(.5e-10)
    plt.show()

def geodesicActiveContours(img, phi, **kwargs):
    """
    Geodesic active contours general implementation according to Casselles et al., 1997

    :param img: the image upon which the active contours is implemented.
    :param phi: the initial level-set function

    --Optionals--
    :param GAC_inputs:
            epsilon, beta, stepsize - initializations for the GAC (default: 1. for all)

    Function g :
    :param g_ksize: kernel size for g derivatives, default: 3.

    Image gradient computation dictionary
    :param img_grad - dictionary with:
            'type': 'L1', 'L2' or 'LoG', default: L2.
            'sigma': sigma to be used, default: 7.5.
            'ksize': kernel size to be used, default: 5.

    Blurring dictionary
    :param blur - dictionary with:
            'ksize': kernel size for blurring, default: (0,0).
            'sigma': sigma size for blurring, default: 2.5

    :return:
    """

    # -------Initializations--------
    GAC_inputs = {'alpha':1.,
                  'beta': 1.,
                  'epsilon': np.finfo(float).eps,
                  'stepsize': 1.}

    gradient_inputs = {'gradientType': 'L2',
                       'sigma': 7.5,
                       'ksize:': 5}
    g_ksize = kwargs.get('g_ksize', 3)

    blurring_inputs = {'ksize': (0,0),
                       'sigma': 2.5}
    # updating inputs:
    if 'GAC_inputs' in list(kwargs.keys()):
        GAC_inputs.update(kwargs['GAC_inputs'])

    eps = GAC_inputs['epsilon']
    beta = GAC_inputs['beta']
    stepsize = GAC_inputs['stepsize']

    if 'img_grad' in list(kwargs.keys()):
        gradient_inputs.update(kwargs['img_grad'])

    if 'blur' in list(kwargs.keys()):
        blurring_inputs.update(kwargs['blur'])

    blur_ksize = blurring_inputs['ksize']
    blur_sigma = blurring_inputs['sigma']

    phi_binary = phi.copy()

    # ---- define g(I) ---
    imgGradient = mt.computeImageGradient(img, **gradient_inputs)
    g = 1 / (1 + imgGradient)

    g_x, g_y = mt.computeImageDerivatives(g, 1, ksize = g_ksize)
    g = cv2.GaussianBlur(g, blur_ksize, blur_sigma)

    for i in range(0,1500):

        # computing phi derivatives
        phi_x, phi_y, phi_xx, phi_yy, phi_xy = mt.computeImageDerivatives(phi, 2)

        # div(\nabla phi/|\nabla phi|)
        div_phi = beta * (
        (phi_xx * phi_y ** 2 + phi_yy * phi_x ** 2 - 2 * phi_xy * phi_x * phi_y) / (phi_x ** 2 + phi_y ** 2 + eps) ** 2)

        # dot product of \nabla g and \nabla phi
        grad_g_dot_grad_phi = g_x * phi_x + g_y * phi_y

        phi_t = g * div_phi + grad_g_dot_grad_phi
        phi += cv2.GaussianBlur(phi_t, blur_ksize, blur_sigma) * stepsize

        # segmenting and presenting the found areas
        phi_binary[np.where(phi > 0)] = 0
        phi_binary[np.where(phi < 0)] = 1
        phi_binary = np.uint8(phi_binary)

        contours = measure.find_contours(phi, 0.)
        blob_labels = measure.label(phi_binary, background=0)
        label_props = measure.regionprops(blob_labels)

        contours = chooseLargestContours(contours, label_props, 10)

        plt.cla()
        plt.axis("off")
        plt.ylim([img.shape[0], 0])
        plt.xlim([0, img.shape[1]])

        mt.imshow(img)

        for c in contours:
            c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
            plt.plot(c[:, 0], c[:, 1], '-r')

        plt.pause(.5e-10)

    plt.show()

if __name__ == '__main__':
    # --- initializations
  #  img = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/ownCloud/Data/Images/doubleTrouble.png'), cv2.COLOR_BGR2GRAY)


    img = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\DoubleTrouble.png'), cv2.COLOR_BGR2RGB)
    img_ = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    sigma = 2.5

    # define region constraints
    region = sl.distance_based(img, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='normals')
    region1 = sl.distance_based(img, filter_size=2.5, feature='normals', method='local')

    region = cv2.GaussianBlur(region, ksize = (5, 5), sigmaX = sigma)
    region1 = cv2.GaussianBlur(np.uint8(region), ksize = (5,5), sigmaX = sigma)

    # define edge function
    edges = cv2.Canny(region1, 30,70)
    kernel = np.ones((3, 3), np.uint8)
 #   edges = cv2.dilate(edges, kernel, iterations = 1)
   # region+=edges

    mt.imshow(edges)
    region = cv2.normalize(region.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized
    # floating point
    plt.figure()
    mt.imshow(region)

    # define initial alpha function (usually a gradient map)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGradient = mt.computeImageGradient(img, gradientType = 'L2', ksize = 5)
    g = 1 / (1 + imgGradient)

    # define an initial contour via phi(x,y) = 0
    phi = np.ones(img.shape[:2])
    img_height, img_width = img.shape[:2]
    width, height = 20, 20
    phi[img_height / 2 - height: img_height / 2 + height, img_width / 2 - width: img_width / 2 + width] = -1
    # phi[img_height - 2 * height: img_height - height, :] = -1

    # define an initial end points via psi(x,y)
    psi = np.ones(img.shape[:2])
    zero_pos = 50  # at +- 250 pixels
    psi[:, zero_pos: zero_pos + width] = 0
    psi[:, -(zero_pos + width):-zero_pos] = 0
    psi[:, :zero_pos] = -1
    psi[:, -zero_pos:] = -1


    # function inputs
    gradient_inputs = {'gradientType': 'L2', 'sigma': 2.5}
    gac_inputs = {'alpha': g,
                  'stepsize': .1,
                  'epsilon': 1e-6}

    blur_inputs = {'sigma': 2.5, 'ksize': (0,0)}
    region_inputs = {'region': region, 'weight': 1}

    geodesicActiveContours(img, phi, GAC_inputs = gac_inputs)

    # geometricActiveContours(img, phi, GAC_inputs = gac_inputs, img_grad = gradient_inputs, blur=blur_inputs,
    #                         region = region_inputs, open_contour = True, psi = psi)
