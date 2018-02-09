'''
infragit
reuma\Reuma
07, Jan, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
import MyTools as mt
import Saliency as sl
from RasterData import RasterData
import cv2
from skimage import measure
from matplotlib import pyplot as plt

EPS = np.finfo(float).eps

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


class LevelSetFactory:
    phi = None  # level set function (nd-array)
    img = None  # the analyzed image (of the data)
    region = None  # region constraint

    step = 0  # step size

    imgGradient = None  # gradient of the analyzed image
    phi_x = phi_y = phi_xx = phi_xy = phi_yy = None  # level set function derivatives
    norm_nabla_phi = None  # |\nabla \phi| - level set gradient size
    kappa = None  # level set curvature
    g = None  # internal force
    f = None  # external force (for GVF)
    psi = None  # internal force, for open contours
    norm_nabla_psi = None  # |\nabla \phi| - open contour level set gradient size

    def __init__(self, img, **kwargs):
        """
        initialize factory.

        :param img: the image upon which the level set is started.
        :param kwargs:
            step - time step for advancing the level set function.
        """
        step = kwargs.get('step', 0.05)
        self.step = step
        if isinstance(img, RasterData):
            self.img = img.data

        else:
            self.img = img

    def drawContours(self, ax, **kwargs):
        """
        Draws the contours of a specific iteration
        :param phi: the potential function
        :param img: the image on which the contours will be drawn
        :return: the figure
        """

        ax.cla()
        ax.axis("off")
        ax.set_ylim([self.img.shape[0], 0])
        ax.set_xlim([0, self.img.shape[1]])

        phi_binary = self.phi.copy()

        # segmenting and presenting the found areas
        phi_binary[np.where(self.phi > 0)] = 0
        phi_binary[np.where(self.phi < 0)] = 1
        phi_binary = np.uint8(phi_binary)

        contours = measure.find_contours(self.phi, 0.)
        blob_labels = measure.label(phi_binary, background = 0)
        label_props = measure.regionprops(blob_labels)

        #  contours = chooseLargestContours(contours, label_props, 1)

        mt.imshow(self.img)

        for c in contours:
            c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
            ax.plot(c[:, 0], c[:, 1], '-r')

        return ax

    def flow(self, type, **kwargs):
        """
        Returns the flow of the level set according to the type wanted
        :param type: can be one of the following:
            'constant': Ct = N ==> phi_t = |\nabla \varphi|
            'curvature': Ct = kN ==> phi_t = div(\nabla \varphi / |\nabla \varphi|)|\nabla \varphi|
            'equi-affine': Ct = k^(1/3) N ==> phi_t = (div(\nabla \varphi / |\nabla \varphi|))^(1/3)*|\nabla \varphi|
            'geodesic': geodesic active contours, according to Casselles et al., 1997
                        Ct = (g(I)k -\nabla(g(I))N)N ==>
                        phi_t = [g(I)*div(\nabla \varphi / |\nabla \varphi|))^(1/3))*|\nabla \varphi|
            'band': band velocity, according to Li et al., 2006.


        ------- optionals ---------
        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param sigma: sigma for LoG gradient
        :param ksize: kernel size, for blurring and derivatives

        ---- band velocity optionals ----
        :param band_width: the width of the contour. default: 5
        :param threshold: for when the velocity equals zero. default: 0.5
        :param stepsize. default 0.05

        :return: phi_t
        """
        ksize = kwargs.get('ksize', 5)
        gradientType = kwargs.get('gradientType', 'L1')
        sigma = kwargs.get('sigma', 2.5)

        if type == 'constant':
            return np.abs(self.norm_nabla_phi)

        if type == 'curvature':
            return self.kappa * self.norm_nabla_phi

        if type == 'equi-affine':
            return np.cbrt(self.kappa) * self.norm_nabla_phi

        if type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curavature flow !!
            g_x, g_y = mt.computeImageDerivatives(self.g, 1, ksize = ksize)
            return cv2.GaussianBlur(self.g * self.kappa * self.norm_nabla_phi + (g_x * self.phi_x + g_y * self.phi_y),
                                    (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

        if type == 'open':
            psi_x, psi_y = mt.computeImageDerivatives(self.psi, 1, ksize = ksize)
            self.norm_nabla_psi = np.sqrt(psi_x ** 2 + psi_y ** 2)
            g_x, g_y = mt.computeImageDerivatives(self.g, 1, ksize = ksize)
            return cv2.GaussianBlur(self.g * self.kappa * self.norm_nabla_psi + (g_x * psi_x + g_y * psi_y),
                                    (processing_props['ksize'], processing_props['ksize']),
                                    processing_props['sigma'])

        if type == 'band':
            return self.__compute_vb(**kwargs)

    def __ls_derivatives_curvature(self, processing_props, **kwargs):
        """
        computes and updates the level set function derivatives and curvature
        :param processing_props: gradient type, sigma and ksize for derivatives and gradient computations

        """

        self.norm_nabla_phi = mt.computeImageGradient(self.phi, gradientType = processing_props['gradientType'])
        self.phi_x, self.phi_y, self.phi_xx, self.phi_yy, self.phi_xy = \
            mt.computeImageDerivatives(self.phi, 2, ksize = processing_props['ksize'],
                                       sigma = processing_props['sigma'])

        self.kappa = cv2.GaussianBlur((self.phi_xx * self.phi_y ** 2 +
                                       self.phi_yy * self.phi_x ** 2 -
                                       2 * self.phi_xy * self.phi_x * self.phi_y) /
                                      (self.norm_nabla_phi + EPS) ** 2,
                                      (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

        if np.any(np.abs(self.kappa) < 1e-5):
            self.kappa[np.abs(self.kappa) < 1e-5] = 0

    def __drawContours(self, function, ax, **kwargs):
        """
        Draws the contours of a specific iteration
        :param phi: the potential function
        :param img: the image on which the contours will be drawn
        :return: the figure
        """

        ax.cla()
        ax.axis("off")
        ax.set_ylim([self.img.shape[0], 0])
        ax.set_xlim([0, self.img.shape[1]])

        color = kwargs.get('color', 'b')
        img = kwargs.get('image', self.img)

        function_binary = function.copy()

        # segmenting and presenting the found areas
        function_binary[np.where(function > 0)] = 0
        function_binary[np.where(function < 0)] = 1
        function_binary = np.uint8(function_binary)

        contours = measure.find_contours(function, 0.)
        blob_labels = measure.label(function_binary, background = 0)
        label_props = measure.regionprops(blob_labels)

        #     contours = chooseLargestContours(contours, label_props, 1)

        mt.imshow(img)
        l_curve = []
        for c in contours:
            c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
            curve, = ax.plot(c[:, 0], c[:, 1], '-', color = color)
            l_curve.append(curve)

        return l_curve, ax
    def __compute_vb(self, **kwargs):
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
        R = (self.phi > 0) * (self.phi <= band_width)
        R_ = (self.phi >= -band_width) * (self.phi < 0)

        SR = np.zeros(self.img.shape)
        SR_ = np.zeros(self.img.shape)

        SR[R] = np.mean(self.img[R])
        SR_[R_] = np.mean(self.img[R_])
        vb = 1 - (SR - SR_) / (np.linalg.norm(SR + SR_))
        vb[vb < threshold] = 0
        vb *= tau
        vb[np.where(self.phi != 0)] = 0
        return vb
    def __compute_vt(self, vectorField, **kwargs):
        """
        Computes the vector field derivative in each direction according to
        vt = g(|\nabla f|)\nabla^2 * v - h(|\nabla f|)*(v - \nabla f)

        :param vectorField: usually created based on an edge map; nxmx2 (for x and y directions)
        :param edge_map
        :param kappa: curvature map

        ---optionals: ----
        gradient computation parameters
        :param gradientType
        :param ksize
        :param sigma

        :return: vt, nxmx2 (for x and y directions)
        """

        # compute the derivatives of the edge map at each direction
        nabla_edge = mt.computeImageGradient(self.f, **kwargs)

        # compute the laplacian of the vector field
        laplacian_vx = cv2.Laplacian(vectorField[:, :, 0], cv2.CV_64F)
        laplacian_vy = cv2.Laplacian(vectorField[:, :, 1], cv2.CV_64F)

        # compute the functions that are part of the vt computation
        g = np.exp(-nabla_edge / (self.kappa + EPS) ** 2)
        h = 1 - g

        vx_t = g * laplacian_vx - h * (vectorField[:, :, 0] - self.f)
        vy_t = g * laplacian_vy - h * (vectorField[:, :, 1] - self.f)

        #    plt.quiver(vx_t, vy_t)

        return np.stack((vx_t, vy_t), axis = 2)

    def __compute_vo(self, **kwargs):
        """
        Computes velocity under orhogonality constraint (the force is computed so that the contours will be orthogonal
        to another function (psi)
        :param kwargs:
        :return:
        """
        psi = kwargs.get('function', self.psi)  # if no other function is given it will be moving according to psi
        processing_props = kwargs.get('processing_props', {'sigma': 2.5, 'ksize': 3, 'gradientType': 'L1'})

        psi_x, psi_y = mt.computeImageDerivatives(psi, 1, **processing_props)
        self.norm_nabla_psi = np.sqrt(psi_x ** 2 + psi_y ** 2)
        grad_psi_grad_phi = psi_x * self.phi_x + psi_y * self.phi_y

        return grad_psi_grad_phi / (self.norm_nabla_phi + EPS)

    def moveLS(self, **kwargs):
        """
        The function that moves the level set until the desired contour is reached

        :param flow_type - flags for flow types (binary):
        'constant', 'curvature', 'equi-affine', 'geodesic'

        :param gvf_flag: flag to add gradient vector flow
        :param open_flag: flag for open contours

        :return the contours after level set
         """
        # ------inputs

        flow_types = kwargs.get('flow_types', ['geodesic'])
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5}
        open_flag = kwargs.get('open_flag', False)

        gvf_w = kwargs.get('gvf_w', 1.)
        vo_w = kwargs.get('vo_w', 1.)
        region_w = kwargs.get('region_w', 1.)

        if 'processing_props' in kwargs.keys():
            processing_props.update(kwargs['processing_props'])

        fig, ax = plt.subplots(num = 1)
        ax2 = plt.figure("phi")
        fig3, ax3 = plt.subplots(num = 3)

        plt.imshow(self.img)
        for i in range(200):
            print i
            if i > 26:
                print 'hello'
            intrinsic = np.zeros(self.img.shape[:2])
            extrinsic = np.zeros(self.img.shape[:2])
            self.__ls_derivatives_curvature(processing_props)

            # ---------- intrinsic movement ----------
            # regular flows
            for item in flow_types:
                intrinsic += self.flow(item, **processing_props)
                if np.any(intrinsic > 20):
                    print i

            # region force
            intrinsic += region_w * self.region * self.norm_nabla_phi

            # open contour
            if open_flag:
                psi_t = self.flow('open', **processing_props)
                # psi_t += self.region_weight * self.region * self.norm_nabla_psi
                self.psi += cv2.GaussianBlur(psi_t, (processing_props['ksize'], processing_props['ksize']),
                                             processing_props['sigma'])
                plt.figure(1)
                l_curve, ax3 = self.__drawContours(self.psi, ax, color = 'b')

            # ---------------extrinsic movement ----------
            f_x, f_y = mt.computeImageDerivatives(self.f, 1, ksize = processing_props['ksize'],
                                                  sigma = processing_props['sigma'])
            v = np.stack((f_x, f_y), axis = 2)
            vt = self.__compute_vt(v, **processing_props)
            v += vt
            extrinsic = (v[:, :, 0] * self.phi_x + v[:, :, 1] * self.phi_y) * gvf_w

            # for constrained contours
            extrinsic += self.__compute_vo() * vo_w
            #  self.psi += extrinsic
            plt.figure(3)
            plt.imshow(self.kappa)
            plt.pause(.5e-10)

            phi_t = self.step * (intrinsic - extrinsic)
            self.phi += cv2.GaussianBlur(phi_t, (processing_props['ksize'], processing_props['ksize']),
                                         processing_props['sigma'])

            plt.figure('phi')
            plt.imshow(self.phi)

            if open_flag:
                for curve in l_curve:
                    self.phi[curve._y.astype('int'), curve._x.astype('int')] = 0

            plt.figure(1)
            _, ax = self.__drawContours(self.phi, ax, color = 'r')
            plt.pause(.5e-10)
        plt.show()
if __name__ == '__main__':
    # initial input:
    img_orig = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\channel91.png'), cv2.COLOR_BGR2RGB)
    img_normed = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point
    sigma = 2.5  # blurring

    ls_obj = LevelSetFactory(img_normed, step = 0.05)

    processing_props = {'sigma': 5, 'ksize': 3, 'gradientType': 'L1'}


    # ------- Initial contour via phi(x,y) = 0 ---------------------------
    phi = np.ones(img_orig.shape[:2])
    psi = np.ones(img_orig.shape[:2])
    img_height, img_width = img_orig.shape[:2]
    width, height = 5,20

    # option 1: rectangle:
    # phi[img_height / 2 - height: img_height / 2 + height, img_width / 2 - width: img_width / 2 + width] = -1

    # option 2: horizontal line
    #phi[img_height - 2 * height: img_height - height, :] = -1

    # option 3: vertical line
    phi[:, img_width - 2 * width: img_width - width] = -1
    ls_obj.phi = phi

    # ------- Initial limits via psi(x,y) = 0 ---------------------------
    # option 1: horizontal line
    # psi[img_height - 2 * height: img_height - height, :] = -1
    # option 2: vertical line
    width_boundary = 20
    psi[:, 0: width_boundary] = -1
    psi[:, -width_boundary:] = -1
    ls_obj.psi = psi
    # ---------------------------------------------------------------------

    # ---------------- Intrinsic forces maps ------------------------------

    # Force I - function g:
    # can be either constant, weights, or function
    # option 1: edge map g = 1/(1+|\nabla G(I) * I|).
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.normalize(img_gray.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    imgGradient = mt.computeImageGradient(img_gray, gradientType = 'L2', sigma = 1.5)
    g = 1 / (1 + imgGradient)

    # option 2: saliency map
    #    g = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='normals')

    ls_obj.g = g

    # Force II - region constraint:
    region = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma],
                               feature = 'normals')
    region = cv2.GaussianBlur(region, ksize = (3, 3), sigmaX = sigma)
    region = cv2.normalize(region.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)

    ls_obj.region = region

    # Force III - open contours:

    # ---------------------------------------------------------------------

    # ----------------Extrinsic forces maps (vector field)------------------

    # The map which the GVF will be defined by
    # option 1: the image itself
    f = cv2.GaussianBlur(img_gray, ksize = (3, 3), sigmaX = sigma)

    # # option 2: edge map
    # f = cv2.Canny(img_orig, 10, 50)
    #  f = cv2.GaussianBlur(f, ksize = (5, 5), sigmaX = 1.5)

    # plt.imshow(f)
    # plt.show()

    #
    # # option 3: saliency map
    # f = cv2.GaussianBlur(region, ksize = (5, 5), sigmaX = sigma)

    #  f = np.zeros(img_gray.shape)
    ls_obj.f = f

    ls_obj.moveLS(flow_types = [], open_flag = False, processing_props = processing_props,
                  gvf_w = 1.,
                  vo_w = 1.,
                  region_w = 0.)
