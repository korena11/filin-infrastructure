'''
infragit
reuma\Reuma
07, Jan, 2018 
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

import MyTools as mt
from LevelSets import Saliency as sl
from RasterData import RasterData
from .LevelSetFunction import LevelSetFunction

EPS = np.finfo(float).eps


class LevelSetFactory:
    # General initializations
    processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5, 'regularization': 0}
    flow_types = {'geodesic': 1.}
    regularization_epsilon = EPS

    iterations = 150
    step = 1.  # step size
    gvf_w = 1.
    vo_w = 1.
    region_w = 1.
    chanvese_w = {'area_w': 0., 'length_w': 1., 'inside_w': 1., 'outside_w': 1.}

    # Level set initializations
    __phi = []  # LevelSetFunction
    img = None  # the analyzed image (of the data)
    img_rgb = 0
    region = None  # region constraint

    imgGradient = None  # gradient of the analyzed image
    g = None  # internal force
    g_x = g_y = None  # g derivatives

    f = None  # external force (for GVF)
    f_x = f_y = None
    __psi = []  # internal force, for open contours;  LevelSetFunction

    def __init__(self, img, **kwargs):
        """
        Initialize factory.

        :param img: the image upon which the level set is started.
        :param imb_rgb: an rgb image if exists
        :param step: time step for advancing the level set function.
        """
        step = kwargs.get('step', 0.05)
        self.step = step
        if 'img_rgb' in list(kwargs.keys()):
            self.img_rgb = kwargs['img_rgb']

        if isinstance(img, RasterData):
            self.img = img.data
            self.data = img
        else:
            self.img = img

    def init_phi(self, **kwargs):
        r"""
        Builds an initial smooth function (Lipschitz continuous) that represents the interface as the set where
        phi(x,y,t) = 0 is the curve.
        The function has the following characteristics:
        .. math:: \begin{cases}
            phi(x,y,t) > 0 & \forall (x,y) \in \Omega \\
            phi(x,y,t) < 0 & \forall (x,y) \not\in \Omega \\
            phi(x,y,t) = 0 & \forall (x,y) on curve \\

        *characteristics of the function*
            :param processing_props: properties for gradient and differentiation:
                'gradientType' - distance computation method
                'sigma' - for smoothing
                'ksize' - for smoothing

            :param width: the width of the area inside the curve
            :param height: the height of the area inside the curve
            :param start: starting point for the inside area of the curve (x_start, y_start)
            :param reularization: regularizataion note for heaviside function
            :param function_type: 'rectangle' (default); 'vertical' or 'horizontal' (for open contours)

            :type processing_props: dict
            :type width: int
            :type height: int
            :type function_type: str
            :type start: tuple
            :type regularization: int 0,1,2

        :return:
        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5}
        processing_props.update(kwargs['processing_props'])
        img_height, img_width = self.img.shape[:2]
        width = kwargs.get('width', img_width / 4)
        height = kwargs.get('height', img_height / 4)
        func_type = kwargs.get('function_type', 'rectangle')
        start_point = kwargs.get('start', (img_height / 2 - height, img_width / 2 - width))
        regularization = kwargs.get('regularization_note', 0)

        phi = LevelSetFunction.build_function(self.img.shape[:2], func_type = func_type,
                                              height = height, width = width, start = start_point)

        x = np.arange(img_width)
        y = np.arange(img_height)
        xx, yy = np.meshgrid(x, y)
        dists = np.sqrt(xx ** 2 + yy ** 2)
        dists /= np.linalg.norm(dists)

        self.__phi.append(
            LevelSetFunction(cv2.GaussianBlur(phi * dists, (9, 9), 0), regularization_note = regularization,
                             epsilon = 1e-8,
                             **processing_props))

    def init_psi(self, psi, **kwargs):
        """
         Initializes the psi function (open edges)

        :param psi: the function
        :param processing_props: gradient process dictionary

        :type psi: nd-array mxn

        """
        x = np.arange(psi.shape[1])
        y = np.arange(psi.shape[0])
        xx, yy = np.meshgrid(x, y)
        dists = np.sqrt(xx ** 2 + yy ** 2)
        dists /= np.linalg.norm(dists)
        self.__psi.append(LevelSetFunction(psi * dists, self.processing_props))

    @property
    def phi(self, index = 0):
        """
        Returns the level set function phi according to the index
        :return: LevelSetFunction self.__phi
        """

        return self.__phi[index]

    @property
    def psi(self, index = 0):
        """
        Returns the level set function phi according to the index
        :return: LevelSetFunction self.__psi
        """

        return self.__psi[index]

    def init_g(self, g, **kwargs):
        """
        Initializes the g function (edge function)
        :param g: the function
        :param kwargs: gradient process dictionary

        """
        self.g = g
        self.g_x, self.g_y = mt.computeImageDerivatives(g, 1, **kwargs)

    def init_f(self, f, **kwargs):
        """
        Initializes the g function (edge function)
        :param g: the function
        :param kwargs: gradient process dictionary

        """
        self.f = f
        self.f_x, self.f_y = mt.computeImageDerivatives(f, 1, **kwargs)

    def init_region(self, method, **kwargs):
        """
        Initializes region function
        :param kwargs:
            :param method: type of the region wanted: 'classification', 'saliency'

            inputs according to method:
            saliency:
                    inputs according to Saliency class
            classification:
            :param winSizes: array or list with different window sizes for classification
            :param class: the classes which are looked for.

            :param

        :return:
        """
        sigma = kwargs.get('sigma', 2.5)
        if method == 'saliency':
            inputs = {'feature': 'normals',
                      'saliency_method': 'frequency',
                      'dist_type': 'Euclidean',
                      'filter_sigma': [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma],
                      'filter_size': 0,
                      'scales_number': 3,
                      'verbose': True}
            inputs.update(kwargs)

            region = sl.distance_based(self.img_rgb, **inputs)
        elif method == 'classification':
            inputs = {'winSizes', np.linspace(0.1, 10, 5),
                      'class', 1}
            inputs.update(kwargs)

            from ClassificationFactory import ClassificationFactory as Cf
            classified, percentMap = Cf.SurfaceClassification(self.img, inputs['winSizes'])
            region = classified.classification(inputs['class'])

        region = 255 - cv2.GaussianBlur(region, ksize = (3, 3), sigmaX = sigma)
        region = cv2.normalize(region.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        self.region = region

    def flow(self, flow_type, function, *args, **kwargs):
        r"""
        Return the flow of the level set according to the type wanted

        :param flow_type: can be one of the following:
            'constant':

            .. math:: Ct = N \Rightarrow phi_t = |\nabla \varphi|

            'curvature':

            .. math:: Ct = kN \Rightarrow phi_t = div(\nabla \varphi / |\nabla \varphi|)|\nabla \varphi|

            'equi-affine':

             .. math:: Ct = k^(1/3) N \Rightarrow phi_t = (div(\nabla \varphi / |\nabla \varphi|))^(1/3)*|\nabla \varphi|

            'geodesic': geodesic active contours, according to Casselles et al., 1997

             .. math::  Ct = (g(I)k -\nabla(g(I))N)N \Rightarrow
                        phi_t = [g(I)*div(\nabla \varphi / |\nabla \varphi|))^(1/3))*|\nabla \varphi|

            'band': band velocity, according to Li et al., 2006.

        :param function: the level set according to which the flow goes (usually phi)
        :param open_flag: boolean for open flag
        :param chanvese_w: weights for chan vese flow: area_w, length_w, inside_w, outside_w

        **Optionals**

        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param sigma: sigma for LoG gradient
        :param ksize: kernel size, for blurring and derivatives
        :param regularization: regularization note for heaviside and dirac

        *Band velocity optionals*
        :param band_width: the width of the contour. default: 5
        :param threshold: for when the velocity equals zero. default: 0.5
        :param stepsize. default 0.05

        :type flow_type: str
        :type function: LevelSetFunction
        :type open_flag: bool
        :type chanvese_w: dict
        :type gradientType: dict
        :type sigma: float
        :type regularization: int 0,1,2
        :type band_width: int
        :type threshold: float
        :type stepsize: float

        :return: phi_t

        """

        flow = None
        open_flag = args[0]
        processing_props = args[1]

        if flow_type == 'constant':
            flow = np.abs(function.norm_nabla)

        if flow_type == 'curvature':
            flow = function.kappa * function.norm_nabla

        if flow_type == 'equi-affine':
            flow = np.cbrt(function.kappa) * function.norm_nabla

        if flow_type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curvature flow !!

            flow = self.g * function.kappa * function.norm_nabla + (self.g_x * function._x + self.g_y * function._y)
            if open_flag:
                psi_t = self.g * self.phi.kappa * self.psi.norm_nabla + (
                        self.g_x * self.psi._x + self.g_y * self.psi._y)
                self.psi.update(self.psi.value - psi_t, regularization_note = processing_props['regularization'])

        if flow_type == 'chan-vese':

            img = self.img

            c1 = np.sum(img * self.phi.heaviside) / np.sum(self.phi.heaviside)
            c2 = np.sum(img * (1 - self.phi.heaviside)) / np.sum(1 - self.phi.heaviside)

            if np.isnan(c1):
                c1 = 0
            if np.isnan(c2):
                c2 = 0
            weights = self.chanvese_w
            flow = self.phi.dirac_delta * (weights['length_w'] * self.phi.kappa - weights['area_w'] -
                                           weights['inside_w'] * (img - c1) ** 2 +
                                           weights['outside_w'] * (img - c2) ** 2)

        if flow_type == 'band':
            vb = self.__compute_vb(**processing_props)
            flow = vb * self.phi.norm_nabla
            psi_t = vb * self.psi.norm_nabla
            self.psi.update(self.psi.value + psi_t, **processing_props)

        return cv2.GaussianBlur(flow, (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

    def __drawContours(self, function, ax, **kwargs):
        """
        Draws the contours of a specific iteration

        :param phi: the potential function
        :param image: the image on which the contours will be drawn

        :return: the figure

        """

        ax.cla()
        ax.axis("off")
        ax.set_ylim([self.img.shape[0], 0])
        ax.set_xlim([0, self.img.shape[1]])

        color = kwargs.get('color', 'b')
        open_flag = kwargs.get('open', False)

        if np.any(self.img_rgb) != 0:
            temp = self.img_rgb
        else:
            temp = self.img
        img = kwargs.get('image', temp)

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

        if open_flag:
            psi_ind = np.nonzero(self.psi.value < 0)
            psi_ind = np.vstack(psi_ind).T

        for c in contours:
            if open_flag:
                # collide_ind = mt.intersect2d(np.int64(np.round(c[:])), psi_ind)
                # if collide_ind.shape[0] == 0:
                #     pass
                #
                # elif collide_ind.shape[0] == c.shape[0]:
                #     continue
                # else:
                collide_ind = np.isin(np.int64(np.round(c[:])), psi_ind)

                c = c[np.sum(collide_ind, axis = 1).astype('bool')]
                if c.shape[0] == 0:
                    continue

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
        band_width = kwargs.get('band_width', 1.e-3)
        threshold = kwargs.get('threshold', 0.5)
        tau = kwargs.get('stepsize', 0.05)
        phi = self.phi.value

        # Define the areas for R and R'
        R = (phi > 0) * (phi <= band_width)
        R_ = (phi >= -band_width) * (phi < 0)

        SR = np.zeros(self.img.shape[:2])
        SR_ = np.zeros(self.img.shape[:2])

        SR[R] = np.mean(self.img[R])
        SR_[R_] = np.mean(self.img[R_])
        vb = 1 - (SR_ - SR) / (np.linalg.norm(SR + SR_) + EPS)
        vb[vb < threshold] = 0
        vb *= tau
        #  vb[np.where(phi != 0)] = 0
        return vb * self.phi.kappa

    def __compute_vt(self, vectorField, **kwargs):
        """
        Computes the vector field derivative in each direction according to
        .. math:: v_t = g(|\nabla f|)\nabla^2 * v - h(|\nabla f|)*(v - \nabla f)

        :param vectorField: usually created based on an edge map; nxmx2 (for x and y directions)
        :param edge_map:
        :param kappa: curvature map

        **Optionals**
        gradient computation parameters

        :param gradientType:
        :param ksize:
        :param sigma:

        :return vt: velocity for x and y directions
         :type vt: nd-array nxmx2
        """

        # compute the derivatives of the edge map at each direction
        nabla_edge = mt.computeImageGradient(self.f, **kwargs)

        # compute the laplacian of the vector field
        laplacian_vx = cv2.Laplacian(vectorField[:, :, 0], cv2.CV_64F)
        laplacian_vy = cv2.Laplacian(vectorField[:, :, 1], cv2.CV_64F)

        # compute the functions that are part of the vt computation
        g = np.exp(-nabla_edge / (self.phi.kappa + EPS) ** 2)
        h = 1 - g

        vx_t = g * laplacian_vx - h * (vectorField[:, :, 0] - self.f)
        vy_t = g * laplacian_vy - h * (vectorField[:, :, 1] - self.f)

        # plt.quiver(vx_t, vy_t, scale=25)

        return np.stack((-vx_t, -vy_t), axis = 2)

    def __compute_vo(self, **kwargs):
        """
        Computes velocity under orhogonality constraint (the force is computed so that the contours will be orthogonal
        to another function (psi)

        :param kwargs:

        :return:
        """
        psi = kwargs.get('function', self.psi)  # if no other function is given it will be moving according to psi

        grad_psi_grad_phi = psi._x * self.phi._x + psi._y * self.phi._y

        return grad_psi_grad_phi / (self.phi.norm_nabla + EPS)

    def moveLS(self, **kwargs):
        """
        The function that moves the level set until the desired contour is reached

        :param flow_type -  flow types and their weight (string, weight):
        'constant', 'curvature', 'equi-affine', 'geodesic', 'chan-vese', 'band'

        *NOTE: The 'chan-vese' flow requires weights (chanvese_w), for the four components of the model:
         {area_w, length_w, inside_w, outside_w} (dictionary)

        :param gvf_flag: flag to add gradient vector flow
        :param open_flag: flag for open contours

        :return the contours after level set

         """
        # ------inputs--------
        verbose = kwargs.get('verbose', False)
        open_flag = kwargs.get('open_flag', False)
        if np.any(self.img_rgb) != 0:
            temp = self.img_rgb
        else:
            temp = self.img

        img_showed = kwargs.get('image_showed', temp)

        # -------- initializations ---------
        flow_types = self.flow_types
        regularization_epsilon = self.regularization_epsilon
        iterations = self.iterations

        gvf_w = self.gvf_w
        vo_w = self.vo_w
        region_w = self.region_w
        processing_props = self.processing_props

        fig, ax = plt.subplots(num = 1)
        if np.any(self.img_rgb) != 0:
            mt.imshow(self.img_rgb)
        else:
            mt.imshow(self.img)
        ax2 = plt.figure("phi")
        mt.imshow(self.phi.value)
        fig3, ax3 = plt.subplots(num = 'kappa')
        mt.imshow(self.phi.kappa)
        fig4, ax4 = plt.subplots(num = 'psi')
        mt.imshow(self.psi.value)

        for i in range(iterations):
            if verbose:
                print(i)
                if i > 26:
                    print('hello')
            intrinsic = np.zeros(self.img.shape[:2])
            extrinsic = np.zeros(self.img.shape[:2])

            # ---------- intrinsic movement ----------
            # regular flows
            for item in list(flow_types.keys()):
                if flow_types[item] == 0:
                    continue
                intrinsic += flow_types[item] * self.flow(item, self.phi, open_flag, processing_props)

                if verbose:
                    if np.any(intrinsic > 20):
                        print(i)

            # region force
            intrinsic += region_w * self.region * self.phi.norm_nabla

            # open contour
            if open_flag:
                #     band_props.update(processing_props)
                #     vb = self.__compute_vb(**band_props)
                #     intrinsic += vb * self.phi.norm_nabla
                #     psi_t = band_w * vb * self.psi.norm_nabla + flow_types['geodesic']*self.flow('geodesic', self.psi, **processing_props)
                #     self.psi.update(
                #         cv2.GaussianBlur(self.psi.value - psi_t, (processing_props['ksize'], processing_props['ksize']),
                #                          sigmaX = processing_props['sigma']))
                plt.figure('psi')
                l_curve, ax4 = self.__drawContours(self.psi.value, ax4, color = 'b', image = self.psi.value)
                plt.pause(.5e-10)

            # ---------------extrinsic movement ----------
            v = np.stack((self.f_x, self.f_y), axis = 2)
            vt = self.__compute_vt(v, **processing_props)
            v += vt
            extrinsic = (v[:, :, 0] * self.phi._x + v[:, :, 1] * self.phi._y) * gvf_w

            # for constrained contours
            extrinsic += self.__compute_vo() * vo_w
            # extrinsic += (1 - mult_phi)
            #  self.psi += extrinsic
            plt.figure('kappa')
            mt.imshow(self.phi.kappa)
            plt.pause(.5e-10)

            phi_t = self.step * (intrinsic - extrinsic)
            self.phi.update(
                cv2.GaussianBlur(self.phi.value + phi_t, (processing_props['ksize'], processing_props['ksize']),
                                 processing_props['sigma']), epsilon = regularization_epsilon)
            if np.max(np.abs(phi_t)) <= 5e-5:
                print('done')
                return
            plt.figure('phi')
            mt.imshow(self.phi.value)

            # if open_flag:
            #     for curve in l_curve:
            #         self.phi.value[curve._y.astype('int'), curve._x.astype('int')] = 0

            plt.figure(1)
            if open_flag:
                _, ax = self.__drawContours(self.phi.value, ax, color = 'r', image = img_showed,
                                            open = True)

            else:
                _, ax = self.__drawContours(self.phi.value, ax, color = 'r', image = img_showed)
            plt.pause(.5e-10)
        plt.show()
        print('Done')
