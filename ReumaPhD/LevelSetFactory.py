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
from LevelSetFunction import LevelSetFunction

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
    phi = None  # LevelSetFunction
    img = None  # the analyzed image (of the data)
    region = None  # region constraint

    step = 0  # step size

    imgGradient = None  # gradient of the analyzed image
    norm_nabla_phi = None  # |\nabla \phi| - level set gradient size
    g = None  # internal force
    g_x = g_y = None  # g derivatives

    f = None  # external force (for GVF)
    f_x = f_y = None
    psi = None  # internal force, for open contours;  LevelSetFunction

    def __init__(self, img, **kwargs):
        """
        initialize factory.

        :param img: the image upon which the level set is started.
        :param kwargs:
            step - time step for advancing the level set function.
        """
        step = kwargs.get('step', 0.05)
        self.step = step
        if 'img_rgb' in kwargs.keys():
            self.img_rgb = kwargs['img_rgb']

        if isinstance(img, RasterData):
            self.img = img.data
            self.data = img
        else:
            self.img = img

    def init_phi(self, **kwargs):
        """
        Builds an initial smooth function (Lipsschitz continuous) that represents the interface as the set where
        phi(x,y,t) = 0 is the curve.
        The function has the following characteristics:
        phi(x,y,t) > 0 for (x,y) \in \Omega
        phi(x,y,t) < 0 for (x,y) \not\in \Omega
        phi(x,y,t) = 0 for (x,y) on curve

        :param kwargs:
        characteristics of the function:
            :param width: the width of the area inside the curve
            :param height: the height of the area inside the curve
            :param type: 'rectangle' (default); 'vertical' or 'horizontal' (for open contours)

        :return:
        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5}

        img_height, img_width = self.img.shape[:2]
        width = kwargs.get('width', img_width / 4)
        height = kwargs.get('height', img_height / 4)
        func_type = kwargs.get('type', 'rectangle')
        start_point = kwargs.get('start', (img_height / 2 - height, img_width / 2 - width))

        phi = LevelSetFunction.build_function(self.img.shape[:2], type = func_type,
                                              height = height, width = width, start = start_point)

        x = np.arange(img_width)
        y = np.arange(img_height)
        xx, yy = np.meshgrid(x, y)
        dists = np.sqrt(xx ** 2 + yy ** 2)
        # dists/= np.linalg.norm(dists)

        self.phi = LevelSetFunction(cv2.GaussianBlur(phi * dists, (9, 9), 0), **processing_props)

    def init_psi(self, psi, **kwargs):
        """
         Initializes the psi function (open edges)
        :param psi: the function
        :param kwargs: gradient process dictionary

        """
        self.psi = LevelSetFunction(psi, **processing_props)

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

    def flow(self, type, function, **kwargs):
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

        :param function: the level set according to which the flow goes (usually phi), a LevelSetFunction object


        ------- optionals ---------
        :param function: the function of which the flow will move

        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param sigma: sigma for LoG gradient
        :param ksize: kernel size, for blurring and derivatives

        ---- band velocity optionals ----
        :param band_width: the width of the contour. default: 5
        :param threshold: for when the velocity equals zero. default: 0.5
        :param stepsize. default 0.05

        :return: phi_t
        """

        processing_props = kwargs.get('processing_props', {'sigma': 2.5, 'ksize': 3, 'gradientType': 'L1'})
        if 'processing_props' in kwargs.keys():
            processing_props.update(kwargs['processing_props'])

        if type == 'constant':
            return np.abs(function.norm_nabla)

        if type == 'curvature':
            return function.kappa * function.norm_nabla

        if type == 'equi-affine':
            return np.cbrt(function.kappa) * function.norm_nabla

        if type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curvature flow !!

            return cv2.GaussianBlur(
                self.g * function.kappa * function.norm_nabla + (self.g_x * function._x + self.g_y * function._y),
                                    (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

        # if type == 'open':
        #     self.norm_nabla_psi = np.sqrt(self.psi_x ** 2 + self.psi_y ** 2)
        #     return cv2.GaussianBlur(
        #         self.g * self.kappa * self.norm_nabla_psi + (self.g_x * self.psi_x + self.g_y * self.psi_y),
        #                             (processing_props['ksize'], processing_props['ksize']),
        #                             processing_props['sigma'])

        if type == 'band':
            return self.__compute_vb(**kwargs) * function.norm_nabla

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
        phi = self.phi.value
        # Define the areas for R and R'
        R = (phi > 0) * (phi <= band_width)
        R_ = (phi >= -band_width) * (phi < 0)

        SR = np.zeros(self.img.shape[:2])
        SR_ = np.zeros(self.img.shape[:2])

        SR[R] = np.mean(self.img[R])
        SR_[R_] = np.mean(self.img[R_])
        vb = 1 - (SR_ - SR) / (np.linalg.norm(SR + SR_))
        vb[vb < threshold] = 0
        vb *= tau
        #  vb[np.where(phi != 0)] = 0
        return vb * self.phi.kappa

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
        img_showed = kwargs.get('image_showed', self.img)
        iterations = kwargs.get('iterations', 150)
        verbose = kwargs.get('verbose', False)

        gvf_w = kwargs.get('gvf_w', 1.)
        vo_w = kwargs.get('vo_w', 1.)
        region_w = kwargs.get('region_w', 1.)

        if 'processing_props' in kwargs.keys():
            processing_props.update(kwargs['processing_props'])

        fig, ax = plt.subplots(num = 1)
        ax2 = plt.figure("phi")
        fig3, ax3 = plt.subplots(num = 'kappa')
        fig4, ax4 = plt.subplots(num = 'psi')

        mult_phi = np.zeros(self.img.shape[:2])

        mt.imshow(self.img)
        for i in range(iterations):

            if verbose:
                print i
                if i > 26:
                    print 'hello'
            intrinsic = np.zeros(self.img.shape[:2])
            extrinsic = np.zeros(self.img.shape[:2])

            # ---------- intrinsic movement ----------
            # regular flows
            for item in flow_types:
                intrinsic += self.flow(item, self.phi, **processing_props)


                if verbose:
                    if np.any(intrinsic > 20):
                        print i

            # region force
            intrinsic += region_w * self.region * self.phi.norm_nabla

            # open contour
            if open_flag:
                intrinsic += self.flow('band', self.phi, **processing_props)
                psi_t = self.flow('band', self.psi, **processing_props)
                self.psi.update(
                    cv2.GaussianBlur(self.psi.value + psi_t, (processing_props['ksize'], processing_props['ksize']),
                                     sigmaX = processing_props['sigma']))
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
            extrinsic += (1 - mult_phi)
            #  self.psi += extrinsic
            plt.figure('kappa')
            mt.imshow(self.phi.kappa)
            plt.pause(.5e-10)

            phi_t = self.step * (intrinsic - extrinsic)
            self.phi.update(
                cv2.GaussianBlur(self.phi.value + phi_t, (processing_props['ksize'], processing_props['ksize']),
                                 processing_props['sigma']))
            if np.all(np.abs(self.phi.kappa)) <= 3e-7:
                print 'done'
                return
            plt.figure('phi')
            mt.imshow(self.phi.value)

            # if open_flag:
            #     for curve in l_curve:
            #         self.phi.value[curve._y.astype('int'), curve._x.astype('int')] = 0

            plt.figure(1)
            _, ax = self.__drawContours(self.phi.value, ax, color = 'r', image = img_showed)
            plt.pause(.5e-10)
        plt.show()
        print ('Done')

if __name__ == '__main__':
    # initial input:
    img_orig = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\Channel91.png'), cv2.COLOR_BGR2RGB)
    img_normed = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point
    sigma = 2.5  # blurring

    ls_obj = LevelSetFactory(img_normed, img_rgb = img_orig, step = 5.)

    processing_props = {'sigma': 5, 'ksize': 5, 'gradientType': 'L2'}
    ls_obj.init_phi(start = (100, 0), width = img_orig.shape[1] / 2, height = 5)

    plt.figure()
    mt.imshow(ls_obj.phi.value)
    plt.show()

    # ------- Initial limits via psi(x,y) = 0 ---------------------------
    # option 1: horizontal line
    # psi[img_height - 2 * height: img_height - height, :] = -1
    # option 2: vertical line
    psi = -np.ones(img_orig.shape[:2])
    width_boundary = 10
    psi[:, 0: width_boundary] = 1
    psi[:, -width_boundary:] = 1
    # psi[0:width_boundary, : ] = -1
    # psi[-width_boundary:, :] = -1
    ls_obj.init_psi(cv2.GaussianBlur(psi, (5, 5), 2.5), **processing_props)
    plt.imshow(psi)
    plt.show()
    # ---------------------------------------------------------------------

    # ---------------- Intrinsic forces maps ------------------------------

    # Force I - function g:
    # can be either constant, weights, or function
    # option 1: edge map g = 1/(1+|\nabla G(I) * I|).
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.normalize(img_gray.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    imgGradient = mt.computeImageGradient(img_gray, gradientType = 'L2', sigma = 1.5)
    g = 1 / (1 + imgGradient **2)
    plt.figure('g')
    mt.imshow(g)

    # option 2: saliency map
    # g = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='pixel_val')
    ls_obj.init_g(g, **processing_props)

    # Force II - region constraint:
    ls_obj.init_region('saliency', saliency_method = 'context', sigma = 0.5, feature = 'pixel_val')
    plt.figure('region')
    mt.imshow(ls_obj.region)

    # Force III - open contours:

    # ---------------------------------------------------------------------

    # ----------------Extrinsic forces maps (vector field)------------------

    # The map which the GVF will be defined by
    # option 1: the image itself
    f = 1 - cv2.GaussianBlur(img_gray, ksize = (5, 5), sigmaX = sigma)

    # # option 2: edge map
    # f = cv2.Canny(img_orig, 10, 50)
    #  f = cv2.GaussianBlur(f, ksize = (5, 5), sigmaX = 1.5)

    # # option 3: saliency map
    # f = cv2.GaussianBlur(region, ksize = (5, 5), sigmaX = sigma)

    #  f = np.zeros(img_gray.shape)
    ls_obj.init_f(f, **processing_props)
    plt.figure('f')
    mt.imshow(f)
    plt.show()

    # PAY ATTENTION to region's weight - it should be a scale or two smaller than the others

    ls_obj.moveLS(open_flag = False, processing_props = processing_props, iterations = 500,
                  gvf_w = 1.,
                  vo_w = 0.,
                  region_w = 0.002)
