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
import Saliency as sl
from LevelSetFunction import LevelSetFunction
from RasterData import RasterData

EPS = np.finfo(float).eps


class LevelSetFlow:
    # General initializations
    __processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5, 'regularization': 0}
    __flow_types = {'geodesic': 1.}
    __regularization_epsilon = EPS
    __iterations = 150
    __step = 1.  # step size

    # Weighting initializations
    __gvf_w = 1.
    __vo_w = 0.
    __region_w = 1.
    __chanvese_w = {'area_w': 0., 'length_w': 1., 'inside_w': 1., 'outside_w': 1.}

    # Level set initializations
    __Phi = []  # LevelSetFunction
    __img = None  # the analyzed image (of the data)
    __img_rgb = 0

    # Constraints initializations
    __imgGradient = None  #

    __g = None  # internal force
    __g_x = __g_y = None  # g derivatives

    __region = None  # region constraint

    __f = None  # external force (for GVF)
    __f_x = __f_y = None
    __psi = []  # internal force, for open contours;  LevelSetFunction

    def __init__(self, img, **kwargs):
        """
        Initialize factory.

        :param img: the image upon which the level set is started.
        :param imb_rgb: an rgb image if exists
        :param step: time step for advancing the level set function.
        :param flow_types: the flow types according to which the level set moves with their weights. According to
        :py:meth:`~LevelSetFactory.LevelSetFactory.flow`
        :param weights: all weights to move the level set. According to
        :py:meth:`~LevelSetFactory.LevelSetFactory.set_weights`

        :type img: np.array
        :type img_rgb: np.array
        :type step: float
        :type flow_types: dict
        :type weights: dict

        """

        self.__step = kwargs.get('step', 0.05)
        if 'img_rgb' in list(kwargs.keys()):
            self.__img_rgb = kwargs['img_rgb']

        if isinstance(img, RasterData):
            self.__img = img.data
            self.data = img
        else:
            self.__img = img

        self.__Phi = [LevelSetFunction(np.zeros(img.shape))]
        self.__psi = [LevelSetFunction(np.zeros(img.shape))]
        self.__region = np.zeros(img.shape)
        self.__f = np.zeros(img.shape)
        self.__g = np.zeros(img.shape)


        if 'weights' in list(kwargs.keys()):
            self.set_weights(**kwargs['weights'])

        if 'flow_types' in list(kwargs.keys()):
            self.set_flow_types(**kwargs['flow_types'])

    @property
    def processing_props(self):
        return self.__processing_props

    @property
    def flow_types(self):
        """
        Types of flow used to progress the curves

        Default: {"geodesic"}

        :rtype: dict
        """
        return self.__flow_types

    def set_flow_types(self, **kwargs):
        """
        Set the flow types and their weights according to which the level set will move

        Options are as in :py:meth:`~LevelSetFactory.LevelSetFactory.flow`

        For example:

        .. code-block:: python

            set_flow_types(geodesic=1., curvature=.2, equi_affine: 0.5})

        """
        flow_types = {'geodesic': 1.,
                      'curvature': 0.,
                      'equi_affine': 0.,
                      'chan_vese': 0.,
                      'band': 0.}
        flow_types.update(kwargs)
        self.__flow_types = flow_types

    @property
    def regularization_epsilon(self):
        """

        :rtype: float
        """
        return self.__regularization_epsilon

    @property
    def iterations(self):
        """
        Number of iterations for the level set

        Default: 150

        :rtype: int

        """
        return self.__iterations

    @property
    def step(self):
        """
        Step size for the level set progression

        Default: 0.05

        :rtype: float
        """
        return self.__step

    @property
    def gvf_w(self):
        """
        Gradient vector flow weight

        Default: 1.

        :rtype: float

        """
        return self.__gvf_w

    @property
    def vo_w(self):
        """
        Open velocity flow weight

        Default: 1.

        :rtype: float

        """
        return self.__vo_w

    @property
    def region_w(self):
        """
        Region weight

        Default: 1.

        :rtype: float

        """
        return self.__region_w

    @property
    def chanvese_w(self):
        """
        Chan-Vese weights: area, length, inside and outside weights

        Default:

           * length, inside and outside weights: 1.
           * area: 0.

        :rtype: dict

        """
        return self.__chanvese_w

    def set_weights(self, **kwargs):
        """
        Set weights for constraints.

        :param chanvese_w: area_w, length_w, inside_w and outside_w
        :param gvf_w: gradient vector flow weight
        :param vo_w: open contour weight (Not working at the moment)
        :param region_w: region constraint weight

        :type chanvese_w: dict
        :type gvf_w: float
        :type vo_w: float
        :type region_w: float

        """
        inputs = {'gvf_w': 1.,
                  'vo_w': 0.,
                  'region_w': 1.,
                  'chanvese_w': {'area_w': 0., 'length_w': 1., 'inside_w': 1., 'outside_w': 1.}}
        if 'chanvese_w' in kwargs:
            chanvese_w = inputs['chanvese_w']
            chanvese_w.update(kwargs['chanvese_w'])
            kwargs['chanvese_w'] = chanvese_w

        inputs.update(kwargs)
        self.__chanvese_w = inputs['chanvese_w']
        self.__region_w = inputs['region_w']
        self.__gvf_w = inputs['gvf_w']
        self.__vo_w = inputs['vo_w']

    @property
    def img(self):
        """
        The analyzed image (of the data)

        :rtype: np.array

        """
        return self.__img

    @property
    def img_rgb(self):
        """
        The analyzed img rgb representation (if exists, otherwise: 0.)

        Default: 0

        :rtype: np.array or 0

        """
        return self.__img_rgb

    @property
    def region(self):
        """
        Region constraint

        :rtype: np.array

        """
        return self.__region

    @property
    def imgGradient(self):
        """
        Gradient of the analyzed image

        :rtype: np.array
        """
        return self.__imgGradient

    @property
    def g(self):
        """
        Internal force (usually edge function)

        :rtype: np.array
        """
        return self.__g

    @property
    def g_x(self):
        """
        First order derivative of the internal force

        :rtype: np.array
        """
        return self.__g_x

    @property
    def g_y(self):
        """
        First order derivative of the internal force

        :rtype: np.array
        """
        return self.__g_y

    @property
    def f(self):
        """
        External force (for gradient vector flow)

        :rtype: np.array
        """
        return self.__g

    @property
    def f_x(self):
        """
        First order derivative of the external force

        :rtype: np.array
        """
        return self.__g_x

    @property
    def f_y(self):
        """
        First order derivative of the external force

        :rtype: np.array
        """
        return self.__g_y

    def phi(self, index = 0):
        """
        Returns the level set function phi according to the index
        :return: LevelSetFunction self.__Phi
        """

        return self.__Phi[index]

    @property
    def psi(self, index = 0):
        """
        Returns the level set function phi according to the index
        :return: LevelSetFunction self.__psi
        """

        return self.__psi[index]


    def init_phi(self, **kwargs):
        r"""
        Builds an initial smooth function (Lipschitz continuous) that represents the interface as the set where
        phi(x,y,t) = 0 is the curve.

        The function has the following characteristics:

        .. math::
            \begin{cases}
             \phi(x,y,t) > 0 & \forall (x,y) \in \Omega \\
             \phi(x,y,t) < 0 & \forall (x,y) \not\in \Omega \\
             \phi(x,y,t) = 0 & \forall (x,y) \textrm{ on curve} \\
             \end{cases}

        with :math:`\left| \nabla \phi \right| = 1`

        **Characteristics of the function**

        :param processing_props: properties for gradient and differentiation:
            - 'gradientType' - distance computation method
            - 'sigma' - for smoothing
            - 'ksize' - for smoothing

        :param radius: if the curve is a circle, the radius should be specified.
        :param center_pt: if the curve is a circle, the center point should be specified.
        :param reularization_note: regularization note for heaviside function
        :param function_type: 'circle' (default); 'vertical' or 'horizontal' (for open contours)

        :type processing_props: dict
        :type radius: int
        :type center_pt: tuple
        :type function_type: str
        :type start: tuple
        :type regularization_note: int 0,1,2

        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5}
        processing_props.update(kwargs['processing_props'])
        img_height, img_width = self.img.shape[:2]
        radius = kwargs.get('radius', np.int(img_width / 4))
        center_pt = kwargs.get('center_pt', [np.int(img_height / 2), np.int(img_width / 2)])
        func_type = kwargs.get('function_type', 'circle')

        regularization = kwargs.get('regularization_note', 0)

        if func_type == 'circle':
            phi = LevelSetFunction.dist_from_circle(center_pt, radius, (img_height, img_width),
                                                    ksize = processing_props['ksize'])

        if np.all(self.__Phi[0].value == 0):
            self.__Phi = []

        self.__Phi.append(
            LevelSetFunction(phi, regularization_note = regularization,
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
        if np.all(self.__psi[0].value == 0):
            self.__psi = []

        self.__psi.append(LevelSetFunction(psi * dists, self.processing_props))

    def init_g(self, g, **kwargs):
        """
        Initializes the g function (edge function)
        :param g: the function
        :param kwargs: gradient process dictionary

        """
        self.__g = g
        self.__g_x, self.__g_y = mt.computeImageDerivatives(g, 1, **kwargs)

    def init_f(self, f, **kwargs):
        """
        Initializes the g function (edge function)
        :param g: the function
        :param kwargs: gradient process dictionary

        """
        self.__f = f
        self.__f_x, self.__f_y = mt.computeImageDerivatives(f, 1, **kwargs)

    def init_region(self, region_method, **kwargs):
        """
        Initializes region function

        :param kwargs:
        :param region_method: type of the region wanted: 'classification', 'saliency'.

        *Inputs according to method:*

        - saliency: inputs according to :py:meth:`~Saliency.distance_based`

        - classification:

        :param winSizes: array or list with different window sizes for classification
        :param class: the classes which are looked for.

        """
        sigma = kwargs.get('sigma', 2.5)
        if region_method == 'saliency':
            inputs = {'feature': 'normals',
                      'method': 'frequency',
                      'dist_type': 'Euclidean',
                      'filter_sigma': [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma],
                      'filter_size': 0,
                      'scales_number': 3,
                      'verbose': True}
            inputs.update(kwargs)

            region = sl.distance_based(self.img, **inputs)
        elif region_method == 'classification':
            inputs = {'winSizes', np.linspace(0.1, 10, 5),
                      'class', 1}
            inputs.update(kwargs)

            from ClassificationFactory import ClassificationFactory as Cf
            classified, percentMap = Cf.SurfaceClassification(self.img, inputs['winSizes'])
            region = classified.classification(inputs['class'])

        region = 255 - cv2.GaussianBlur(region, ksize = (3, 3), sigmaX = sigma)
        region = cv2.normalize(region.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        self.__region = region

    def update_region(self, new_region):
        """
        Updates the region according to a given new_region

        :param new_region: the new region according to which the level set should progress

        :type new_region: np.array


        """
        self.__region = new_region

    def flow(self, flow_type, function, *args, **kwargs):
        r"""
        Return the flow of the level set according to the type wanted

        :param flow_type: can be one of the following:
            - 'constant':

               .. math:: C_t = N \Rightarrow \phi_t = |\nabla \varphi|

            - 'curvature':

               .. math:: C_t = kN \Rightarrow phi_t = div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)|\nabla \varphi|

            - 'equi_affine':

                .. math:: C_t = k^{1/3} N \Rightarrow
                        \phi_t = div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)^{1/3}*|\nabla \varphi|

            - 'geodesic': geodesic active contours, according to Casselles et al., 1997

                .. math::  C_t = (g(I)k -\nabla(g(I))N)N \Rightarrow
                        \phi_t = [g(I)\cdot div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)^{1/3}*|\nabla \varphi|

            - 'band': band velocity, according to Li et al., 2006.

            - 'chan_vese':

        :param function: the level set according to which the flow goes (usually phi)
        :param open_flag: boolean for open flag
        :param chanvese_w: weights for chan vese flow: area_w, length_w, inside_w, outside_w

        **Optionals**

        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param sigma: sigma for LoG gradient
        :param ksize: kernel size, for blurring and derivatives
        :param regularization: regularization note for heaviside and dirac

        **Band velocity optionals**

        :param band_width: the width of the contour. default: 5
        :param threshold: for when the velocity equals zero. default: 0.5
        :param stepsize: default 0.05

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

        if flow_type == 'equi_affine':
            flow = np.cbrt(function.kappa) * function.norm_nabla

        if flow_type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curvature flow !!

            flow = self.g * function.kappa * function.norm_nabla + (self.g_x * function._x + self.g_y *
                                                                    function._y)
            if open_flag:
                psi_t = self.g * self.phi().kappa * self.psi.norm_nabla + (
                        self.g_x * self.psi._x + self.g_y * self.psi._y)
                self.psi.update(self.psi.value - psi_t, regularization_note = processing_props['regularization'])

        if flow_type == 'band':
            vb = self.__compute_vb(**processing_props)
            flow = vb * self.phi().norm_nabla
            psi_t = vb * self.psi.norm_nabla
            self.psi.update(self.psi.value + psi_t, **processing_props)

        return cv2.GaussianBlur(flow, (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

    def mumfordshah_flow(self, img = True, nu = 1):
        """
        Computes the Mumford-Shah flow for a multi-phase level set, according to :cite:`vese.chan2002`.

        Updates the self.Phi variable for

        :param img: the specific image upon which the level sets are computed
        :param nu: the length weight

        :type img: np.array
        :type nu: float

        :return: phi_t - the flows that move the level set
        """

        if img:
            img = self.img
        Phi = self.__Phi

        m_levelsets = len(Phi)
        n_phases = 2 ** m_levelsets
        combinations = '01' * m_levelsets
        # fig, ax = plt.subplots(num = 'panorama')

        import itertools
        counter = 0
        combinations = itertools.combinations(combinations, m_levelsets)
        kappa_flag = True
        for combination in combinations:
            dPhi = self.__ms_element(combination, img)

            for i in range(m_levelsets):
                i = int(i)
                Phi[i].move_function(dPhi[:, :, i])
                counter += 1

                if counter > m_levelsets:
                    kappa_flag = False

                if kappa_flag:
                    Phi[i].move_function(nu * self.phi(i).kappa)

        self.__Phi = Phi

        # mt.draw_contours(self.phi(0).value, ax, self.img_rgb, color = 'b')
        # mt.draw_contours(self.phi(1).value, ax, self.img_rgb, hold = True, color = 'r')
        # plt.pause(.5e-10)


    def __ms_element(self, combination, img):
        """
        An element for summation within the Mumford-Shah model.

        For example, for a 3-phase model, with a unison of the three level sets (all inside):

        ..math::

            \begin{eqnarray}
            \phi_{1t}\rightarrow(u_{0} - c_{000})^2 \delta(\phi_1)H(\phi_2)H(\phi_3) \\
            \phi_{2t}\rightarrow(u_{0} - c_{000})^2 H(\phi_1)\delta(\phi_2)H(\phi_3) \\
            \phi_{3t}\rightarrow(u_{0} - c_{000})^2 H(\phi_1)H(\phi_2)\delta(\phi_3) \\

        This method comoputes the constant of the region, according to the image

        :param combination: the current combination of the level-sets (e.g., '000', '001', '101')
        :param img: the image upon which the level set is moving (for the specific channel)

        :type combination: tuple
        :type img: np.array

        :return: the elements for summation for each level set

        :rtype: list

        """

        H = []
        diracs = []
        mult_dirac = []
        dPhi = np.zeros((img.shape[0], img.shape[1], len(combination)))
        for index in combination:
            i = int(index)
            if i == 0:  # inside the level set
                H.append(self.phi(i).heaviside)
                diracs.append(self.phi(i).dirac_delta)
            else:
                H.append(1 - self.phi(i).heaviside)
                diracs.append(-self.phi(i).dirac_delta)

        import functools
        mult = functools.reduce(lambda x, y: x * y, H)
        c = np.sum(img * mult) / (np.sum(mult))

        for index in range(len(self.__Phi)):
            i = int(index)
            H_ = H.copy()
            H_.pop(i)
            if len(H_) > 1:
                H_ = functools.reduce(lambda x, y: x * y, H_)
            else:
                H_ = H_[0]
            mult_dirac.append(diracs[i] * H_)
            dPhi[:, :, i] += (img - c) ** 2 * mult_dirac[i]

        return dPhi



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
        phi = self.phi().value

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
        return vb * self.phi().kappa

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
        g = np.exp(-nabla_edge / (self.phi().kappa + EPS) ** 2)
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

        grad_psi_grad_phi = psi._x * self.phi()._x + psi._y * self.phi()._y

        return grad_psi_grad_phi / (self.phi().norm_nabla + EPS)

    def moveLS(self, **kwargs):
        r"""
        The function that moves the level set until the desired contour is reached

        :param flow_type -  flow types and their weight (string, weight):

        'constant', 'curvature', 'equi-affine', 'geodesic', 'chan-vese', 'band'

        .. note:: The 'chan-vese' flow requires weights (chanvese_w), for the four components of the model:

        .. code-block:: python

            {area_w, length_w, inside_w, outside_w}

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

        fig, ax = plt.subplots(num = 'img')
        if np.any(self.img_rgb) != 0:
            mt.imshow(self.img_rgb)
        else:
            mt.imshow(self.img)

        ax2 = plt.figure("phi")
        mt.imshow(self.phi().value)
        fig3, ax3 = plt.subplots(num = 'kappa')
        mt.imshow(self.phi().kappa)
        if open_flag:
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
                intrinsic += flow_types[item] * self.flow(item, self.phi(), open_flag, processing_props)

                if verbose:
                    if np.any(intrinsic > 20):
                        print(i)

            # region force
            intrinsic += region_w * self.region * self.phi().norm_nabla
            self.mumfordshah_flow()

            # open contour
            if open_flag:
                #     band_props.update(processing_props)
                #     vb = self.__compute_vb(**band_props)
                #     intrinsic += vb * self.phi().norm_nabla
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
            extrinsic = (v[:, :, 0] * self.phi()._x + v[:, :, 1] * self.phi()._y) * gvf_w

            # for constrained contours
            extrinsic += self.__compute_vo() * vo_w
            # extrinsic += (1 - mult_phi)
            #  self.psi += extrinsic
            plt.figure('kappa')
            mt.imshow(self.phi().kappa)
            plt.pause(.5e-10)

            phi_t = self.step * (intrinsic - extrinsic)
            Phi = LevelSetFunction(self.phi().value + phi_t)

            Phi_t = np.sign(Phi.value) * (1 - (np.sqrt(Phi._x ** 2 + Phi._y ** 2)))

            self.phi().update(
                cv2.GaussianBlur((Phi.value + Phi_t), (processing_props['ksize'], processing_props['ksize']),
                                 processing_props['sigma']), epsilon = regularization_epsilon)
            # if np.max(np.abs(phi_t)) <= 5e-5:
            #     print('done')
            #     return
            plt.figure('phi')
            mt.imshow(self.phi().value)

            # if open_flag:
            #     for curve in l_curve:
            #         self.phi().value[curve._y.astype('int'), curve._x.astype('int')] = 0

            plt.figure('img')
            if open_flag:
                _, ax = self.__drawContours(self.phi().value, ax, color = 'r', image = img_showed,
                                            open = True)

            else:
                colors = 'rgbm'
                for i in range(len(self.__Phi)):
                    if i > 0:
                        _, ax = mt.draw_contours(self.phi(i).value, ax, img = img_showed, hold = True,
                                                 color = colors[i])
                    else:
                        _, ax = mt.draw_contours(self.phi(i).value, ax, img = img_showed, hold = False,
                                                 color = colors[i])
            plt.pause(.5e-10)
        plt.show()
        print('Done')
