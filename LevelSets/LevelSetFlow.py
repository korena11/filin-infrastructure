'''
infragit
reuma\Reuma
07, Jan, 2018 
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import Utils.MyTools as mt
from LevelSets.LevelSetFunction import LevelSetFunction
from DataClasses.RasterData import RasterData

EPS = np.finfo(float).eps


class LevelSetFlow:
    # General initializations
    __processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5, 'regularization': 0}
    __flow_types = {'geodesic': 1.}
    __regularization_epsilon = EPS
    __iterations = 150
    __step = 1.  # step size

    # Weighting initializations
    __gvf_w = 0.
    __vo_w = 0.
    __region_w = 0.
    __ms_w = 0.
    # Level set initializations
    __Phi = []  # LevelSetFunction
    __img = []  # the analyzed image (of the data)
    __img_rgb = 0

    # Constraints initializations
    __imgGradient = None  #

    __g = None  # internal force
    __g_x = __g_y = None  # g derivatives

    __region = None  # region constraint

    __GVF = None  # external force (for GVF)

    __psi = []  # internal force, for open contours;  LevelSetFunction *currently not in use*

    # ------------------------- INITIALIZATIONS AND SETTINGS----------------------
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
        self.__iterations = kwargs.get('iterations', 150)
        self.__iterations += 2
        if img is not None:
            self.init_img(img)

        if 'img_rgb' in list(kwargs.keys()):
            self.__img_rgb = kwargs['img_rgb']
        elif img is not None:
            self.__img_rgb = img

        if 'weights' in list(kwargs.keys()):
            self.set_weights(**kwargs['weights'])

        if 'flow_types' in list(kwargs.keys()):
            self.set_flow_types(**kwargs['flow_types'])

        self.scale_x = 1
        self.scale_y = 1

    def init_img(self, img):
        """
        Set an image for the level set flow, according to which the flow will move (each of the Phi's)

        :param img: the image to set

        """
        if isinstance(img, RasterData):
            self.__img.append(img.data)
            # self.data = img
        else:
            if isinstance(img, list):
                self.__img = img
            else:
                self.__img.append(img)
        if len(self.__Phi) ==0:
            self.__GVF = np.zeros(self.img().shape)
            self.__g = np.zeros(self.img().shape)
            self.__Phi = [LevelSetFunction(np.zeros(self.img().shape))]
            self.__psi = [LevelSetFunction(np.zeros(self.img().shape))]
            self.__region = np.zeros(self.img().shape)

    def set_flow_types(self, **kwargs):
        """
        Set the flow types and their weights according to which the level set will move

        Options are as in :py:meth:`~LevelSetFactory.LevelSetFactory.flow`

        For example:

        .. code-block:: python

            set_flow_types(geodesic=1., curvature=.2, equi_affine: 0.5})

        """
        flow_types = {'geodesic': 0.,
                      'curvature': 0.,
                      'equi_affine': 0.,
                      'band': 0.}
        flow_types.update(kwargs)
        self.__flow_types = flow_types

    def set_weights(self, **kwargs):
        """
        Set weights for constraints. All are set to zero as defualt

        :param gvf_w: gradient vector flow weight
        :param vo_w: open contour weight (Not working at the moment)
        :param region_w: region constraint weight

        :type gvf_w: float
        :type vo_w: float
        :type region_w: float

        """
        inputs = {'gvf_w': 0.,
                  'vo_w': 0.,
                  'region_w': 0.,
                  'ms_w': 0.}

        inputs.update(kwargs)
        self.__region_w = inputs['region_w']
        self.__gvf_w = inputs['gvf_w']
        self.__vo_w = inputs['vo_w']
        self.__ms_w = inputs['ms_w']

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
        self.__g_x, self.__g_y = mt.computeImageDerivatives_numeric(g, 1, **kwargs)

    def init_GVF(self, f, mu, iterations, ksize=3, sigma=1.,
                 resolution=1., blur_window=(0, 0), **kwargs):
        """
        Initializes the vector field (gradient vector flow)

        Computes the GVF of an edge map f according the the paper of :cite:`Xu.Prince1998`

        :param f: the edge map according to which the GVF is computed
        :param mu: the GVF regularization coefficient. (according to the paper: 0.2).
        :param iterations: the number of iterations that will be computed. (according to the example: 80).
        :param ksize: size of the differentiation window
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param blur_window: tuple of window size for blurring

        :type f: np.array
        :type mu: float
        :type iterations: int
        :type ksize: int
        :type resolution: float
        :type blur_window: tuple (2 elements)

        """
        from GVF import GVF
        self.__GVF = GVF(f, mu, iterations, ksize=ksize, sigma=sigma,
                         resolution=resolution, blur_window=blur_window)

    def init_region(self, region):
        """
        Initializes region function. Normalizes between (-1,1)

        :param region: region function initialize

        """

        # region = cv2.normalize(region.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        self.__region = region

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
        :param function_type:

            - 'circle' (default);
            - 'ellipse'
            - 'egg_crate' - egg crate function :math:`x^2 + y^2 - amplitude * (\sin^2 x + \sin^2 y*)
            - 'periodic_circles' - more like squares at dx and dy distances from one another
            - 'checkerboard' - binary squares (from scikit-image)

        :param epsilon: threshold to be considered zero. Default: 1e-4

        :type processing_props: dict
        :type radius: int
        :type center_pt: tuple
        :type function_type: str
        :type start: tuple
        :type regularization_note: int 0,1,2
        :type epsilon: float

        """
        from MyTools import scale_values
        eps = kwargs.get('epsilon', 1e-4)
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5, 'resolution': 1.}
        if 'processing_props' in kwargs:
            processing_props.update(kwargs['processing_props'])
        img_height, img_width = self.img().shape[:2]
        radius = kwargs.get('radius', np.int(img_width / 4))
        center_pt = kwargs.get('center_pt', [np.int(img_height / 2), np.int(img_width / 2)])
        func_type = kwargs.get('function_type', 'circle')

        func_shape = (img_height, img_width)
        phi = np.zeros(func_shape)

        regularization = kwargs.get('regularization_note', 0)

        if func_type == 'circle':

            phi = LevelSetFunction.dist_from_circle(center_pt, radius, func_shape,
                                                    resolution=processing_props['resolution'])

        elif func_type == 'ellipse':
            phi = LevelSetFunction.dist_from_ellipse(center_pt, radius, func_shape,
                                                     resolution=processing_props['resolution'])
        elif func_type == 'egg_crate':
            phi = LevelSetFunction.dist_from_eggcrate(amplitude=kwargs['amplitude'], func_shape=func_shape, resolution=kwargs['resolution'])

        elif func_type == 'periodic_circles':
            phi = LevelSetFunction.dist_from_circles(kwargs['dx'], kwargs['dy'], radius, func_shape=func_shape)

        elif func_type == 'checkerboard':
            import skimage.segmentation as seg
            phi = seg.checkerboard_level_set(func_shape, radius)

        # scale between [-1, 1], while keeping the level set unmoved

        phi[phi >= 0] = scale_values(phi[phi >= 0], 0., 1.)
        phi[phi <= 0] = scale_values(phi[phi <= 0], -1., 0.)

        if np.all(self.__Phi[0].value == 0):
            self.__Phi = []

        self.__Phi.append(
            LevelSetFunction(phi, regularization_note=regularization,
                             epsilon=eps,
                             **processing_props))
    # --------------- SEMI-PROPERTIES ---------------------
    def img(self, index=0):
        """
        The analyzed image (of the data). Multiple can exist.

        :param index: the index of the analyzed images.

        :type index: int

        :rtype: np.ndarray

        """
        return self.__img[index]

    def phi(self, index=0):
        """
        Returns the level set function phi according to the index

        :param index: the number of the level set

        :type index: int

        :return: LevelSetFunction self.__Phi
        """

        return self.__Phi[index]

    def psi(self, index=0):
        """
        Returns the level set function phi according to the index

        :return: LevelSetFunction self.__psi

        .. warning::
            Not used anywhere
        """

        return self.__psi[index]
    # ------------------------------- PROPERTIES -------------------------------------
    @property
    def num_ls(self):
        """
        Number of level sets in the flow

        :return: int

        """
        return len(self.__Phi)

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

        Default: 0.

        :rtype: float

        """
        return self.__gvf_w

    @property
    def vo_w(self):
        """
        Open velocity flow weight

        Default: 0.

        :rtype: float

        """
        return self.__vo_w

    @property
    def region_w(self):
        """
        Region weight

        Default: 0.

        :rtype: float

        """
        return self.__region_w

    @property
    def ms_w(self):
        """
        Mumford-Shah (Chan Vese) weight

        Default 0, if mumford_shah flag is True

        :return: float

        """
        return self.__ms_w

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
        return self.__g_x * self.scale_x

    @property
    def g_y(self):
        """
        First order derivative of the internal force

        :rtype: np.array
        """
        return self.__g_y * self.scale_y

    @property
    def GVF(self):
        """
        External force (for gradient vector flow)

        :rtype: np.array
        """
        return self.__GVF

    # ---------------------- FLOWS ------------------------------
    def flow(self, flow_type, function, *args, **kwargs):
        r"""
        Return the flow of the level set according to the type wanted

        :param flow_type: can be one of the following:
            - 'constant':

               .. math::

                   C_t = N \Rightarrow \phi_t = |\nabla \varphi|

            - 'curvature':

               .. math::

                    C_t = kN \Rightarrow phi_t = div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)|\nabla \varphi|

            - 'equi_affine':

                .. math::

                    C_t = k^{1/3} N \Rightarrow
                    \phi_t = div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)^{1/3}\cdot|\nabla \varphi|

            - 'geodesic': geodesic active contours, according to :cite:`Caselles.etal1997`

                .. math::

                    C_t = (g(I)(c+\kappa) -\nabla(g(I))N)N \Rightarrow \\
                    \phi_t = g(I)\cdot div\left(\frac{\nabla \varphi}{|\nabla \varphi|}\right)\cdot|\nabla \varphi| + g(I)|\nabla \varphi|

            - 'band': band velocity, according to :cite:`Li.etal2006`

        :param function: the level set according to which the flow goes (usually phi)
        :param open_flag: boolean for open flag


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
            flow = self.g*function.kappa * function.norm_nabla

        if flow_type == 'equi_affine':
            flow = np.cbrt(function.kappa) * function.norm_nabla

        if flow_type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curvature flow !!

            while np.any(function.norm_nabla > 1e10):
                new_value = function.value
                new_value[function.norm_nabla > 1e10] = np.sign(new_value[function.norm_nabla > 1e10]) * 1
                function.update(new_value)

            # function.norm_nabla[np.abs(function.norm_nabla)<1e-4] = 0
            # function._x[np.abs(function.norm_nabla)<1e-4]=0
            # function._y[np.abs(function.norm_nabla)<1e-4]=0

            flow = self.g * function.kappa * function.norm_nabla + (self.g_x * function._x + self.g_y *
                                                                    function._y)
            # if open_flag:
            #     psi_t = self.g * self.function.kappa * self.psi.norm_nabla + (
            #             self.g_x * self.psi._x + self.g_y * self.psi._y)
            #     self.psi.update(self.psi.value - psi_t, regularization_note=processing_props['regularization'])

        if flow_type == 'band':
            vb = self.__compute_vb(**processing_props)
            flow = vb * self.phi().norm_nabla
            psi_t = vb * self.psi().norm_nabla
            self.psi().update(self.psi().value + psi_t, **processing_props)

        return cv2.GaussianBlur(flow, (processing_props['ksize'], processing_props['ksize']), processing_props['sigma'])

    def mumfordshah_flow(self,  mu=1, nu=0, img=True,):
        """
        Computes the Mumford-Shah flow for a multi-phase level set, according to :cite:`Vese.Chan2002`.
        Return the flow to move the level set.


        :param mu: the curvature weight. Default: 1
        :param nu: the length weight. Default: 0.
        :param img: the specific image upon which the level sets are computed. Uses the object images unless specified differently.

        :type mu: float
        :type nu: float
        :type img: np.array or bool

        :return: phi_t - the flows that move the level set
        """

        if img:
            images = self.__img
        else:
            images = img

        Phi = self.__Phi.copy()

        m_levelsets = self.num_ls
        n_phases = 2 ** m_levelsets
        _combinations = '01' * m_levelsets
        # fig, ax = plt.subplots(num = 'panorama')
        import itertools
        counter = 0

        kappa_flag = True
        dphi =  np.zeros((images[0].shape[0], images[0].shape[1], self.num_ls))

        for img in images:
            dphi_ = np.zeros((img.shape[0], img.shape[1], self.num_ls))

            if img.ndim == 3:
                image = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
                images.append(image[:, :, 0])
                images.append(image[:, :, 1])
                images.append(image[:, :, 2])
                continue

            combinations = itertools.combinations(_combinations, m_levelsets)
            for combination in combinations:
                dphi_ +=  self.__ms_element(combination, img)

            for i in range(m_levelsets):
                i = int(i)
                dphi_[:,:, i] += mu * self.phi(i).dirac_delta * self.phi(i).kappa
                dphi_[:,:, i] -= nu * self.phi(i).dirac_delta

                # Phi[i].move_function(dphi_[:, :, i])
                # counter += 1
                #
                # if counter > m_levelsets:
                #     kappa_flag = False
                #
                # if kappa_flag:
                #
            dphi += dphi_
        # self.__Phi = Phi
        return dphi

        # mt.draw_contours(self.phi(0).value, ax, self.img_rgb, color = 'b')
        # mt.draw_contours(self.phi(1).value, ax, self.img_rgb, hold = True, color = 'r')
        # plt.pause(.5e-10)

    # --------------------------- PRIVATE FUNCTIONS FOR FLOW PURPOSES -----------------------
    def __ms_element(self, combination, img):
        r"""
        An element for summation within the Mumford-Shah model.

        For example, for a 3-phase model, with a unison of the three level sets (all inside):

        .. math::

            \begin{eqnarray}
            \phi_{1t}\rightarrow(u_{0} - c_{000})^2 \delta(\phi_1)H(\phi_2)H(\phi_3) \\
            \phi_{2t}\rightarrow(u_{0} - c_{000})^2 H(\phi_1)\delta(\phi_2)H(\phi_3) \\
            \phi_{3t}\rightarrow(u_{0} - c_{000})^2 H(\phi_1)H(\phi_2)\delta(\phi_3)

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
        dPhi = np.zeros((img.shape[0], img.shape[1], self.num_ls))
        for index in range(self.num_ls):
            i = int(combination[index])
            if i == 0:  # inside the level set
                H.append(self.phi(index).heaviside)
                diracs.append(-self.phi(index).dirac_delta)
            else: #outside the level set
                H.append(1 - self.phi(index).heaviside)
                diracs.append(self.phi(index).dirac_delta)

        import functools
        mult = functools.reduce(lambda x, y: x * y, H)
        if np.all(mult == 0):
            c = 0
        else:
            c = np.sum(img * mult) / (np.sum(mult))
            # print('\n segment type {}, \t pixels average {}'.format(i,c))
        if np.isnan(c):
            c = 0
        for index in range(self.num_ls):
            i = int(index)
            H_ = H.copy()
            H_.pop(i)
            if len(H_) > 1:
                H_ = functools.reduce(lambda x, y: x * y, H_)
                mult_dirac.append(diracs[i] * H_)

            elif len(H_) == 0:
                mult_dirac = [diracs[i]]

            else:
                H_ = H_[0]
                mult_dirac.append(diracs[i] * H_)
            dPhi[:, :, i] += mt.make_zero((img - c) ** 2 * mult_dirac[i])

        return dPhi

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

        SR = np.zeros(self.img().shape[:2])
        SR_ = np.zeros(self.img().shape[:2])

        SR[R] = np.mean(self.img()[R])
        SR_[R_] = np.mean(self.img()[R_])
        vb = 1 - (SR_ - SR) / (np.linalg.norm(SR + SR_) + EPS)
        vb[vb < threshold] = 0
        vb *= tau
        #  vb[np.where(phi != 0)] = 0
        return vb * self.phi().kappa

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

    # ------------------------------ MAIN FUNCTION FOR MOVEMENT ------------------------

    def moveLS(self, **kwargs):
        r"""
        The function that moves the level set until the desired contour is reached

        :param flow_type -  flow types and their weight (string, weight):

        'constant', 'curvature', 'equi-affine', 'geodesic', 'band'

        :param movie_name: the name of the generated movie. Defualt: "Level set example"
        :param movie_folder: path to final generated movie. Default: project path
        :param mumford_shah: flag for mumford_shah flow
        :param nu: Chan-Vese (Mumford-Shah) weight for curve length. Default: 1
        :param mu: Chan-Vese  (Mumford-Shah) weight for curve area. Default: 0
        :param linewidth: line width for drawing. Default: 1.
        :param color_random: randomize colors or use one. Default: randomize (True)
        :param color: the color in which the contours will be drawn


        :type nu: float
        :type mu: float


        :return the contours after level set

         """
        from matplotlib import animation

        # ================================ MOVIE INITIALIZATIONS ===========================
        # Movie initializations
        movie_name = kwargs.get('movie_name', 'Level set example')
        movie_folder = kwargs.get('movie_folder', '')

        metadata = dict(title=movie_name, artist='Reuma',
                        comment='Movie support!')
        writer = animation.FFMpegFileWriter(fps=10, metadata=metadata)

        # ------inputs--------
        verbose = kwargs.get('verbose', False)
        color_random = kwargs.get('color_random', True)
        linewidth = kwargs.get('linewidth', 1)
        mumford_shah_flag = kwargs.get('mumford_shah', False)
        color = kwargs.get('color', 'b')
        open_flag = False
        mu = 1.
        nu=0

        blob_size = kwargs.get('blob_size', 0.1)
        if mumford_shah_flag:
            mu = kwargs.get('mu', 1.)
            nu = kwargs.get('nu', 0)
        if np.any(self.img_rgb) != 0:
            temp = self.img_rgb
        else:
            temp = self.img(0)

        img_showed = kwargs.get('image_showed', temp)

        # -------- initializations ---------
        flow_types = self.flow_types
        regularization_epsilon = self.regularization_epsilon
        iterations = self.iterations

        gvf_w = self.gvf_w
        vo_w = self.vo_w
        region_w = self.region_w
        processing_props = self.processing_props

        fig, ax = plt.subplots(num='img', figsize=(16,9))
        ax.axis('off')

        if np.any(self.img_rgb) != 0:
            mt.imshow(self.img_rgb, origin='lower', )
        else:
            mt.imshow(self.img(0), origin='lower')

        _, ax2 = plt.subplots(num='phi')
        mt.imshow(self.phi().value, origin='lower')
        ax2.axis('off')
        # fig3, ax3 = plt.subplots(num='kappa')
        # mt.imshow(self.phi().kappa)

        with writer.saving(fig, movie_folder + movie_name + ".mp4", 100):
            from tqdm import trange
            for iteration in trange(iterations, desc='Running level set'):
                # print(iteration)
                # if iteration ==100 :
                #     print('hello')
                dphi_intrinsic = np.zeros((self.img().shape[0],self.img().shape[1],  self.num_ls))
                extrinsic = np.zeros((self.img().shape[0],self.img().shape[1],  self.num_ls))
                dphi_r =  np.zeros((self.img().shape[0],self.img().shape[1],  self.num_ls))

                # ---------- intrinsic movement ----------

                # regular flows
                for item in list(flow_types.keys()):
                    if flow_types[item] == 0:
                        continue
                    for i in range(self.num_ls):
                      # intrinsic (GAC, curvature etc)
                      dphi_intrinsic[:,:,i] += flow_types[item] * self.flow(item, self.phi(i), open_flag, processing_props)
                      # region force
                dphi_intrinsic[:,:,0] += region_w * self.region * self.phi(0).norm_nabla

                # ---------------- mumford_shah movement --------
                if mumford_shah_flag:
                    dphi_ms = self.ms_w* self.mumfordshah_flow(mu=mu, nu=nu)
                else:
                    dphi_ms = np.zeros(dphi_intrinsic.shape)

                intrinsic = np.zeros(dphi_intrinsic.shape)

                # maximal/minimal forces between positive dphi_intrinsic and dphi_r:
                if np.all(dphi_intrinsic == 0):
                    intrinsic = dphi_r
                else:
                    intrinsic[(dphi_intrinsic - dphi_r) > 0] = np.maximum(dphi_intrinsic[(dphi_intrinsic - dphi_r) > 0], dphi_r[(dphi_intrinsic - dphi_r) > 0])
                    intrinsic[(dphi_intrinsic - dphi_r) < 0] = np.minimum(dphi_intrinsic[(dphi_intrinsic - dphi_r) < 0], dphi_r[(dphi_intrinsic - dphi_r) < 0])
                # maximal/minimal forces between positive from what was found and mumford shah
                if np.all(intrinsic == 0):
                    intrinsic = dphi_ms
                else:
                    intrinsic[(intrinsic - dphi_ms) > 0] = np.maximum(intrinsic[(intrinsic - dphi_ms) > 0], dphi_ms[(intrinsic - dphi_ms) > 0])
                    intrinsic[(intrinsic - dphi_ms) < 0] = np.minimum(intrinsic[(intrinsic - dphi_ms) < 0], dphi_ms[(intrinsic - dphi_ms) < 0])

                # ---------------extrinsic movement ----------
                for k in range(self.num_ls):
                    extrinsic[:,:,k] = (self.GVF[:, :, 0] * self.phi(k)._x + self.GVF[:, :, 1] * self.phi(k)._y) * gvf_w

                    # for constrained contours
                    # extrinsic += self.__compute_vo() * vo_w
                    phi_t = self.step * (intrinsic - extrinsic)
                    # reinitializtion every 10 iterations:
                    if iteration % 10== 0 and iteration != 0:
                        self.phi(k).reinitialization(phi_t[:, :, k])
                        # plt.figure('3d')
                        # ax4 = plt.axes(projection='3d')
                        # ax4.view_init(45,65)
                        # X, Y = np.meshgrid(np.arange(0, self.phi().value.shape[1]), np.arange(0, self.phi().value.shape[0]))
                        # ax4.plot_surface(X, Y, self.phi().value, cmap='gray')

                    else:
                        self.phi(k).move_function(phi_t[:, :, k])

                plt.figure('phi')
                mt.imshow(np.flipud(self.phi().value))
                # plt.pause(.5e-10)

                plt.figure('img')

                for i in range(len(self.__Phi)):
                    if i > 0:
                        l_curve, ax = mt.draw_contours(self.phi(i).value, ax, img=img_showed, hold=True,
                                                       color='r', linewidth=linewidth, blob_size=blob_size)
                    else:
                        l_curve, ax = mt.draw_contours(self.phi(i).value, ax, img=img_showed, hold=False,
                                                       color_random=color_random, linewidth=linewidth, blob_size=blob_size, color=color)
                title = ax.text(0, 1.07, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                transform=ax.transAxes, ha="center")
                title.set_text('Iteration #: {}'.format(iteration))
                ax.axis('off')
                            # title(')
                writer.grab_frame()
        plt.pause(.0001)
        # plt.show()
        plt.savefig(movie_folder + movie_name + '_final.png', figsize=(16,9),  bbox_inches='tight', dpi=300 )
        print('Done with level sets')
        return l_curve
