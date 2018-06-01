'''
infraGit
photo-lab-3\Reuma
30, Jan, 2017 

'''
from functools import partial

import matplotlib

matplotlib.use('TkAgg')

import cv2
import matplotlib.mlab as mlab
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from numpy import sin, cos, pi
from scipy.ndimage import filters

from .ContourDevelopment import getValueSubpix
from .MyTools import imshow


def createGaussianImage(gridSpacing = .025, noise = 0.01):
    x = np.arange(-3.0, 3.0, gridSpacing)
    y = np.arange(-2.0, 2.0, gridSpacing)
    X, Y = np.meshgrid(x, y)
    img = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0., 0.0)
    gaussian = img.copy()

    img = img + cv2.randn(gaussian, 0, noise)


def returnPentadiagonal(n, a, b, **kwargs):
    '''
    M= (I-stepSize A)
    (alpha is the tension weight; beta is the stiffness weight)
    :param n: number of points
    :param a: alpha * timeStep/spaceStep^2
    :param b: beta * timeStep / spaceStep^4
    :param open: True or False. if the contour is open, compute forward and backward derivatives for the first and last row
    :param transversality: number of points the their forward and backward derivatives are computed for (default 1)

    :return: (I-stepSize * A)
    '''
    # trans = kwargs.get('transversality', 1)
    p = b
    q = -(a + 4 * b)
    r = 1 + 2 * a + 6 * b

    M = np.diag(np.tile(r, n))
    M += np.diag(np.tile(q, n - 1), 1) + np.diag(np.tile(q, 1), -n + 1)
    M += np.diag(np.tile(q, n - 1), -1) + np.diag(np.tile(q, 1), n - 1)
    M += np.diag(np.tile(p, n - 2), 2) + np.diag(np.tile(p, 2), -n + 2)
    M += np.diag(np.tile(p, n - 2), -2) + np.diag(np.tile(p, 2), n - 2)

    if kwargs.get('open'):
        zeros = np.zeros((n - 5, 1))
        m1 = np.array([1 - a + b, 2 * a - 4 * b, -a + 6 * b, -4 * b, b])
        m2 = np.append(m1, zeros)
        M[0, :] = m2
        M[-1, :] = np.flipud(m2)

        # for i in range(trans):
        #     m2 = np.insert(zeros, i, m1)
        #     M[i, :] = m2
        #     M[i-1, :] = np.flipud(m2)

    return M


def generateImageEnergy(I, w_line, w_edge, w_term, **kwargs):
    '''
    Compute an image of the image energy at each pixel
    :param I: image
    :param w_line: weights of the line term
    :param w_edge: weights of the edge term
    :param w_term: weights of the termination term

    :param edgeEnergyType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I), squared; 'LoG' Laplacian of gaussian
    :param sobel_ksize: kernel size for Sobel filter
    :param blur_ksize: kernel size for bluring
    :param laplace_sigma: guassian sigma in the LoG filter
    :return: energy image
    '''

    edgeEnergyType = kwargs.get('edgeEnergyType', 'L1')
    sobel_ksize = kwargs.get('sobel_ksize', 5)
    blur_ksize = kwargs.get('blur_ksize', 5)
    laplace_sigma = kwargs.get('laplace_sigma', 1.)

    img = cv2.blur(I, (blur_ksize, blur_ksize))

    # compute image gradient
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_ksize)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_ksize)

    grad_img = img

    if edgeEnergyType == 'L1':
        grad_img = np.abs(dx) + np.abs(dy)  # L1-norm of grad(I)
    elif edgeEnergyType == 'L2':
        grad_img = (dx ** 2 + dy ** 2)
    elif edgeEnergyType == 'LoG':
        grad_img = filters.gaussian_laplace(I, laplace_sigma)
    elif edgeEnergyType == 'value':
        grad_img = img

    # ---- Second derivatives ---
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize = sobel_ksize)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize = sobel_ksize)
    dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize = sobel_ksize)

    energy_line = I
    energy_edge = -grad_img
    energy_term = (dyy * dx ** 2 - 2 * dxy * dx * dy + dxx * dy ** 2) / ((1 + dx ** 2 + dy ** 2) ** (1.5))

    return w_line * energy_line + w_edge * energy_edge + w_term * energy_term


def GVF(energy, **kwargs):
    """
    Computes the gradient vector flow field given
    :param energy: and edge map of some kind

    ** optional
    :param mu: regularization parameter, tradeoff between the two parts of the equation
    :param dx, dy: spcaing between pixels, resolution
    :param dt: time-step. dt <= (dx*dy)/(4*mu)

    :return: vector field u(x,y), v(x,y)
    """
    dx = kwargs.get('dx', 1.)
    dy = kwargs.get('dy', 1.)
    mu = kwargs.get('mu', 1.)
    dt = kwargs.get('dt', dx * dy / (4. * mu))

    r = mu * dt / (dx * dy)
    # normalize f
    fmax = np.max(np.max(energy))
    fmin = np.min(np.min(energy))
    energy = (energy-fmin)/(fmax-fmin)

    fx, fy = np.gradient(-energy)
    v = fy.copy()
    u = fx.copy()

    b = fx ** 2 + fy ** 2
    b_1 = 1-b*dt
    c1 = b * fx
    c2 = b * fy

    # derivative computation
    epsilon = 500

    while epsilon > 3:
        del_u = filters.laplace(u)
        del_v = filters.laplace(v)

        ut = mu * 4 * del_u - b * (u - fx)
        vt = mu * 4 * del_v - b * (v - fy)

        u += ut
        v += vt

        epsilon = np.max((LA.norm(ut), LA.norm(vt)))

    return u, v


if __name__ == '__main__':

    # --- initializations
    img = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\twosink.png', 1),
                       cv2.COLOR_BGR2GRAY)

    plt.imshow(img, interpolation = 'nearest', cmap = 'gray')

    # tension and stiffness
    alpha = 50
    beta = 0
    w_line = 0.4
    w_edge = 0.5
    w_term = 0.6

    # space and time steps
    dt = 0.7
    ds = 10.
    ds2 = ds ** 2

    # other constants
    a = alpha * dt / ds2
    b = beta * dt / ds2 ** 2

    # define an initial contour
    radius = 1250
    s = np.arange(0, 2 * pi * radius, ds)

    # tau \in [0,1];  s (arclencth reparametrization) \in [0,2\pi*radius]
    x_tau = img.shape[0] / 2 + radius * cos(s / radius)
    #    x_tau = np.append(x_tau, x_tau[0])

    y_tau = img.shape[1] / 2 + radius * sin(s / radius)
    #   y_tau = np.append(y_tau, y_tau[0])
    c = np.vstack((y_tau, x_tau)).T

    energy_image = generateImageEnergy(img, w_line, w_edge, w_term, edgeEnergyType = 'value')
    energy_image = cv2.blur(energy_image, (7, 7))
    energy_image_y = filters.sobel(energy_image, 0)
    energy_image_x = filters.sobel(energy_image, 1)

    energy_image_x = cv2.blur(energy_image_x, (7, 7))
    energy_image_y = cv2.blur(energy_image_y, (7, 7))

    M = returnPentadiagonal(x_tau.shape[0], a, b)
    eyeM = np.eye(M.shape[0])
    invM = np.linalg.inv(M)
    # invM = np.linalg.inv(M + dt * eyeM)

    # ----------create GVF-----------

    e_external = -filters.gaussian_laplace(img, 1.5)

    energy_image_x, energy_image_y = GVF(e_external, mu = 2., dt=dt, image=img)


    # contours = measure.find_contours(img, 0.5)
    plt.plot(c[:, 0], c[:, 1], '-r')

    epsilon = 1000
    iternum = list(range(500))
    for i in iternum:
        # external force - using energy_image
        fx = np.array(list(map(partial(getValueSubpix, energy_image_x), c[:, 1], c[:, 0])))
        fy = np.array(list(map(partial(getValueSubpix, energy_image_y), c[:, 1], c[:, 0])))

        # internal force - solving the (I-timeStep * A) = x, +timeStep * fx,y
        # contour movement
        ct_1 = c.copy()
        c[:, 0] = invM.dot(c[:, 0] + dt * fx)
        c[:, 1] = invM.dot(c[:, 1] + dt * fy)

        epsilon = np.linalg.norm(c - ct_1)
        print(epsilon)
        plt.cla()
        plt.axis("off")
        plt.ylim([img.shape[0], 0])
        plt.xlim([0, img.shape[1]])

        imshow(img, cmap = 'gray')

        plt.plot(c[:, 0], c[:, 1], '-r.')
        plt.pause(1e-6)


        #   c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second

    plt.show()
