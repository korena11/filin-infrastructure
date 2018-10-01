"""
infraGit
photo-lab-3\Reuma
06, Feb, 2017

 Geodesic active contours (Caselles, Kimmel and Sapiro, 1997), PART I
"""

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from .ContourDevelopment import getValueSubpix
from numpy.linalg import norm
from numpy import sin, cos, pi
from matplotlib import pyplot as plt
from . import MyTools as mt
from functools import partial
import numpy as np

import cv2

if __name__ == '__main__':
    # --- initializations
    img = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\doubleTrouble.png', 1),
                       cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point

    # img  = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/ownCloud/Data/Images/doubleTrouble.png'), cv2.COLOR_BGR2RGB)
    # sigma = 2.5
    # img = sl.distance_based(img, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma],
    #                    feature='normals')
    dt = 0.05
    alpha = 1e-3

    #    plt.imshow(img, interpolation='nearest', cmap='gray')

    # define an initial contour
    ds = 5.

    radius = 20
    s = np.arange(0, 2 * pi * radius, ds)
    tau = np.arange(0, 2 * pi, 0.05)

    # tau \in [0,1];  s (arclencth reparametrization) \in [0,2\pi*radius]
    x_tau = img.shape[1] / 2 + radius * cos(s / (radius))
    y_tau = img.shape[0] / 2 + radius * sin(s / (radius))

    # x_tau = img.shape[1] / 2 + radius * cos(tau)
    x_tau = np.append(x_tau, x_tau[0])

    # y_tau = img.shape[0] / 2 + radius * sin(tau)
    y_tau = np.append(y_tau, y_tau[0])
    c = np.vstack((y_tau, x_tau)).T

    # ---- define g(\nabla I) ---
    imgGradient = mt.computeImageGradient(img, gradientType='L1', ksize=3)
    #    imgGradient[np.isinf(imgGradient)] = 1e6
    g = - imgGradient
    g_x = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    g_y = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    g_x = cv2.GaussianBlur(g_x, (0, 0), 2.5)
    g_y = cv2.GaussianBlur(g_y, (0, 0), 2.5)
    g = cv2.GaussianBlur(g, (0, 0), .5)

    d_tmp = round(np.max(img.shape) / 64)
    #   g_x_ = g_x[::d_tmp, ::d_tmp]
    #   g_y_ = g_x[::d_tmp, ::d_tmp]

    # display
    plt.figure()
    plt.subplot(221), plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.subplot(222), plt.imshow(imgGradient, interpolation='nearest', cmap='gray')
    plt.subplot(223), plt.ylim([img.shape[0], 0]), plt.axis("equal"), plt.xlim([0, img.shape[1]]), plt.quiver(g_y, g_x)
    plt.subplot(224), plt.imshow(g, interpolation='nearest', cmap='gray')

    epsilon = 1000

    plt.plot(c[:, 1], c[:, 0], '-b')
    #  plt.show()
    plt.figure()
    # -------- Curve evolution -----------------
    while epsilon > 0.06:
        if np.any(c < 0) or np.any(c[:, 0] > img.shape[0]) or np.any(c[:, 1] > img.shape[1]):
            break

        c, dc_dtau, d2c_dtau2 = mt.curveCentralDerivatives(c, ds)
        #  curve tangent
        dc_dtau__ = np.gradient(c, ds, axis=0)
        tangent = dc_dtau / norm(dc_dtau, axis=1)[:, None]  # normalize T

        # curve curvature
        d2c_dtau2__ = np.gradient(dc_dtau, ds, axis=0)  # curve's second derivative

        #   curvature_c = norm(d2c_dtau2, axis=1) / norm(dc_dtau, axis=1)
        curvature_c = (dc_dtau[:, 0] * d2c_dtau2[:, 1] - dc_dtau[:, 1] * d2c_dtau2[:, 0]) / norm(dc_dtau, axis=1,
                                                                                                 ord=2) ** 3

        # curve normal
        normal_c = tangent.dot(np.array([[0, 1], [-1, 0]]))

        # g at contour points
        g_I = np.asarray(list(map(partial(getValueSubpix, g), c[:, 0], c[:, 1]))) + alpha

        # \nabla g at contour points with the direction of the curve's normal
        grad_gx = np.asarray(list(map(partial(getValueSubpix, g_x), c[:, 0], c[:, 1])))
        grad_gy = np.asarray(list(map(partial(getValueSubpix, g_y), c[:, 0], c[:, 1])))
        grad_g = np.vstack((grad_gx, grad_gy))
        grad_gN = np.sum(grad_g.T * normal_c, axis=1)

        # development scheme
        dc_dt = dt * (g_I * curvature_c - grad_gN)[:, None] * normal_c
        epsilon = norm(dc_dt)
        c += dc_dt

        print(epsilon)
        plt.cla()
        plt.axis("off")
        plt.ylim([img.shape[0], 0])
        plt.xlim([0, img.shape[1]])

        plt.imshow(g, cmap='gray')

        plt.plot(c[:, 1], c[:, 0], '-r.')
        plt.pause(1e-10)

    plt.show()
