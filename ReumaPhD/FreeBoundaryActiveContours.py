'''
infraGit
photo-lab-3\Reuma
30, Mar, 2017

'''
from functools import partial

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.ndimage import filters

from . import ActiveContours as ac
from .ContourDevelopment import getValueSubpix

matplotlib.use('TkAgg')


def closestBoundaryPoint(point, boundary_curve, **kwargs):
    '''
    Finds the closest point in boundary_curve to a given point
    :param point: point [x y] (1x2) numpy array
    :param boundary_curve: the boundary curve [x y] (n x 2) numpy array
    :param kwargs:

    :return: the closest point in the boundary curve to the given point np.ndarray(2,1)
    '''
    # TODO: threshold when to resample the boundary for better "guess"

    dist2 = np.sum(np.power(boundary_curve - point,2), axis=1)

    return boundary_curve[np.argmin(dist2), :]



if __name__ == '__main__':
    # --- initializations
    img = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\channel91.png', 1),
                       cv2.COLOR_BGR2GRAY)

    # tension and stiffness
    alpha = 50
    beta = 0
    w_line = 0.4
    w_edge = 0.5
    w_term = 0.6

    # space and time steps
    dt = 0.01
    ds = 5.
    ds2 = ds ** 2

    # other constants
    a = alpha * dt / ds2
    p = b = beta * dt / ds2 ** 2
    q = -(a + 4 * b)
    r = 1 + 2 * a + 6 * b

    # define an initial contour - a line
    line_length = norm(img.shape)
    s = np.arange(0, line_length, ds)

    x_tau = 0 + s / line_length * (img.shape[1])
    y_tau = 120 - s/line_length * (img.shape[0]-70)
    c = np.vstack((x_tau, y_tau)).T

    # initial boundary contours (BC)
    b_length = img.shape[0]
    s = np.arange(0, b_length,ds)
    b0_x = np.ones((s.shape[0], 1)) * 1
    b0_y = s.copy()

    b0 = np.hstack((b0_x, b0_y[:,None]))

    b1_x = np.ones((s.shape[0], 1)) * img.shape[1]-2
    b1_y = s.copy()
    b1 = np.hstack((b1_x, b1_y[:,None]))

    # ================================ MOVIE INITIALIZATIONS ===========================
    # Movie initializations
    # FFMpegWriter = manimation.writers['ffmpeg']
    # metadata = dict(title='Open boundary active contour', artist='Reuma',
    #                 comment='Movie support!')
    # writer = FFMpegWriter(fps=30, metadata=metadata)
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.axis('off')
    plt.plot(b0_x, b0_y, '-r')
    plt.plot(b1_x, b1_y, '-r')
    l_curve, = plt.plot(x_tau,y_tau, 'm', linewidth=2.5)

    # generate image
    energy_image = ac.generateImageEnergy(img, w_line, w_edge, w_term, edgeEnergyType='value', blur_ksize=13, sobel_ksize=5)
    energy_image = cv2.blur(energy_image, (11, 11))
    energy_image_x = filters.sobel(energy_image, 1)
    energy_image_y = filters.sobel(energy_image, 0)

    energy_image_x = cv2.blur(energy_image_x, (7, 7))
    energy_image_y = cv2.blur(energy_image_y, (7, 7))

    M = ac.returnPentadiagonal(x_tau.shape[0], a,b, open=True)
    eyeM = np.eye(M.shape[0])
    invM = np.linalg.inv(M)

    # with writer.saving(fig, "freeBoundary.mp4", 100):
    for i in range(500):
        # external force - using energy_image
        fx = np.array(list(map(partial(getValueSubpix, energy_image_x), c[:, 1], c[:, 0])))
        fy = np.array(list(map(partial(getValueSubpix, energy_image_y), c[:, 1], c[:, 0])))

        # internal force - solving the (I-timeStep * A) = x, +timeStep * fx,y
        # contour movement
        ct_1 = c.copy()
        c[:, 0] = invM.dot(c[:, 0] + dt * fx)
        c[:, 1] = invM.dot(c[:, 1] + dt * fy)

        # replace at the beginning and end of the curve with the closest boundary point
        b0_closest = closestBoundaryPoint(c[1, :], b0)
        b1_closest = closestBoundaryPoint(c[-2, :], b1)

        c[0, :] = b0_closest
        c[-1, :] = b1_closest

        l_curve.set_data(c[:, 0], c[:, 1])

        # writer.grab_frame()
        plt.show()

        print ('done')
