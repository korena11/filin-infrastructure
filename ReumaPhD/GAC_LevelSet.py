'''
infraGit
photo-lab-3\Reuma
27, Feb, 2017 
'''

import platform


if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')


from ContourDevelopment import getValueSubpix
from numpy.linalg import norm
from numpy import sin, cos, pi
from matplotlib import pyplot as plt
import MyTools
from MyTools import imshow
from functools import partial
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
        if prop.area >= minArea:
            contours_filtered.append(c)

    return contours_filtered


if __name__ == '__main__':
    # --- initializations
    img = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/Dropbox/PhD/Data/ActiveContours/Images/doubleTrouble.png', 1),
                       cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point

    beta = 1
    stepsize = .05
    alpha = 1.e-5
    # define an initial contour via phi(x,y) = 0
    phi = np.ones(img.shape)
    img_height, img_width = img.shape
    width, height = 20, 20
    phi[img_height/2-height : img_height/2 + height, img_width/2 - width: img_width/2 + width] = -1
    phi_binary = phi.copy()

    MyTools.imshow(img)

    # ---- define g(I) ---
    imgGradient = MyTools.computeImageGradient(img, gradientType='L2', ksize=5, sigma=7.5)
    g = 1/(1+imgGradient)

    g_x, g_y = MyTools.computeImageDerivatives(g, 1, ksize=3)
    # and its derivatives
    # g_x = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
    # g_y = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
    # g_x = cv2.GaussianBlur(g_x, (0, 0), 5.5)
    # g_y = cv2.GaussianBlur(g_y, (0, 0), 2.5)
    g = cv2.GaussianBlur(g, (0, 0), 2.5)

    epsilon = 100
    while epsilon > 15:
    #for i in range(0,1500):

        # computing phi derivatives
        phi_x, phi_y, phi_xx, phi_yy, phi_xy = MyTools.computeImageDerivatives(phi, 2)

        # div(phi/|phi|^2)
        div_phi =  beta *((phi_xx * phi_y**2 + phi_yy * phi_x**2 - 2 * phi_xy * phi_x * phi_y) / (1+phi_x**2 + phi_y**2)**3) + alpha

        # dot product of \nabla g and \nabla phi
        grad_g_dot_grad_phi = g_x * phi_x + g_y * phi_y

        phi_t = g * div_phi + grad_g_dot_grad_phi
        phi += cv2.GaussianBlur(phi_t, (0,0), 2.5) * stepsize

        epsilon = norm(phi_t)
        print epsilon
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

        MyTools.imshow(img)

        for c in contours:
            c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
            plt.plot(c[:, 0], c[:, 1], '-r')

        plt.pause(.5e-10)

    plt.show()

    'hello'