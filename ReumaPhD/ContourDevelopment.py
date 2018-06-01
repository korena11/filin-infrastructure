'''
infraGit
photo - lab - 3
15, Jan, 2017
'''

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.linalg import norm
from skimage import measure
from functools import partial


def getValueSubpix(img, location_row, location_column):
    """

    :param img: the image to subsample (grayscale, 8bit)
    :param location_row: the row location where we want to subsample
    :param location_column: the column location where we want to subsample
    :param kind: interpolation type. Default: bilinear

    :return: the image subpixel value
    """
    # TODO: add other interpolations
    i = np.int(round(location_row))
    j = np.int(round(location_column))

    # pad image for edge cases
    newImage = img

    if i < 0:
        newImage = np.ones((img.shape[0] + np.abs(i), img.shape[1])) * 1000
        newImage[np.abs(i):, :] = img
        i = 0

    if j < 0:
        newImage = np.ones((img.shape[0], img.shape[1]+ np.abs(j))) * 1000
        newImage[:, np.abs(j):] = img
        j = 0

    if i >= img.shape[0]-1:
        add = (i - img.shape[0]) + 2
        newImage = np.ones((img.shape[0] + add, img.shape[1])) * -1000
        newImage[:-add, :] = img

    if j >= img.shape[1]-1:
        add = (j - img.shape[1]) + 2
        newImage = np.ones((img.shape[0], img.shape[1]  + add)) * -1000
        newImage[:, :-add] = img

    # Bilinear interpolation: [1-a a] [[img[i,j] img[i, j+1],[img[i+1,j], img[i+1,j+1]] [[1-b][b]]

    a = location_row - i
    b = location_column - j

    a = np.array([1-a, a])
    b = np.array([1-b,b])

    imgVals = newImage[i:i+2, j:j+2]

    return a.dot(imgVals).dot(b.T)




if __name__ == '__main__':

    w = 70
    winSize = (5, 5)
    deltaHeight = 0.2 # m

    # # Construct some test data
    # gridSpacing = pi
    # x, y = np.ogrid[-pi:gridSpacing:100j, -pi:gridSpacing:100j] # the '100j' is the number of elements
    # img = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))
    img = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\doubleTrouble.png', 1), cv2.COLOR_BGR2GRAY)

    # compute image gradient
    dimg_dx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
    dimg_dy = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)
    grad_img = np.abs(dimg_dx) + np.abs(dimg_dy)

    fig = plt.figure()
    plt.imshow(grad_img, cmap='gray')


    # --- Define a contour ---
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.5)

   # plt.figure()
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.hold(True)
    dh = 2



    for c in contours:
        for k in np.arange(0, 10):
            c[:, [0, 1]] = c[:, [1, 0]] #swapping between columns: x will be at the first column and y on the second

            if np.any(c < 0) or np.any(c[:,0] > grad_img.shape[1]) or np.any(c[:,1] > grad_img.shape[0]):
                break

            plt.plot(c[:,0], c[:,1], '-r')



            # contour tangent

            dc_dtau = np.gradient(c, axis=0)
            dc_dtau = dc_dtau / norm(dc_dtau, axis=1)[:, None] #normalize T


            # curve normal
            normal_c = dc_dtau.dot(np.array([[0, 1], [-1, 0]]))

            # image gradient at contour points
            grad_img_c = np.asarray(list(map(partial(getValueSubpix,grad_img), c[:,0], c[:,1])))

            #contour movement
            dc_dh = (dh * 1/grad_img_c[:,None]) * normal_c
            c+=dc_dh
            plt.plot(c[:, 0], c[:, 1])
            plt.hold(True)

            plt.pause(0.1)
            plt.draw()
            c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second

    plt.show()


