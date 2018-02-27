'''
infragit
reuma\Reuma
11, Feb, 2018 
'''

# import platform
#
# if platform.system() == 'Linux':
#     import matplotlib
#     matplotlib.use('TkAgg')

import cv2
import numpy as np
from matplotlib import pyplot as plt

import MyTools as mt
from LevelSetFactory import LevelSetFactory

# Classification codes
RIDGE = 1
PIT = 2
VALLEY = 3
FLAT = 4
PEAK = 5
SADDLE = 6

if __name__ == '__main__':
    # initial input:
    # raster = IOFactory.rasterFromAscFile(r'D:\Documents\ownCloud\Data\Channel91.txt')
    img_orig = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\Image.bmp'), cv2.COLOR_BGR2RGB)
    # raster.roughness = 0.5
    # img_orig = raster.data
    img_normed = img_orig.copy()
    img_normed[img_normed == -9999] = np.mean(np.mean(img_normed[img_normed != -9999]))
    img_normed = cv2.normalize(img_normed.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point
    sigma = 2.5  # blurring
    # hillshade = rv.hillshade(img_orig)
    ls_obj = LevelSetFactory(img_normed, img_rgb = img_orig, step = 2.)

    processing_props = {'sigma': 5, 'ksize': 5, 'gradientType': 'L2'}
    ls_obj.init_phi(width = 80, height = 80, start = (10, 10))

    plt.figure()
    mt.imshow(ls_obj.phi)
    plt.show()

    #  phi *= dists
    # option 2: horizontal line
    # phi[img_height - 2 * height: img_height - height, :] = -1

    # option 3: vertical line
    # phi[:, img_width/2 : img_width/2 + width] = -1


    # ------- Initial limits via psi(x,y) = 0 ---------------------------
    # option 1: horizontal line
    # psi[img_height - 2 * height: img_height - height, :] = -1
    # option 2: vertical line
    psi = np.ones(img_orig.shape[:2])
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
    g = 1 / (1 + imgGradient ** 2)
    plt.imshow(g)

    # option 2: saliency map
    #    g = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='normals')
    ls_obj.init_g(g, **processing_props)

    # Force II - region constraint:
    ls_obj.init_region('saliency')
    plt.imshow(ls_obj.region)

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

    ls_obj.moveLS(open_flag = False, processing_props = processing_props, iterations = 500,
                  gvf_w = 1.,
                  vo_w = 0.,
                  region_w = 0.)
