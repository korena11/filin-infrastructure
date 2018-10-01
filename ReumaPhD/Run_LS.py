'''
infragit
reuma\Reuma
06, Apr, 2018 
'''

# import matplotlib
#
# matplotlib.use('TkAgg')

import cv2
from matplotlib import pyplot as plt

import MyTools as mt
from LevelSets.LevelSetFlow import LevelSetFlow

if __name__ == '__main__':
    # initial input:
    img_orig = cv2.cvtColor(cv2.imread(r'D:\ownCloud\Data\Images\channel91.png'), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.normalize(img_gray.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    sigma = 2.5  # blurring
    # ----------- Define properties for the Level Set Factory -----------
    ls_obj = LevelSetFlow(img_gray, img_rgb=img_orig, step=.1)

    processing_props = {'sigma': 2.5, 'ksize': 5, 'gradientType': 'L2', 'regularization': 0}
    ls_obj.set = processing_props

    ls_obj.set_flow_types(geodesic=1.0)

    ls_obj.set_weights(gvf_w=1, region_w=.05)

    # ls_obj.band_props = {'band_width': 5e-4, 'threshold': 0.1, 'stepsize': 0.5}

    # ---------------------------------------------------------------------

    ls_obj.init_phi(radius=100, regularization_note=0, processing_props=processing_props, center_pt=(80, 160))
    ls_obj.init_phi(radius=100, regularization_note=0, processing_props=processing_props, center_pt=(60, 100))

    fig, ax = plt.subplots(num=1)
    mt.draw_contours(ls_obj.phi(0).value, ax, img_gray, color='b')
    mt.draw_contours(ls_obj.phi(1).value, ax, img_gray, hold=True, color='r')
    plt.show()

    # ------- Initial limits via psi(x,y) = 0 ---------------------------
    # option 1: horizontal line
    # psi[img_height - 2 * height: img_height - height, :] = -1
    # option 2: vertical line
    # psi = -np.ones(img_orig.shape[:2])
    # width_boundary = 10
    # psi[:, 0: width_boundary] = 1
    # psi[:, -width_boundary:] = 1
    # # psi[0:width_boundary, : ] = -1
    # # psi[-width_boundary:, :] = -1
    # ls_obj.init_psi(cv2.GaussianBlur(psi, (3, 3), 2.5), **processing_props)
    # mt.imshow(psi)
    # plt.show()
    # ---------------------------------------------------------------------

    # ---------------- Intrinsic forces maps ------------------------------

    # Force I - function g:
    # can be either constant, weights, or function
    # option 1: edge map g = 1/(1+|\nabla G(I) * I|).

    imgGradient = mt.computeImageGradient(img_gray, gradientType='L2', sigma=2.5)
    g = 1 / (1 + imgGradient ** 2)
    plt.figure('g')
    mt.imshow(g)

    # option 2: saliency map
    # g = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='pixel_val')
    ls_obj.init_g(g, **processing_props)

    # Force II - region constraint:
    ls_obj.init_region('saliency', saliency_method='context', sigma=0.5, feature='normals')
    plt.figure('region')
    mt.imshow(ls_obj.region)

    # Force III - open contours:

    # ---------------------------------------------------------------------

    # ----------------Extrinsic forces maps (vector field)------------------

    # The map which the GVF will be defined by
    # option 1: the image itself
    f = 1 - cv2.GaussianBlur(img_gray, ksize=(9, 9), sigmaX=sigma)

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

    ls_obj.moveLS(open_flag=False, verbose=False, mumford_shah=False)
