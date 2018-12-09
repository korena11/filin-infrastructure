'''
infragit
reuma\Reuma
09, Aug, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from IOmodules.IOFactory import IOFactory
from PanoramaFactory import PanoramaFactory
from LevelSetFlow import LevelSetFlow
from ColorProperty import ColorProperty
import MyTools as mt
import cv2

if __name__ == '__main__':
    colors = []
    pts = []
    pcl = IOFactory.ReadPts(r'D:\OwnCloud\Data\PCLs\bulbus.pts', pts, colors, merge=False)

    _panorama = PanoramaFactory.CreatePanorama_byPoints(pcl[0], azimuthSpacing=0.045, elevationSpacing=0.028,
                                                        voidData=23.7, intensity=False)
    _subset = _panorama.extract_area((20, 25), (100, 140))

    _panorama = PanoramaFactory.CreatePanorama_byPoints(_subset, azimuthSpacing=0.045, elevationSpacing=0.028,
                                                        voidData=22.5, intensity=False)
    _panorama_intensity = PanoramaFactory.CreatePanorama_byPoints(_subset, azimuthSpacing=0.045,
                                                                  elevationSpacing=0.028,
                                                                  voidData=22.5, intensity=True)
    n, gradient, xyz, filtered = _panorama.normals()

    color = ColorProperty(_subset, colors[0].RGB[_subset.GetIndices])
    _color_panorama = PanoramaFactory.CreatePanorama_byProperty(color,
                                                                azimuthSpacing=0.045, elevationSpacing=0.028,
                                                                voidData=23.7)

    img_orig = _panorama.PanoramaImage
    img_norm = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    img_rgb = cv2.normalize(_color_panorama.PanoramaImage.astype('float'), None, 0.0, 1.0,
                            cv2.NORM_MINMAX)  # Convert to normalized floating point
    img_intensity = cv2.normalize(_panorama_intensity.PanoramaImage.astype('float'), None, 0.0, 1.0,
                                  cv2.NORM_MINMAX)  # Convert to normalized floating point
    fig, ax = plt.subplots(num='panorama')
    plt.title('panorama')
    plt.imshow(img_orig)
    imgs = [img_norm, img_rgb[:, :, 0], img_intensity]
    processing_props = {'sigma': 2.5, 'ksize': 3, 'gradientType': 'L2'}

    ls_obj = LevelSetFlow(imgs, img_rgb=img_rgb, step=1., iterations=150,
                          flow_types={'geodesic': .0, 'curvature': 0.0})

    ls_obj.set_weights(gvf_w=1., vo_w=0., region_w=.05)

    ls_obj.init_phi(radius=80, processing_props=processing_props, regularization_note=0, center_pt=(30, 86))
    ls_obj.init_phi(radius=80, processing_props=processing_props, regularization_note=0, center_pt=(50, 60))

    mt.draw_contours(ls_obj.phi(0).value, ax, img_rgb, color='b')
    mt.draw_contours(ls_obj.phi(1).value, ax, img_rgb, hold=True, color='r')
    # plt.show()
    plt.figure()
    mt.imshow(ls_obj.phi().value)
    mt.imshow(ls_obj.phi(1).value)

    # plt.show()

    # ---------------- Intrinsic forces maps ------------------------------

    # Force I - function g:
    # can be either constant, weights, or function
    # option 1: edge map g = 1/(1+|\nabla G(I) * I|).

    imgGradient = mt.computeImageGradient(img_norm, gradientType='L1', sigma=0.5)
    g = 1 / (1 + imgGradient ** 2)
    plt.title('gradients')
    plt.imshow(g)
    plt.show()

    # option 2: saliency map
    #    g = sl.distance_based(img_orig, filter_sigma = [sigma, 1.6*sigma, 1.6*2*sigma, 1.6*3*sigma], feature='normals')
    ls_obj.init_g(g, **processing_props)

    # Force II - region constraint:
    tensor_property = Ten
    ls_obj.init_region('saliency', feature='normals', saliency_method='context', filter_size=3, normals=n)
    # ls_obj.update_region(g)
    ls_obj.update_region(ls_obj.region)

    plt.title('region')
    plt.imshow(ls_obj.region)
    plt.show()

    # ---------------------------------------------------------------------

    # ----------------Extrinsic forces maps (vector field)------------------

    # The map which the GVF will be defined by
    # option 1: the image itself
    f = 1 - cv2.GaussianBlur(img_norm, ksize=(5, 5), sigmaX=1.5)

    # # option 2: edge map
    # f = cv2.Canny(img_orig, 10, 50)
    #  f = cv2.GaussianBlur(f, ksize = (5, 5), sigmaX = 1.5)

    # # option 3: saliency map
    # f = cv2.GaussianBlur(region, ksize = (5, 5), sigmaX = sigma)

    #  f = np.zeros(img_gray.shape)
    ls_obj.init_f(f, **processing_props)

    ls_obj.moveLS(open_flag=False, image_showed=img_rgb, mumford_shah=True)

    # print ("hello")
