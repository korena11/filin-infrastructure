import numpy as np
import Properties.Panoramas.PanoramaUtils as pu
import open3d as o3d
from DataClasses.PointSet import PointSet
from DataClasses.PointSetOpen3D import PointSetOpen3D
from Properties.Panoramas.PanoramaFactory import PanoramaFactory, PanoramaProperty
from Utils import MyTools as mt

from IOmodules.IOFactory import IOFactory
from VisualizationClasses.VisualizationO3D import VisualizationO3D
from matplotlib import pyplot as plt

def create_scanned_sphere(radius=1, az_res=1, elev_res=1):
    """
    Creates a point cloud of a sphere as if scanned from (0,0,0)

    :param radius: the radius of the sphere (meters)
    :param az_res: the azimuth scanning resolution (degrees). default 1 deg
    :param elev_res: the elevation scanning resolution (degrees). default 1 deg

    :type radius: float
    :type az_res: float
    :type elev_res: float

    :return: the point cloud
    """
    theta = np.arange(0, np.pi/4, np.deg2rad(az_res))
    phi = np.arange(0, np.pi/2,  np.deg2rad(elev_res))

    tt, pp = np.meshgrid(theta, phi)

    radius = radius_function(tt, pp)

    x = (radius * np.cos(pp) * np.cos(tt)).flatten()
    y = (radius * np.cos(pp) * np.sin(tt)).flatten()
    z = (radius * np.sin(pp)).flatten()

    xyz = np.array([x,y,z])
    return PointSet(xyz.T)

def radius_function(theta, phi):
    """
    A function so that the radius is changing as a function of theta and phi

    :param theta:
    :param phi:

    ..math::
        \theta+\phi/2

    :return:
    """
    return theta + phi + 1

if __name__ == '__main__':


    az_res = 1
    elev_res =  1
    pts = create_scanned_sphere(1, az_res, elev_res)
    # pts = IOFactory.ReadPts('../test_data/bulbus_100k.pts',merge=True)
    vis1 = VisualizationO3D()
    vis1.visualize_pointset(pts)

    panorama = PanoramaFactory.CreatePanorama_byPoints(pts, azimuthSpacing=az_res+0.001, elevationSpacing=elev_res+0.001, voidData=25)

    plt.imshow(panorama.PanoramaImage)
    plt.show()
    pano = panorama.PanoramaImage
    # pano, mean_sigma, mean_kernel = pu.adaptive_smoothing(panorama, .1)
    # print('mean sigma {} mean kernel {}'.format(mean_sigma, mean_kernel))
    r_t, r_p, r_tt, r_pp, r_tp = mt.computeImageDerivatives_numeric(pano, 2, resolution=az_res, ksize=15, sigma=0)



    r_t = r_t[panorama.row_indexes, panorama.column_indexes]
    r_p = r_p[panorama.row_indexes, panorama.column_indexes]
    r_tt = r_tt[panorama.row_indexes, panorama.column_indexes]
    r_pp = r_pp[panorama.row_indexes, panorama.column_indexes]
    r_tp = r_tp[panorama.row_indexes, panorama.column_indexes]

    # rs_t, rs_p, rs_tt, rs_pp, rs_tp = mt.computeImageDerivatives(pano_smoothed, 2, sigma=0)

    cos_theta = np.cos(np.radians(panorama.sphericalCoordinates.azimuths))
    sin_theta = np.sin(np.radians(panorama.sphericalCoordinates.azimuths))
    cos_phi = np.cos(np.radians(panorama.sphericalCoordinates.elevations))
    sin_phi = np.sin(np.radians(panorama.sphericalCoordinates.elevations))

    r = panorama.sphericalCoordinates.ranges
    # define the surface
    s_theta = np.vstack((r_t * cos_phi * cos_theta - r * cos_phi * sin_theta,
                        r_t * cos_phi * sin_theta + r * cos_phi * cos_theta,
                        r_t * sin_phi))
    s_phi= np.vstack((r_p * cos_phi * cos_theta - r * sin_phi * cos_theta,
                      r_p * cos_phi * sin_theta - r * sin_phi * sin_theta,
                      r_p * sin_phi + r * cos_phi))

    st_st = np.einsum('ji,ji->i', s_theta, s_theta)
    sp_sp = np.einsum('ji,ji->i', s_phi, s_phi)
    st_sp = np.einsum('ji,ji->i', s_theta, s_phi)

    normals = np.cross(s_phi, s_theta, axis=0).T
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    pts_o3d = PointSetOpen3D(pts)
    pts_o3d_1 = PointSetOpen3D(pts)
    pts_o3d.CalculateNormals(.5)
    dn = np.asarray(pts_o3d.data.normals) - normals
    print(r'differences in normals mean {} \pm std {}'.format(dn.mean(), dn.std()))
    pts_o3d_1.data.normals = o3d.Vector3dVector(normals)
    # pts_o3d.data.has_normals = True

    vis = VisualizationO3D()
    vis.visualize_pointset(pts_o3d_1)

    # define the FFM
    # G = np.array([[s_theta.dot(s_theta.T), s_theta.dot(s_phi.T)],
    #               [s_theta.dot(s_phi.T), s_phi.dot(s_phi.T)]])



    # plt.imshow(pano)
    # plt.show()
