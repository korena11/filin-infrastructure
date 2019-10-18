import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from DataClasses.PointSet import PointSet
from DataClasses.PointSetOpen3D import PointSetOpen3D
from Properties.Panoramas.PanoramaFactory import PanoramaFactory
from Utils import MyTools as mt
from VisualizationClasses.VisualizationO3D import VisualizationO3D
from IOmodules.IOFactory import IOFactory

def create_scanned_wall(dist, az_res, elev_res):
    """
    Create a wall that was scanned

    :param dist: distance from the scanner
    :param az_res: the azimuth scanning resolution (degrees).
    :param elev_res: the elevation scanning resolution (degrees).

    :return:  the point cloud of a wall
    """
    from Properties.Transformations import RotationUtils as ru

    pts = create_scanned_floor(dist, az_res, elev_res)

    # make rectangle
    xx = pts.X[np.abs(pts.X) < dist * np.cos(np.pi / 4)]
    yy = pts.Y[np.abs(pts.X) < dist * np.cos(np.pi / 4)]
    zz = pts.Z[np.abs(pts.X) < dist * np.cos(np.pi / 4)]

    x = xx[np.abs(yy) < dist * np.sin(np.pi / 4)]
    y = yy[np.abs(yy) < dist * np.sin(np.pi / 4)]
    z = zz[np.abs(yy) < dist * np.sin(np.pi / 4)]

    xyz = np.vstack((x,y,z))
    R = ru.BuildRotationMatrix(90, 0, 90)

    rotated_pts = R.dot(xyz)


    return PointSet(rotated_pts.T)


def create_scanned_floor(radius, az_res, elev_res):
    """
    Create a floor that was scanned

    :param az_res: the azimuth scanning resolution (degrees). default 1 deg
    :param elev_res: the elevation scanning resolution (degrees). default 1 deg


    :return:  the point cloud of a floor
    """

    theta = np.arange(0, np.pi , np.deg2rad(az_res))
    phi = np.arange(0, np.pi / 2, np.deg2rad(elev_res))

    tt, pp = np.meshgrid(theta, phi)

    # radius = radius_function(tt, pp)

    x = (radius * np.cos(pp) * np.cos(tt)).flatten()
    y = (radius * np.cos(pp) * np.sin(tt)).flatten()
    z = np.ones(x.shape)


    xyz = np.array([x, y, z])
    return PointSet(xyz.T)

def create_scanned_sphere(radius=None, az_res=1, elev_res=1):
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
    if radius is None:
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

    az_res = 0.15
    elev_res =  .15
    pts = create_scanned_floor(1, az_res, elev_res)
    az_res += 0.001
    elev_res += 0.001
    # pts = IOFactory.ReadPts(r'C:\Users\reuma\Documents\ownCloud\Data\PCLs\agri_floor2.pts',merge=True)
    vis1 = VisualizationO3D()
    vis1.visualize_pointset(pts)

    panorama = PanoramaFactory.CreatePanorama_byPoints(pts, azimuthSpacing=az_res, elevationSpacing=elev_res, voidData=250)

    plt.imshow(panorama.PanoramaImage)
    plt.show()
    pano = panorama.PanoramaImage
    # pano, mean_sigma, mean_kernel = pu.adaptive_smoothing(panorama, .1)
    # print('mean sigma {} mean kernel {}'.format(mean_sigma, mean_kernel))
    r_t, r_p, r_tt, r_pp, r_tp = panorama.computePanoramaDerivatives_adaptive( 2, resolution=az_res, ksize=3, sigma=0)

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
    s_theta = np.vstack(( r_t * cos_phi * cos_theta - r * cos_phi * sin_theta,
                         r_t * cos_phi * cos_theta + r * cos_phi * cos_theta,
                        r_t * cos_phi * cos_theta ))
    s_phi= np.vstack((r_p * cos_phi * cos_theta - r * sin_phi * cos_theta,
                      r_p * cos_phi * sin_theta - r * sin_phi * sin_theta,
                      r_p * sin_phi + r * cos_phi))

    st_st = np.einsum('ji,ji->i', s_theta, s_theta)
    sp_sp = np.einsum('ji,ji->i', s_phi, s_phi)
    st_sp = np.einsum('ji,ji->i', s_theta, s_phi)

    normals = np.cross(s_theta, s_phi, axis=0).T
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    from Properties.Normals.NormalsFactory import NormalsFactory
    normals_pano_reem = NormalsFactory.normals_panorama_xyz(panorama, ksize=15, resolution=az_res)

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
