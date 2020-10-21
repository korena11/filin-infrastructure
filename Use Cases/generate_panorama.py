import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from DataClasses.PointSet import PointSet
from DataClasses.PointSetOpen3D import PointSetOpen3D
from Properties.Panoramas.PanoramaFactory import PanoramaFactory
from Utils import MyTools as mt
from VisualizationClasses.VisualizationO3D import VisualizationO3D
from IOmodules.IOFactory import IOFactory



if __name__ == '__main__':

    az_res = 0.115
    elev_res =  .115

    az_res += 0.0001
    elev_res += 0.0001
    pts = IOFactory.ReadPts(r'D:\Documents\Python Scripts\phd\ReumaPhD\data\Bulbus\st3_selection.pts',merge=True)
    # vis1 = VisualizationO3D()
    # vis1.visualize_pointset(pts)

    panorama = PanoramaFactory.CreatePanorama(pts, azimuthSpacing=az_res, elevationSpacing=elev_res, voidData=21)

    plt.imshow(panorama.rangeImage, cmap='gray')
    plt.show()
