import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from Properties.BaseProperty import BaseProperty
from Properties.Normals.NormalsProperty import NormalsProperty
from DataClasses.PointSubSetOpen3D import PointSetOpen3D, PointSubSetOpen3D


class VisualizationO3D:

    def __init__(self):
        self.pcd = None
        # define for toggling
        self.colors = None
        self.index = 0
        self.high = 1
        self.attribute_name = None
        self.pointset = None

    @classmethod
    def initialize_key_to_callback(cls):
        """
        Sets extra keys to callback, to be available for all VisualizeO3D

        :return: key_to_callback dictionary

        :rtype: dict
        """
        key_to_callback = {}
        key_to_callback[ord("B")] = cls.toggle_black_white_background

        return key_to_callback

    @classmethod
    def toggle_black_white_background(cls, vis):
        """
        Change background from white to black and vise versa

        :param vis: an open3d visualization object

        :type vis: open3d.Visualizer

        """
        opt = vis.get_render_option()
        if np.array_equal(opt.background_color, np.ones(3)):
            opt.background_color = np.zeros(3)
        else:
            opt.background_color = np.ones(3)
        return False

    @classmethod
    def visualize_pointset(cls, pointset, colors=None, drawCoordianteFrame=False,
                           coordinateFrameSize=2.0, coordinateFrameOrigin='default', originOffset=1.5):
        """
        Visualize PointSet with color property, if exists

        :param pointset: the point set to visualize
        :param colors: ColorProperty.ColorProperty

        :type pointset: PointSetOpen3D, PointSet.PointSet
        """
        from Properties.Color.ColorProperty import ColorProperty
        from Properties.Segmentation.SegmentationProperty import SegmentationProperty

        key_to_callback = cls.initialize_key_to_callback()

        # Change the pointset to an instance of PointSetOpen3D
        if isinstance(pointset, PointSubSetOpen3D):
            pcd = PointSetOpen3D(pointset.ToNumpy())

        elif isinstance(pointset, PointSetOpen3D):
            pcd = pointset

        elif isinstance(pointset, SegmentationProperty):
            pcd = PointSetOpen3D(pointset.Points)
            colors = pointset.RGB

        else:
            try:
                pcd = PointSetOpen3D(pointset)
            except:
                print('pointset type has to be convertible to PointSetOpen3D')
                raise TypeError

        if not (colors is None):
            if isinstance(colors, ColorProperty):
                colors_ = colors.rgb
            elif isinstance(colors, np.ndarray):
                colors_ = colors
            if colors_.max() > 1:
                colors_ /= 255
            pcd.data.colors = o3d.Vector3dVector(colors_)

        if drawCoordianteFrame:
            if coordinateFrameOrigin == 'min':
                cf = o3d.geometry.create_mesh_coordinate_frame(size=coordinateFrameSize,
                                                               origin=pcd.ToNumpy().min(axis=0) - originOffset)
            else:
                cf = o3d.geometry.create_mesh_coordinate_frame(size=coordinateFrameSize)
            o3d.draw_geometries_with_key_callbacks([pcd.data, cf], key_to_callback)
        else:
            o3d.draw_geometries_with_key_callbacks([pcd.data], key_to_callback)

    def visualize_neighborhoods(self, neighborhoodProperty):
        """
        Visualize points neighborhoods in white. Center point in red, other points in green

        Going through the points according to their order in the property by pressing 'X'

        :param neighborhoodProperty: the neighborhood property which will be shown

        :type neighborhoodProperty: Properties.Neighborhood.NeighborsProperty.NeighborsProperty

        ..note::
            To view the neighborhood, make sure the background is black by pressing 'B'

        """
        from Properties.Neighborhood.NeighborsProperty import NeighborsProperty
        self.current_neighborhood = neighborhoodProperty.getNeighborhood(0).neighbors
        self.pointset = PointSetOpen3D(neighborhoodProperty.Points)
        self.neighborhood = neighborhoodProperty
        key_to_callback = self.initialize_key_to_callback()
        key_to_callback[ord('X')] = self.next_neighborhood

        o3d.draw_geometries_with_key_callbacks([self.pointset.data], key_to_callback)

    def next_neighborhood(self, vis):
        """
        Light the next neighborhood in the list

        two new point clouds: one holds only the point (colored in red) and another holds the neighborhood colored in gray levels (can be changed)
        """
        current_neighborhood = self.neighborhood.__next__()

        color_by_neighborhood = np.ones((self.pointset.Size, 3)) * np.array([128,255,0]) / 255 # paint all point cloud in green
        # color_by_neighborhood[current_neighborhood.neighborhoodIndices] = np.ones((current_neighborhood.Size, 3))
        color_by_neighborhood[current_neighborhood.neighborhoodIndices] = self.__make_color_array(current_neighborhood.weighted_distances)
        color_by_neighborhood[current_neighborhood.center_point_idx] = np.array([1, 0, 0])

        self.pointset.data.colors = o3d.Vector3dVector(color_by_neighborhood)
        vis.update_geometry()

    def visualize_property(self, propertyclass):
        """
        Visualize property classes

        :param propertyclass: a property class. Can have multiple attributes to visualize

        :type propertyclass: BaseProperty
        """
        from numpy import ndarray
        from itertools import cycle

        # initialize custom keys for visualization window
        key_to_callback = self.initialize_key_to_callback()
        self.pointset = PointSetOpen3D(propertyclass.Points)

        # to present normals
        if isinstance(propertyclass, NormalsProperty):
            if not self.pointset.data.has_normals:
                self.pointset.data.CalculateNormals()
            self.pointset.data.normals = o3d.Vector3dVector(propertyclass.Normals)
        elif isinstance(propertyclass.Points, PointSetOpen3D):
            self.pointset.data.normals = propertyclass.Points.data.normals

        colors_new = []
        attribute_name = []
        for att in dir(propertyclass):
            # filter out private properties
            if '__' in att:
                continue
            # filter out properties that are not arrays and cannot be converted into color arrays
            if isinstance(propertyclass.__getattribute__(att), (np.ndarray, ndarray)):
                colors_new.append(self.__make_color_array(propertyclass.__getattribute__(att)))
                attribute_name.append(att)
        if len(colors_new) == 0:
            colors_new = [self.__make_color_array(propertyclass.getValues())]
            attribute_name = ['default']

        self.high = len(attribute_name)

        self.colors = colors_new
        self.attribute_name = attribute_name
        self.colormap_list = cycle(['jet', 'summer', 'winter', 'hot', 'gray', 'PiYG', 'coolwarm', 'RdYlBu'])

        key_to_callback[ord('A')] = self.toggle_attributes_colors
        key_to_callback[ord('C')] = self.toggle_colormaps

        o3d.draw_geometries_with_key_callbacks([self.pointset.data], key_to_callback)

    def toggle_attributes_colors(self, vis):
        """
        Change visualization colors according to attributes of the property

        :param vis:

        :type vis: open3d.Visualizer

        """

        if self.index >= self.high - 1:
            self.index = 0
        else:
            self.index += 1

        print(self.attribute_name[self.index])
        self.pointset.data.colors = self.colors[self.index]
        vis.update_geometry()

    def toggle_colormaps(self, vis):
        """
        changes between colormaps
        :param vis:
        :return:
        """
        colorname = self.colormap_list.__next__()
        print(colorname)
        cm = plt.get_cmap(colorname)

        new_colors = self.__change_colormap(cm)
        self.pointset.data.colors = new_colors
        vis.update_geometry()


    def __change_colormap(self, colormap):
        """
        changes the array to a given colormap

        :param array: the array of colors to transform
        :param colormap: a colormap object

        :type array: np.array
        :type colormap: plt.colormap

        :return: the array in the new colormap
        """
        array = np.asarray(self.colors)[self.index]
        colored = colormap(array)
        R = colored[:, 0, 0].flatten()
        G = colored[:, 0, 1].flatten()
        B = colored[:, 0, 2].flatten()

        return o3d.Vector3dVector(np.vstack((R, G, B)).T)

    @classmethod
    def __make_color_array(cls, array):
        """
        Prepare an array to be used as a color array for visualization

        :param array: array of numbers to be converted into color array

        :type array: numpy.array

        :return: an open3d vector
        :rtype: o3d.Vector3D
        """
        size_array = array.shape

        # check the dimension of the array, if only a list of numbers, it should be transformed into a 2D array (nx3)
        if len(size_array) <= 2:
            rgb = np.ones((size_array[0], 3))
            try:
                rgb *= array[:, None]
            except:
                rgb *= array

        elif len(size_array) == 3:
            rgb = array.reshape((size_array[0] * size_array[1], 3))

        else:
            rgb = array

        # normalize to range [0...1]
        rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)

        return o3d.Vector3dVector(rgb_normed)
