import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from BaseProperty import BaseProperty
from PointSubSetOpen3D import PointSetOpen3D, PointSubSetOpen3D


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
        key_to_callback[ord("K")] = cls.toggle_black_white_background

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
    def visualize_pointset(cls, pointset, colors=None):
        """
        Visualize PointSet with color property, if exists

        :param pointset: the point set to visualize
        :param colors: ColorProperty.ColorProperty

        :type pointset: PointSetOpen3D, PointSet.PointSet
        """
        from Color.ColorProperty import ColorProperty
        key_to_callback = cls.initialize_key_to_callback()

        # Change the pointset to an instance of PointSetOpen3D
        if isinstance(pointset, PointSubSetOpen3D):
            pcd = PointSetOpen3D(pointset.ToNumpy())

        elif isinstance(pointset, PointSetOpen3D):
            pcd = pointset

        else:
            try:
                pcd = PointSetOpen3D(pointset)
            except:
                print('pointset type has to be convertible to PointSetOpen3D')
                raise TypeError

        if isinstance(colors, ColorProperty):
            colors_ = colors.rgb
            pcd.data.colors = o3d.Vector3dVector(colors_)

        o3d.draw_geometries_with_key_callbacks([pcd.data], key_to_callback)

    def visualize_property(self, propertyclass, attribute_name=None, zero_black=False, epsilon=0.05):
        """
        Visualize property classes

        :param propertyclass: a property class. Can have multiple attributes to visualize
        :param attribute_name: name of the attribute to visualize. If None -- can cycle through the attribute using "A"
        :param zero_black: show close to zero values as black dots. Default: False
        :param epsilon: the thershold for zero value       

        :type propertyclass: BaseProperty
        :type attribute_name: str
        :type zero_black: bool
        :type epsilon: float

        """

        # initialize custom keys for visualization window
        key_to_callback = self.initialize_key_to_callback()

        self.pointset = PointSetOpen3D(propertyclass.Points)
        colors_new = []
        attribute_name = []
        for att in dir(propertyclass):
            # filter out private properties
            if '__' in att:
                continue
            # filter out properties that are not arrays and cannot be converted into color arrays
            if isinstance(propertyclass.__getattribute__(att), np.ndarray):
                colors_new.append(self.__make_color_array(propertyclass.__getattribute__(att), zero_black, epsilon))
                attribute_name.append(att)
        if len(colors_new) == 0:
            colors_new = [self.__make_color_array(propertyclass.getValues())]
            attribute_name = ['default']

        self.high = len(attribute_name)

        self.colors = colors_new
        self.attribute_name = attribute_name

        from itertools import cycle
        self.colormap_list = cycle(['coolwarm', 'RdYlBu', 'PuOr', 'PiYG', 'jet', 'summer', 'winter', 'hot', 'gray'])
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
        array = np.asarray(self.colors[self.index])
        colored = colormap(array)
        R = colored[:, 0, 0].flatten()
        G = colored[:, 0, 1].flatten()
        B = colored[:, 0, 2].flatten()

        return o3d.Vector3dVector(np.vstack((R, G, B)).T)

    @classmethod
    def __make_color_array(cls, array, zero_black=False, epsilon=0.05):
        """
        Prepare an array to be used as a color array for visualization

        :param array: array of numbers to be converted into color array
        :param zero_black: show close to zero values as black dots
        :param epsilon: the definition of "close to" 

        :type array: numpy.array
        :type zero_black: bool

        :return: an open3d vector
        :rtype: o3d.Vector3D
        """
        if isinstance(array, list):
            array = np.asarray(array)
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

        if zero_black:
            if rgb[np.where(rgb < 0)].size == 0:
                rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            else:
                # make zero the lowest number, number below zero higher in a notch
                rgb[np.where(np.abs(array) < epsilon), :] = 0
                rgb_normed = np.max(rgb[np.where(rgb < 0)]) + (rgb - rgb.min()) / (rgb.max() - rgb.min())
                rgb_normed[np.where(np.abs(array) < epsilon), :] = 0
        else:
            # normalize to range [0...1]
            rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        return o3d.Vector3dVector(rgb_normed)
