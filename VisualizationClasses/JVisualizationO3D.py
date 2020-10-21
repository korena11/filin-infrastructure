"""
An experimental support for visualization in Jupyter
"""
from open3d import JVisualizer

from BaseProperty import BaseProperty
from PointSubSetOpen3D import PointSetOpen3D
from VisualizationO3D import VisualizationO3D


class JVisualizationO3D(VisualizationO3D):

    def __init__(self):
        super(JVisualizationO3D, self).__init__()
        self.vis = JVisualizer()

    def visualize_property(self, propertyclass, attribute_name=None):
        """
         Visualize property classes

        :param propertyclass: a property class.
        :param attribute_name: name of the attribute to visualize

        .. note::
            Here the attribute_name must be sent, as the callback buttons do not work in Jupyter

        :type propertyclass: BaseProperty

        """
        # initialize custom keys for visualization window
        key_to_callback = self.initialize_key_to_callback()

        self.pointset = PointSetOpen3D(propertyclass.Points)

        # prepare color according to attribute name
        colors_new = (self.__make_color_array(propertyclass.__getattribute__(attribute_name)))
        self.pointset.data.colors = colors_new
        self.vis.add_geometry([self.pointset.data])
        self.vis.show()
