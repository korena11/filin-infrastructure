# General Imports
import numpy as np
import vtk

import VisualizationUtils
# Framework Imports
from BaseProperty import BaseProperty
from ColorProperty import ColorProperty
from PointSet import PointSet
from PointSubSet import PointSubSet
from SegmentationProperty import SegmentationProperty


class VisualizationVTK:
    """
    In order to show PointSet

    One viewport, binding two Pointsets

    .. code-block:: py

        visualization_object = VisualizationVTK(number_of_viewports=1)
        visualization_object.SetRenderWindowName("Test - One Viewport. PointSet. Uniform Color.")
        visualization_object.Bind(input_data=pointset1, render_flag='color', color=(255, 0, 0), new_layer=False)
        visualization_object.Bind(input_data=pointset2, render_flag='color', color=(0, 255, 0), new_layer=True)

    Two viewports, binding two halves with two Pointsets in each half.

    .. code-block:: py

        visualization_object = VisualizationVTK(number_of_viewports=2, two_viewports_vertical_horizontal='V')
        visualization_object.SetRenderWindowName("Test - Two Viewport. PointSet and PointSubSet. Uniform Color.")
        visualization_object.BindFirstHalf(input_data=pointset1, render_flag='color', color=(255, 0, 0), new_layer=False)

        visualization_object.BindFirstHalf(input_data=pointset2, render_flag='color', color=(0, 255, 0), new_layer=True)
        visualization_object.BindSecondHalf(input_data=pointsubset1, render_flag='color', color=(255, 0, 0), new_layer=False)
        visualization_object.BindSecondHalf(input_data=pointsubset2, render_flag='color', color=(0, 255, 0), new_layer=True)

    Four viewports, binding four quarters each with a Pointset or a PointSubSet

    .. code-block:: py

        visualization_object = VisualizationVTK(number_of_viewports=4)
        visualization_object.SetRenderWindowName("Test - Four Viewport. PointSet and PointSubSet. Uniform Color.")
        visualization_object.BindTopLeft(input_data=pointset1, render_flag='color', color=(255, 0, 0), new_layer=False)
        visualization_object.BindTopRight(input_data=pointset2, render_flag='color', color=(0, 255, 0), new_layer=False)
        visualization_object.BindBottomLeft(input_data=pointsubset1, render_flag='color', color=(255, 0, 0), new_layer=False)
        visualization_object.BindBottomRight(input_data=pointsubset2, render_flag='color', color=(0, 255, 0), new_layer=False)

    """

    def __init__(self, number_of_view_ports=1, two_viewports_vertical_horizontal='H'):
        # Setting Viewports
        self.number_of_view_ports = number_of_view_ports
        self.two_viewports_vertical_horizontal = two_viewports_vertical_horizontal
        self.x_mins, self.x_maxs, self.y_mins, self.y_maxs = [0], [1], [0], [1]
        self.set_number_of_viewports()

        self.InitializeParametersVTK()

    def set_number_of_viewports(self):
        if self.number_of_view_ports not in (1, 2, 4):
            raise ValueError('Number of supported viewports can be [1, 2, 4].')

        else:
            if self.number_of_view_ports == 1:
                pass

            elif self.number_of_view_ports == 2:
                if self.two_viewports_vertical_horizontal == 'V':
                    self.x_mins = [0, .5]
                    self.x_maxs = [0.5, 1]
                    self.y_mins = [0, 0]
                    self.y_maxs = [1, 1]
                elif self.two_viewports_vertical_horizontal == 'H':
                    self.x_mins = [0, 0]
                    self.x_maxs = [1, 1]
                    self.y_mins = [0, 0.5]
                    self.y_maxs = [0.5, 1]
                else:
                    raise ValueError('Split Two Viewports Options: \'V\' - Vertical. \'H\' - Horizontal.')

            else:
                # # This way index 0 is bottom left. Index 4 is top right
                # self.x_mins = [0, .5, 0, .5]
                # self.x_maxs = [0.5, 1, 0.5, 1]
                # self.y_mins = [0, 0, .5, .5]
                # self.y_maxs = [0.5, 0.5, 1, 1]

                # This way index 0 is top left. Index 4 is bottom right
                self.x_mins = [0, .5, 0, .5]
                self.x_maxs = [0.5, 1, 0.5, 1]
                self.y_mins = [.5, .5, 0, 0]
                self.y_maxs = [1, 1, .5, .5]

    def InitializeParametersVTK(self):
        '''
        Initialize VTK parameters
        '''

        self.vtkPolydataDictByViewport = {k: [] for k in range(self.number_of_view_ports)}
        self.vtkCombinedPolyData = {}

        self.vtkRenderWindow = vtk.vtkRenderWindow()  # Initialize a VTK render window
        self.vtkRenderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.vtkRenderWindowInteractor.SetRenderWindow(self.vtkRenderWindow)

        self.listVTKRenderers = []
        self.listVTKMappers = []
        self.listVTKActors = []
        for iRenderer in range(self.number_of_view_ports):
            # Initialize VTK Renderers
            self.listVTKRenderers.append(vtk.vtkRenderer())
            # Add the VTK renderer to the VTK render window
            self.vtkRenderWindow.AddRenderer(self.listVTKRenderers[iRenderer])
            self.listVTKRenderers[iRenderer].SetViewport(self.x_mins[iRenderer], self.y_mins[iRenderer],
                                                         self.x_maxs[iRenderer], self.y_maxs[iRenderer])
            # Initialize VTK Mappers
            self.listVTKMappers.append(vtk.vtkPolyDataMapper())

            # Initialize VTK Actors
            self.listVTKActors.append(vtk.vtkActor())
            self.listVTKActors[iRenderer].SetMapper(self.listVTKMappers[iRenderer])
            self.listVTKRenderers[iRenderer].AddActor(self.listVTKActors[iRenderer])

    def __CreatePolyDataForRendering(self, input_data, render_flag, color=(255, 255, 255)):
        if isinstance(input_data, BaseProperty):
            polyData = input_data.Points.ToPolyData()
            numPoints = input_data.Points.Size
        else:
            polyData = input_data.ToPolyData()
            numPoints = input_data.Size

        if render_flag == 'color':
            scalars = np.asarray(np.tile(color, (numPoints, 1)), dtype=np.uint8)

        elif render_flag == 'externRgb' and isinstance(input_data, ColorProperty):
            scalars = input_data.RGB

        elif render_flag == 'rgb' and (
                isinstance(input_data, PointSet) or isinstance(input_data, PointSubSet)) and input_data.RGB != None:
            scalars = input_data.RGB

        elif render_flag == 'intensity' and (
                isinstance(input_data, PointSet) or isinstance(input_data,
                                                               PointSubSet)) and input_data.Intensity != None:
            scalars = input_data.Intensity

        elif render_flag == 'height' and (
                isinstance(input_data, PointSet) or isinstance(input_data, PointSubSet)) and input_data.Z != None:
            scalars = input_data.Z

        elif render_flag == 'height' and isinstance(input_data, BaseProperty):
            scalars = input_data.Points.Z

        elif render_flag == 'segmentation' and isinstance(input_data, SegmentationProperty):
            scalars = input_data.RGB

        elif render_flag == 'parametericColor':
            if len(color[0].shape) == 1:
                scalars = np.expand_dims(color[0], 1)
            else:
                scalars = color[0]

        else:  # display in some default color
            print('Rendering using default color')
            scalars = np.asarray(255 * np.tile((0.5, 0, 0), (len(polyData.points), 1)), dtype=np.uint8)
            render_flag = 'default'

        VisualizationUtils.AddScalarToPolydata(polyData, scalars, render_flag)
        return polyData

    # region  # Region: Binding Functions
    def Bind(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1, new_layer=False):
        self.BindTopLeftData(input_data=input_data, render_flag=render_flag, color=color, point_size=point_size,
                             new_layer=new_layer)

    # ----- #

    def BindFirstHalf(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1, new_layer=False):
        self.BindTopLeftData(input_data=input_data, render_flag=render_flag, color=color, point_size=point_size,
                             new_layer=new_layer)

    def BindSecondHalf(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1, new_layer=False):
        self.BindTopRightData(input_data=input_data, render_flag=render_flag, color=color, point_size=point_size,
                              new_layer=new_layer)

    # ----- #

    def BindTopLeftData(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1,
                        new_layer=False):
        self.InitializePointsArray(region_index=0, input_data=input_data, render_flag=render_flag, color=color,
                                   point_size=point_size, new_layer=new_layer)

    def BindTopRightData(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1,
                         new_layer=False):
        self.InitializePointsArray(region_index=1, input_data=input_data, render_flag=render_flag, color=color,
                                   point_size=point_size, new_layer=new_layer)

    def BindBottomLeftData(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1,
                           new_layer=False):
        self.InitializePointsArray(region_index=2, input_data=input_data, render_flag=render_flag, color=color,
                                   point_size=point_size, new_layer=new_layer)

    def BindBottomRightData(self, input_data, render_flag='color', color=(255, 255, 255), point_size=1,
                            new_layer=False):
        self.InitializePointsArray(region_index=3, input_data=input_data, render_flag=render_flag, color=color,
                                   point_size=point_size, new_layer=new_layer)

    # endregion

    def InitializePointsArray(self, region_index, input_data, render_flag, color=(255, 255, 255), point_size=1,
                              new_layer=False):

        poly_data = self.__CreatePolyDataForRendering(input_data=input_data, render_flag=render_flag, color=color)
        if new_layer:
            self.vtkPolydataDictByViewport[region_index].append(poly_data)
            combined_polydata = VisualizationUtils.CombineVTKPolyDatas(self.vtkPolydataDictByViewport[region_index])
            self.vtkCombinedPolyData[region_index] = combined_polydata
            self.listVTKActors[region_index].GetProperty().SetPointSize(point_size)
            self.listVTKMappers[region_index].SetInputConnection(self.vtkCombinedPolyData[region_index].GetOutputPort())
        else:
            self.vtkPolydataDictByViewport[region_index] = [poly_data]
            self.vtkCombinedPolyData[region_index] = poly_data
            self.listVTKActors[region_index].GetProperty().SetPointSize(point_size)
            self.listVTKMappers[region_index].SetInputData(self.vtkCombinedPolyData[region_index])

        self.listVTKRenderers[region_index].ResetCamera()

    # -----------------------------------------------------------------------------------------

    def SetRenderWindowName(self, nameOfRenderWindow):
        self.vtkRenderWindow.SetWindowName(str(nameOfRenderWindow))

    def SetRenderWindowBackgroundColor(self, color):
        '''

        Change each render window background.

        - Example Four Viewports

        Set same background color:

        .. code-block:: py

            visualization_object = VisualizationVTK(number_of_viewports=4)
            background_color = (255, 0, 0)
            visualization_object.SetRenderWindowBackgroundColor(background_color)

        Set TopLeft to Red, BottomRight to Blue, other two remain default background color (black)

        .. code-block:: py

            visualization_object = VisualizationVTK(number_of_viewports=4)
            background_color = [(255, 0, 0), None, None, (0, 0, 255)]
            visualization_object.SetRenderWindowBackgroundColor(background_color)

        :param color: Tuple of ints (255, 255, 255) or List of tuples of ints ((255, 255, 255), (0, 0, 255))
        :return:
        '''
        if not isinstance(color[0], tuple):
            color = [color]
        for ind in range(len(self.listVTKRenderers)):
            current_renderer = self.listVTKRenderers[ind]
            current_color = color[ind % len(color)]
            if current_color is None:
                current_color = (0, 0, 0)
            current_renderer.SetBackground(current_color)

    def Show(self):
        self.vtkRenderWindowInteractor.Initialize()
        self.vtkRenderWindow.Render()
        self.vtkRenderWindowInteractor.Start()


def RenderPointSet(input_data, render_flag='color', input_data_color=(255, 255, 255), background_color=(0, 0, 0),
                   window_title="One Viewport", vis_obj=None):
    if not vis_obj:
        visualization_object = VisualizationVTK()
        visualization_object.SetRenderWindowName(window_title)
        visualization_object.SetRenderWindowBackgroundColor([background_color])
        visualization_object.Bind(input_data, render_flag=render_flag, color=input_data_color, new_layer=False)
        return visualization_object

    else:
        vis_obj.SetRenderWindowName(window_title)
        vis_obj.SetRenderWindowBackgroundColor([background_color])
        vis_obj.Bind(input_data, render_flag=render_flag, color=input_data_color, new_layer=True)
        return vis_obj


def TestBackgroundColors():
    visualization_object = VisualizationVTK(number_of_view_ports=4)
    visualization_object.SetRenderWindowName("Red Background Color for All.")
    background_color = (255, 0, 0)
    visualization_object.SetRenderWindowBackgroundColor(background_color)
    visualization_object.Show()

    visualization_object.SetRenderWindowName("TopLeft is Red, BottomRight is Blue, other two are default (black).")
    background_color = [(255, 0, 0), None, None, (0, 0, 255)]
    visualization_object.SetRenderWindowBackgroundColor(background_color)
    visualization_object.Show()

    visualization_object.SetRenderWindowName("Each quarter different color.")
    background_color = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
    visualization_object.SetRenderWindowBackgroundColor(background_color)
    visualization_object.Show()


def TestPointSize():
    points1 = (np.random.rand(1000, 3) - 0.5) * 1000.0
    pointset1 = PointSet(points=points1)
    pointsubset1 = PointSubSet(points=pointset1, indices=list(range(0, len(points1), 3)))

    points2 = (np.random.rand(1000, 3) - 0.5) * 1000.0
    pointset2 = PointSet(points=points2)
    pointsubset2 = PointSubSet(points=pointset2, indices=list(range(0, len(points2), 3)))

    visualization_object = VisualizationVTK(number_of_view_ports=2, two_viewports_vertical_horizontal='V')
    visualization_object.SetRenderWindowName("Testing Point Size")

    visualization_object.BindFirstHalf(input_data=pointset1, render_flag='color', color=(255, 0, 0), point_size=5,
                                       new_layer=False)
    visualization_object.BindFirstHalf(input_data=pointset2, render_flag='color', color=(0, 255, 0), point_size=5,
                                       new_layer=True)

    visualization_object.BindSecondHalf(input_data=pointsubset1, render_flag='color', color=(255, 0, 0), point_size=5,
                                        new_layer=False)
    visualization_object.BindSecondHalf(input_data=pointsubset2, render_flag='color', color=(0, 255, 0), point_size=5,
                                        new_layer=True)

    visualization_object.Show()


if __name__ == '__main__':
    # TestBackgroundColors()
    TestPointSize()

    # points1 = (np.random.rand(1000, 3) - 0.5) * 1000.0
    # pointset1 = PointSet(points=points1)
    # pointsubset1 = PointSubSet(points=pointset1, indices=list(range(0, len(points1), 3)))
    #
    # points2 = (np.random.rand(1000, 3) - 0.5) * 1000.0
    # pointset2 = PointSet(points=points2)
    # pointsubset2 = PointSubSet(points=pointset2, indices=list(range(0, len(points2), 3)))
    #
    # number_of_viewports = 2
    # # Initialize VisualizationVTK Object. Number of Viewports = 1, 2, 4
    # visualization_object = VisualizationVTK(number_of_viewports, two_viewports_vertical_horizontal='V')
    # visualization_object.SetRenderWindowName("Test - One Viewport. PointSet. Uniform Color.")
    # visualization_object.SetRenderWindowBackgroundColor([(0.25, 0.25, 0.5), None])
    #
    # visualization_object.BindFirstHalf(input_data=pointset1, render_flag='color', color=(255, 0, 0), new_layer=False)
    # visualization_object.BindFirstHalf(input_data=pointset2, render_flag='color', color=(0, 255, 0), new_layer=True)
    #
    # visualization_object.BindSecondHalf(input_data=pointsubset1, render_flag='color', color=(255, 0, 0),
    #                                     new_layer=False)
    # visualization_object.BindSecondHalf(input_data=pointsubset2, render_flag='color', color=(0, 255, 0), new_layer=True)
    #
    # visualization_object.Show()
