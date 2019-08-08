import ctypes

import numpy as np
import vtk
from vtk.util import numpy_support


def AddScalarToPolydata(vtkPolydata, scalars, scalars_name='colors'):
    vtkfloat = numpy_support.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkfloat.SetName(scalars_name)
    vtkPolydata.GetPointData().AddArray(vtkfloat)

    vtkPolydata.GetPointData().SetActiveScalars(scalars_name)


def MakeVTKPoints(points, deep=True):
    """ Convert numpy points to a vtkPoints object """

    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetData(numpy_support.numpy_to_vtk(points, deep=deep))
    return vtkPoints


def MakeVTKPointsMesh(points):
    """ Create a PolyData object from a numpy array containing just points """
    if points.ndim != 2:
        points = points.reshape((-1, 3))

    npoints = points.shape[0]

    # Make VTK cells array
    cells = np.hstack((np.ones((npoints, 1)), np.arange(npoints).reshape(-1, 1)))
    cells = np.ascontiguousarray(cells, dtype=ctypes.c_int64)
    vtkCellArray = vtk.vtkCellArray()
    vtkCellArray.SetCells(npoints, numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))

    # Convert points to vtk object
    vtkPoints = MakeVTKPoints(points)

    # Create polydata
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(vtkPoints)
    pdata.SetVerts(vtkCellArray)
    return pdata


def CombineVTKPolyDatas(list_of_polydata):
    if len(list_of_polydata) == 1:
        return list_of_polydata[0]

    # Append the two meshes
    appendFilter = vtk.vtkAppendPolyData()

    [appendFilter.AddInputData(tempPoly) for tempPoly in list_of_polydata]

    appendFilter.Update()
    # return appendFilter

    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()

    return cleanFilter
