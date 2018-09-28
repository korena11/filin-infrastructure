'''
infragit
reuma\Reuma
16, Sep, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from numpy import pi, cos, sin
from matplotlib import pyplot as plt
from CurvatureFactory import CurvatureFactory
from PointSet import PointSet

if __name__ == '__main__':
    phi_ = np.arange(0, 2 * pi, .05)
    theta_ = np.arange(0, pi / 2, 0.05)
    phi, theta = np.meshgrid(phi_, theta_)
    R = 1

    x = R * sin(theta) * cos(phi)
    y = R * sin(theta) * sin(phi)
    z = R * cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z)
    ax.axis('equal')
    plt.show()

    pts = PointSet(np.vstack((x, y, z)).T)
    for pt in pts.ToNumpy():
        curv = CurvatureFactory.Curvature_FundamentalForm(pt, pts, 5)
