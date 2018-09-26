import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

if __name__ == '__main__':
    '''
    A time variant circle
    '''

    # define the initial circle
    radius = 10
    s = np.arange(0, 2 * pi * radius, 0.1)

    # tau \in [0,1];  s (arclencth reparametrization) \in [0,2\pi*radius]
    x = radius * cos(s / (radius))
    y = radius * sin(s / (radius))
    c = np.vstack((x, y))
    # define velocity
    velocity = 0.5

    # draw
    plt.plot(x, y, '-b')
    plt.axis('equal')
    plt.show()

    plt.figure()
    plt.axis('equal')

    for t in range(0, 100):
        xt = -cos(s / radius) * velocity * 1 / radius
        yt = -sin(s / radius) * velocity * 1 / radius
        x += xt
        y += yt
        plt.plot(x, y, '-r')
        plt.pause(0.01)
        plt.draw()
        plt.hold(False)
    plt.show()
