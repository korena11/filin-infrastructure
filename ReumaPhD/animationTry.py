'''
infraGit
photo-lab-3\Reuma
18, Apr, 2017 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import cv2
from numpy.linalg import norm


import matplotlib.animation as animation

import pylab as pl



if __name__ == '__main__':




    img = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/ownCloud/Data/Images/channel91.png', 1),
                       cv2.COLOR_BGR2GRAY)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()

    plt.imshow(img, interpolation='nearest', cmap='gray')

    dt = 0.01
    ds = 5.
    ds2 = ds ** 2

    # initial boundary contours (BC)
    # space and time steps
    line_length = norm(img.shape)
    s = np.arange(0, line_length, ds)

    # tau \in [0,1];  s (arclencth reparametrization) \in [0,2\pi*radius]
    x_tau = 0 + s / line_length * (img.shape[1])

    y_tau = 120 - s / line_length * (img.shape[0] - 70)
    c = np.vstack((x_tau, y_tau)).T

    b_length = img.shape[0]
    s = np.arange(0, b_length, ds)
    b0_x = np.ones((s.shape[0], 1)) * 1
    b0_y = s.copy()

    b0 = np.hstack((b0_x, b0_y[:, None]))

    b1_x = np.ones((s.shape[0], 1)) * img.shape[1] - 2
    b1_y = s.copy()
    b1 = np.hstack((b1_x, b1_y[:, None]))

    # Plot a scatter that persists (isn't redrawn) and the initial line.
    plt.plot(b0_x, b0_y, '-r')
    plt.plot(b1_x, b1_y, '-r')
    l, = plt.plot(x_tau, y_tau, '-')

   # plt.cla()
   # plt.axis("off")
    plt.ylim([img.shape[0], 0])
    plt.xlim([0, img.shape[1]])



    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)

    x0, y0 = 0, 0

    with writer.saving(fig, "writer_test1.mp4", 100):
        for i in range(100):
          #  x0 += 10 * np.random.randn()
            y_tau += 10
            l.set_data(x_tau, y_tau)
            writer.grab_frame()

    print ('done')