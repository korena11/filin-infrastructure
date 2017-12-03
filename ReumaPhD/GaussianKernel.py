import matplotlib
import matplotlib.pyplot as plt
from numpy import zeros, exp, sum, meshgrid, arange, pi

matplotlib.use('TkAgg')


def GaussianKernel(size, sigma):
    # Size = 256;
    # s = 30;

    G = zeros((size, size))
    m = size  # (G,2)
    xo = (m + 1.0) / 2.0
    n = size  # (G,1)
    yo = (n + 1.0) / 2.0

    X, Y = meshgrid(arange(1, m + 1) - xo, arange(1, n + 1) - yo)

    G = 1 / (2 * pi * sigma ** 2) * exp(-(X ** 2 + Y ** 2) / 2 / sigma ** 2)
    G = G / sum(G)

    plt.imshow(G)
    #
    #    ax.set_xlabel('X Label')
    #    ax.set_ylabel('Y Label')
    #    ax.set_zlabel('Z Label')
    #
    #    plt.show()

    return G


if __name__ == '__main__':
    G = GaussianKernel(256, 20)

    print
    G
