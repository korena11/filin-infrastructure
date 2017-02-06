'''
infraGit
photo-lab-3\Reuma
30, Jan, 2017 

Active contours model using the tensorflow gradient descent.
'''
from ContourDevelopment import getValueSubpix
from scipy.linalg import toeplitz
from scipy.ndimage import filters
from numpy import sin, cos, pi
from matplotlib import pyplot as plt
from MyTools import imshow
from skimage import measure
from functools import partial
import matplotlib.mlab as mlab
import numpy as np
import platform
import cv2

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')

def createGaussianImage(gridSpacing = .025, noise = 0.01):

    x = np.arange(-3.0, 3.0, gridSpacing)
    y = np.arange(-2.0, 2.0, gridSpacing)
    X, Y = np.meshgrid(x, y)
    img = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0., 0.0)
    gaussian = img.copy()

    img = img + cv2.randn(gaussian, 0, noise)


def _buildPentadiagonal(n, array):
    '''
    Builds a pentadiagonal matrix
    :param n: numer of points (repetitions of array)
    :param array: array according to which the pentadiagonal matrix is solved

    :return: a Toeplitz matrix array to the arrays sent
    '''

    #--- build array according to which the Toeplitz matrix will be built ---
    numElements =  n % array.shape[1]
    numRepetitions = np.int(n / array.shape[1])


    for i in range(numRepetitions):
        array = np.append(array, array)

    if numElements != 0:
        partArray = array[:numElements]
        array = np.append(array, partArray)
    return toeplitz(array)

def returnPentadiagonal(n, a, b, c):
    '''
    inverts (I-stepSize A)
    :param n: number of points
    :param a: 1 + * timeStep / spaceStep^2 * (2 * alpha  + 6 * beta) (alpha is the tension weight; beta is the stiffness weight)
    :param b: -timeStep / spaceStep^2 * (alpha + 4 * beta)
    :param c: timeStep * beta

    :return: inverted (I-stepSize * A)
    '''
    M = np.diag(np.tile(a, n))
    M += np.diag(np.tile(b, n - 1), 1) + np.diag(np.tile(b, 1), -n + 1)
    M += np.diag(np.tile(b, n - 1), -1) + np.diag(np.tile(b,1), n - 1)
    M += np.diag(np.tile(c, n - 2), 2) + np.diag(np.tile(c, 2), -n + 2)
    M += np.diag(np.tile(c, n - 2), -2) + np.diag(np.tile(c, 2), n - 2)



    return (M)

def generateImageEnergy(I, w_line, w_edge, w_term, edgeEnergyType = 'L1'):
    '''
    Compute an image of the image energy at each pixel
    :param I: image
    :param w_line: weights of the line term
    :param w_edge: weights of the edge term
    :param w_term: weights of the termination term
    :param edgeEnergyType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
    :return: energy image
    '''
    img = cv2.blur(I, (5, 5))

    # compute image gradient
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    if edgeEnergyType == 'L1':
        grad_img = np.abs(dx) + np.abs(dy)  # L1-norm of grad(I)
    elif edgeEnergyType== 'L2':
        grad_img = np.sqrt(dx **2 + dy**2)
    elif edgeEnergyType == 'LoG':
        grad_img = filters.gaussian_laplace(I, 1.)


    # ---- Second derivatives ---
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=5)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=5)
    dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=5)

    energy_line = I
    energy_edge = -grad_img
    energy_term = (dyy*dx**2 - 2*dxy*dx*dy + dxx*dy**2)/((1 + dx**2 + dy**2)**(1.5))

    return w_line * energy_line + w_edge * energy_edge + w_term * energy_term



if __name__ == '__main__':

    #--- initializations
    img = cv2.cvtColor(cv2.imread(r'/home/photo-lab-3/Dropbox/PhD/Data/ActiveContours/Images/tractor.png', 1),
                       cv2.COLOR_BGR2GRAY)

    plt.imshow(img, interpolation='nearest', cmap='gray')

    # tension and stiffness
    alpha = 50
    beta = 0
    w_line = 0.4
    w_edge = 0.5
    w_term = 0.6

    # space and time steps
    dt = 0.07
    ds = 2.
    ds2 = ds**2

    # other constants
    a = alpha * dt / ds2
    p = b = beta * dt / ds2**2
    q = -(a + 4 * b)
    r = 1 + 2 * a + 6 * b


    # define an initial contour
    radius = 75
    s = np.arange(0,  2* pi * radius, ds)

    # tau \in [0,1];  s (arclencth reparametrization) \in [0,2\pi*radius]
    x_tau = img.shape[0]/2 +radius * cos(s / radius)
#    x_tau = np.append(x_tau, x_tau[0])

    y_tau = img.shape[1]/2 +radius * sin(s / radius)
 #   y_tau = np.append(y_tau, y_tau[0])
    c = np.vstack((y_tau, x_tau)).T



    energy_image = generateImageEnergy(img, w_line, w_edge, w_term, edgeEnergyType='LoG')
    energy_image = cv2.blur(energy_image, (7,7))
    energy_image_y = filters.sobel(energy_image, 0)
    energy_image_x = filters.sobel(energy_image, 1)

    energy_image_x = cv2.blur(energy_image_x, (7,7))
    energy_image_y = cv2.blur(energy_image_y, (7,7))

    M = returnPentadiagonal(x_tau.shape[0], r, q, p)
    eyeM = np.eye(M.shape[0])
    invM = np.linalg.inv(M)
    #invM = np.linalg.inv(M + dt * eyeM)

    #contours = measure.find_contours(img, 0.5)
    plt.plot(c[:, 0], c[:, 1], '-r')

    epsilon = 1000


    while epsilon > 5:
        if np.any(c < 0) or np.any(c[:, 0] > img.shape[1]) or np.any(c[:, 1] > img.shape[0]):
            break



        # external force - using energy_image
        fx = np.array(map(partial(getValueSubpix, energy_image_x), c[:,1], c[:,0]))
        fy = np.array(map(partial(getValueSubpix, energy_image_y), c[:,1], c[:,0]))


        # internal force - solving the (I-timeStep * A) = x, +timeStep * fx,y
        # contour movement
        ct_1 = c.copy()
        c[:,0] = invM.dot(c[:, 0] + dt * fx)
        c[:,1] = invM.dot(c[:, 1] + dt * fy)

        epsilon = np.linalg.norm(c - ct_1)
        print epsilon
        plt.cla()
        plt.axis("off")
        plt.ylim([img.shape[0], 0])
        plt.xlim([0, img.shape[1]])

        imshow(img, cmap='gray')

        plt.plot(c[:,0], c[:,1], '-r.')
        plt.pause(1e-6)


     #   c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second




    plt.show()



