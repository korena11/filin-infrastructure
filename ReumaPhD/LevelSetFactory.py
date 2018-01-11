'''
infragit
reuma\Reuma
07, Jan, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
import MyTools as mt


class LevelSetFactory:
    phi = None  # level set function (nd-array)
    img = None  # the analyzed image (RasterDate)
    __internal = None  # internal forces
    __external = None  # external forces
    __imgGradient = None  # gradient of the analyzed image
    phi_x = phi_y = phi_xx = phi_xy = phi_yy = None  # level set function derivatives
    g = None  # internal force; can be either constant, weights or function (such as edge function

    # g = 1/(1+|\nabla G(I) * I|).



    def flow(self, type, **kwargs):
        """
        Returns the flow of the level set according to the type wanted
        :param type: can be one of the following:
            'constant': Ct = N ==> phi_t = |\nabla \varphi|
            'curvature': Ct = kN ==> phi_t = div(\nabla \varphi / |\nabla \varphi|)|\nabla \varphi|
            'equi-affine': Ct = k^(1/3) N ==> phi_t = (div(\nabla \varphi / |\nabla \varphi|))^(1/3)*|\nabla \varphi|
            'geodesic': Ct = (g(I)k -\nabla(g(I))N)N ==>
                        phi_t = [g(I)*div(\nabla \varphi / |\nabla \varphi|))^(1/3))*|\nabla \varphi|
            'band':

        ------- optionals ---------
        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param sigma: sigma for LoG gradient
        :param ksize: kernel size, for blurring and derivatives

        :return: phi_t
        """
        ksize = kwargs.get('ksize', 5)
        gradientType = kwargs.get('gradientType', 'L1')
        sigma = kwargs.get('sigma', 2.5)
        kappa = 0.

        norm_nabla_phi = mt.computeImageGradient(self.phi, gradientType = gradientType)
        if type == 'constant':
            return np.abs(norm_nabla_phi)
        else:
            phi_x, phi_y, phi_xx, phi_yy, phi_xy = mt.computeImageDerivatives(self.phi, 2,
                                                                              ksize = ksize, sigma = sigma)
            kappa = ((phi_xx * phi_y ** 2 + phi_yy * phi_x ** 2 - 2 * phi_xy * phi_x * phi_y) / norm_nabla_phi ** 3)

        if type == 'curvature':
            return kappa * norm_nabla_phi

        if type == 'equi-affine':
            return np.cbrt(kappa) * norm_nabla_phi

        if type == 'geodesic':
            # !! pay attention, if self.g is constant - then this flow is actually curavature flow !!
            g_x, g_y = mt.computeImageDerivatives(self.g, 1, ksize = ksize)
            return self.g * kappa * norm_nabla_phi + (g_x * phi_x + g_y * phi_y)


if __name__ == '__main__':
