import numpy as np

from BaseProperty import BaseProperty


class RotationMatrixProperty( BaseProperty ):

    def __init__( self, points, rotation_matrix ):
        super(RotationMatrixProperty, self).__init__(points)
        self.__rotation_matrix = rotation_matrix
    
    @property    
    def RotationMatrix( self ):
        """
        Return points' rotation matrix with respect to reference pointset 
        """  
        return self.__rotation_matrix
    
        
    @classmethod
    def EulerAngles_from_R( cls, R ):
        """
        given rotation matrix compute rotation angles
        :param R: - rotation matrix
        :type R: 3x3 nd-array
        :returns omega,phi,kappa: Euler angles
        :rtype tuple

        """
        R = cls.__rotation_matrix
        omega = np.arctan( -R[1, 2] / R[2, 2] )
        phi = np.arcsin( R[0, 2] )
        kappa = np.arctan( -R[0, 1] / R[0, 0] )
        
        return omega, phi, kappa
    

        
