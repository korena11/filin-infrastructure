from RotationMatrixProperty import RotationMatrixProperty
import numpy as np

class RotationMatrixFactory( object ):
    '''
    Different methods for rotation matrix computation
    '''

    @staticmethod
    def Rotation_2Vectors( points, a1, a2 ):
        '''
        compute rotation matrix from vector a1 to a2
        rotation by angle between a1 and a2 around the unit vector
        :Args:
            - points: (PointSet) points to rotate
            - a1,a2: row vectors
        :Returns:
            - R: (3x3 array) rotation matrix
        '''
        a1 = a1 / np.linalg.norm( a1 )  # vector normalization
        a2 = a2 / np.linalg.norm( a2 )  # vector normalization
        v = np.cross( a1, a2 )  # cross product - unit vector
        c = np.dot( a1, a2 )  # scalar product
        h = ( 1 - c ) / ( 1 - c ** 2 )
        
        R = np.array( [[c + h * v[0] ** 2, h * v[0] * v[1] - v[2], h * v[0] * v[2] + v[1]],
                     [h * v[0] * v[1] + v[2], c + h * v[1] ** 2, h * v[1] * v[2] - v[0]],
                     [h * v[0] * v[2] - v[1], h * v[1] * v[2] + v[0], c + h * v[2] ** 2]] )
    
        return RotationMatrixProperty( points, R )
    
    @staticmethod
    def Rotation_EulerAngles( points, angles ):
        '''
        given rotation angles build a rotation matrix
        :Args:
            - points: (PointSet) points to rotate
            - angles: (tuple) omega, phi, kappa
        :Returns: 
            - R: (3x3 array) rotation matrix
        '''
        omega, phi, kappa = angles[0], angles[1], angles[2]
        co, so = np.cos( omega ), np.sin( omega )
        cp, sp = np.cos( phi ), np.sin( phi )
        ck, sk = np.cos( kappa ), np.sin( kappa )
        
        R = np.array( [[cp * ck, -cp * sk, sp],
                      [co * sk + so * sp * ck, co * ck - so * sp * sk, -so * cp],
                      [so * sk - co * sp * ck, so * ck + co * sp * sk, co * cp]] )
        
        return RotationMatrixProperty( points, R )
            
    @staticmethod
    def Rotation_AxisAngle( points, axis, theta ):
        '''
        create rotation matrix given axis and angle
        :Args:
            - points: (PointSet) points to rotate
            - axis: (vector) axis to rotate around
            - theta: rotation angle in degrees
            
        :Returns:
            - R: (3x3 array) rotation matrix
        '''
        s = np.sin( np.radians( theta ) )
        c = np.cos( np.radians( theta ) )
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]
        R = np.array( [[t * x ** 2 + c, t * x * y + s * z, t * x * z - s * y],
                  [t * x * y - s * z, t * y ** 2 + c, t * y * z + s * x],
                  [t * x * z + s * y, t * y * z - s * x, t * z ** 2 + c]] )
        return RotationMatrixProperty( points, R )
    
    @staticmethod
    def Rotation_Quaternion( points, q ):
        '''
        create rotation matrix given axis and angle
        :Args:
            - points: (PointSet) points to rotate
            - q: (4x1) quaternion
        :Returns:
            - R: (3x3 array) rotation matrix
        '''
        R = np.array( [[q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * ( q[1] * q[2] - q[0] * q[3] ), 2 * ( q[1] * q[3] + q[0] * q[2] )],
                    [2 * ( q[1] * q[2] + q[0] * q[3] ), q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2, 2 * ( q[3] * q[2] - q[0] * q[1] )],
                    [2 * ( q[1] * q[3] - q[0] * q[2] ), 2 * ( q[3] * q[2] + q[0] * q[1] ), q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2]] )
        return RotationMatrixProperty( points, R )
