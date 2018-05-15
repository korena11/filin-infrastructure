from numpy import nonzero, zeros, uint8 #logical_and

from ColorProperty import ColorProperty


class ColorFactory:
    """
    Color a PointSet using different methods
    """
        
    @staticmethod
    def ProjectImages(points, images, matrices):
        """
        Project images onto a points set (giving each point a color)

        :param points: The point cloud to color
        :param images: The images to project
        :param matrices: Camera matrices

         :type points: PointSet.PointSet
         :type images: list of nd-array
         :param matrices: list of 3x4 nd-array

        :return: Each point's color is taken from the appropriate image

        :rtype: ColorProperty

        """           
        xyz = points.ToNumpy()
        
        # Defining the color vector for the points in the point cloud
        rgb = zeros((xyz.shape), dtype = uint8)
        
        for img, transformMatrix in zip(images, matrices):             
    
            # Calculating the image coordinates of all the points
            w = (transformMatrix[2, 0] * xyz[:, 0] + transformMatrix[2, 1] * xyz[:, 1] + 
                 transformMatrix[2, 2] * xyz[:, 2] + transformMatrix[2, 3])
            v = (transformMatrix[1, 0] * xyz[:, 0] + transformMatrix[1, 1] * xyz[:, 1] + 
                 transformMatrix[1, 2] * xyz[:, 2] + transformMatrix[1, 3]) / w
            u = (transformMatrix[0, 0] * xyz[:, 0] + transformMatrix[0, 1] * xyz[:, 1] + 
                 transformMatrix[0, 2] * xyz[:, 2] + transformMatrix[0, 3]) / w
        
#            v = img.shape[0] - 1 - v.astype(int)
            v = v.astype(int)
            u = u.astype(int)
     
            # Finding the indexes of all the points that are in the image
#            pointsInImageIndexes = nonzero(logical_and(logical_and(logical_and(u >= 0, u < img.shape[1]),
#                                                                   logical_and(v >= 0, v < img.shape[0])), w > 0))[0]
            pointsInImageIndexes = nonzero((u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0]) & (w > 0))[0]

            # Assigning the color based on image coordinates
            rgb[pointsInImageIndexes, 0] = img[v[pointsInImageIndexes], u[pointsInImageIndexes], 0]  
            rgb[pointsInImageIndexes, 1] = img[v[pointsInImageIndexes], u[pointsInImageIndexes], 1]
            rgb[pointsInImageIndexes, 2] = img[v[pointsInImageIndexes], u[pointsInImageIndexes], 2]
        
        
        colorProperty = ColorProperty(points, rgb)
        return colorProperty 