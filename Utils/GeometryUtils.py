import numpy as np
from shapely import geometry

def curve2shapely(contour, crs=(0,0), eps=.2):
    """
    Convert matplotlib curves to shapely geometry

    :param contour: a list of contours generated by matplotlib
    :param crs: coordinate system to move the curves to the original system (min_x, min_y)
    :param eps: threshold for closed polygon. If smaller -- the curve is a polygon. Default: 0.2

    :type contour: list
    :type crs: tuple

    :return: polygons that enable easier work (vectoric)

    :rtype: list of shapely_polygon.geometry.polygon.Polygon

    """


    poly = []

    for col in contour:

        try:
            cp = np.stack(col.get_path().to_polygons()[0])
        except IndexError:
            try:
                cp = np.stack(col.get_path().to_polygons())
            except:
                continue

        dist_last = np.sqrt(np.sum((cp[-2, :] - cp[-1, :]) ** 2))

        x = cp[:, 0] + crs[0]
        y = cp[:, 1] + crs[1]

        if dist_last < eps:
            new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(x, y)])

        else:
            x = x[:-1]
            y = y[:-1]
            new_shape = geometry.LineString([(i[0], i[1]) for i in zip(x, y)])

        poly.append(new_shape)

    return poly

def reclassify_shapely(shapes, eps=.2):
    """
    Reclassifies shapely shapes into Polygons and LineString according to a new threshold

    :param shapes: shapes to be reclassified
    :param eps: the new threshold according to which the shape is classified as a Polygon of LineString

    :type shapes: list, shapely.geometry.LineString.LineString, shapely.geometry.polygon.Polygon

    :return: reclasiffied shapes

    :rtype: list or shapely.geometry.LineString.LineString, shapely.geometry.polygon.Polygon
    """
    new_shapes = []
    if not isinstance(shapes, list):
        # make as list
        shapes = [shapes]

    else:
        for shape in shapes:
            new_shape = shape
            if shape.type == 'Polygon':
                xy = np.stack(shape.exterior.xy)
                # check the distance between the two last points
                dist = np.sqrt(np.sum((xy[:,-1]- xy[:,-2])**2))
                if dist > eps:
                    new_shape = geometry.LineString([*xy.T])

            else:
                xy = np.stack(shape.xy)
                # check the distance between the last and the first points
                dist = np.sqrt(np.sum((xy[:,-1]- xy[:,0])**2))

                if dist < eps:
                    new_shape = geometry.Polygon([*xy.T])

            new_shapes.append(new_shape)

    return new_shapes


def circularity_measure(poly):
    r"""
    Measure the circularity of a polygon

    :param polygon: shapely polygon

    :type polygon: shapely_polygon.geometry.polygon

    :return: the circularity measure

    The circularity is measured by:

    .. math::
        C = \frac{\ell^2}{4\pi A}

    where :math:`\ell` is the perimeter and :math:`A` is the area of the polygon
    """
    import shapely.geometry.polygon
    if isinstance(poly, shapely.geometry.polygon.Polygon):
        return poly.length**2 / (4 * np.pi * poly.area )
    else:
        return 3


def points_in_polygon(multi_p, poly, pointset):
    """
    Finds all points that are within a polygon

    :param multi_p: the point cloud as a shapely MultiPoint
    :param pointset: the point cloud in which points are searched
    :param poly:  the bounding polygon within which the points should be found

    :type multi_p: shapely_polygon.geometry.point
    :type pointset: DataClasses.PointSet.PointSet,
    :type poly: shapely_polygon.geometry.polygon

    :return: all points within the polygon

    :rtype: DataClasses.PointSubSet.PointSubSet

    """

    from DataClasses.PointSubSet import PointSubSet
    from tqdm import tqdm

    # find all points in polygon
    idx = [p.within(poly) for p in tqdm(multi_p, desc='looking for points within the polygon')]
    id = np.where(idx)[0]

    return PointSubSet(pointset, id)


def point_on_polygon(poly, pointset, leafsize=40):
    """
    Find the point along the polygon

    :param poly:  the bounding polygon within which the points should be found
    :param pointset: the point cloud in which points are searched
    :param leafsize: size of a leaf for the KDtree. Default: 40.

    :type pointset: DataClasses.PointSet.PointSet,
    :type poly: shapely_polygon.geometry.polygon.Polygon
    :type leafsize: int

    :return: closest points from the point cloud to the polygon

    :rtype: DataClasses.PointSubSet.PointSubSet
    """

    from sklearn.neighbors import KDTree
    from DataClasses.PointSubSet import PointSubSet

    kdt = KDTree(pointset.ToNumpy()[:, :2], leafsize)
    dists, id2 = kdt.query(np.stack(poly.exterior.xy).T, 1)

    return PointSubSet(pointset, id2[0])


def polygon_to_linestring(pols):
    """
    Converts shapely polygons to shapely linestrings

    :param pols: polygons to convert

    :type pols: list of shapely_polygon.geometry.polygon.Polygon
    :return: list of lines

    :rtype: list
    """
    lines = []
    for pol in pols:
        boundary = pol.boundary
        if boundary.type == 'MultiPoint':
            boundary = pol[0]

        if boundary.type == 'MultiLineString':
            for line in boundary:
                lines.append(line)
        else:
            lines.append(boundary)

    return lines

def fit_line_LS(xy):
    """
    fit a line to xy points by linear least squares

    :param xy: points coordinates

    :type xy: np.array (nx2)

    :return: x0: fitted coefficients, RMSE: root mean squared error, v

    :rtype: tuple

    The model:

    .. math::
        y = ax + b
    """

    # model
    n = xy.shape[0]
    l = xy[:, 1]

    A = np.ones((n, 2))
    A[:,0] = xy[:,0]

    x0 = np.linalg.solve(A.T.dot(A), A.T.dot(l))
    v = A.dot(x0) - l

    sig2 = v.T.dot(v)/(n-2)

    return  x0, np.sqrt(sig2), v

def fit_circle_GH(xy, r_approx, ):
    """
    Fit a circle using Gauss-Helmert model
    :param xy:
    :return:
    """