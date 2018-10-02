import numpy as np


# from numba import jit, double


def azimuth(dx, dy):
    if (dx >= 0 and dy >= 0) or (dx == 0 and dy < 0):
        theta = np.arctan2(dx, dy)
    elif dx < 0 and dy == 0:
        theta = 3.0 * np.pi / 2.0
    elif dx < 0:
        theta = 2.0 * np.pi + np.arctan2(dx, dy)
    elif dy < 0 < dx:
        theta = 3.0 * np.pi / 2.0 - np.arctan2(dx, dy)
    return theta


# ----------- TODO: sit with Elia. Shouldn't this be in the curve class? ---------
def sample_points_on_curve(curve_object, sample_distance):
    """
    
    :param curve_object: 
    :param sample_distance: 
    :return: 
    """
    list_of_sample_points = [curve_object.points[0]]
    for current_distance in np.arange(sample_distance, curve_object.length, sample_distance):
        current_segment = next((x[0] for x in enumerate(curve_object.cdf) if x[1] >= current_distance), None)
        current_segment_length = curve_object.cdf[current_segment] - curve_object.cdf[current_segment - 1]

        previous_point = curve_object.points[current_segment - 1]
        next_point = curve_object.points[current_segment]

        current_distance_segment = current_distance - curve_object.cdf[current_segment - 1]
        ratio_of_distances = current_distance_segment / current_segment_length

        list_of_sample_points.append((1.0 - ratio_of_distances) * previous_point + ratio_of_distances * next_point)

    return np.array(list_of_sample_points)


def get_short_long_curve(a_curve, b_curve):
    """
    
    :param a_curve: 
    :param b_curve: 
    :return: 
    """
    if a_curve.length < b_curve.length:
        return a_curve, b_curve
    else:
        return b_curve, a_curve


def get_short_curve_char(a_curve, b_curve):
    """
    
    :param a_curve: 
    :param b_curve: 
    :return: 
    """
    if a_curve.length < b_curve.length:
        return 'a'
    else:
        return 'b'
