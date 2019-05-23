import math
from Vec import VecSE2


def deltaangle(a, b):
    assert isinstance(a, int) or isinstance(a, float)
    assert isinstance(b, int) or isinstance(b, float)

    return math.atan(math.sin(b-a)/math.cos(b-a))


def lerp_angle(a, b, t):
    return a + deltaangle(a, b) * t


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def inertial2body(point, reference):

    '''
    Convert a point in an inertial cartesian coordinate frame
    to be relative to a body's coordinate frame
    The body's position is given relative to the same inertial coordinate frame
    '''

    s, c = math.sin(reference.theta), math.cos(reference.theta)
    delta_x = point.x - reference.x
    delta_y = point.y - reference.y
    return VecSE2.VecSE2(c*delta_x + s*delta_y, c*delta_y - s*delta_x, point.theta - reference.theta)
