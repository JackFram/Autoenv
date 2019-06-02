import math
from src.Vec import VecSE2, VecE2


'''
deltaangle(a::Real, b::Real)
Return the minimum δ such that
    a + δ = mod(b, 2π)
'''


def deltaangle(a, b):
    assert isinstance(a, int) or isinstance(a, float)
    assert isinstance(b, int) or isinstance(b, float)

    return math.atan2(math.sin(b-a), math.cos(b-a))


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


def dot_product(a: VecE2.VecE2, b: VecE2.VecE2):
    return a.x*b.x + a.y*b.y


def cross_product(a: VecE2.VecE2, b: VecE2.VecE2):
    return a.x*b.y - a.y*b.x


def are_collinear(a: VecSE2.VecSE2, b: VecE2.VecE2, c: VecE2, tol: float=1e-8):
    # http://mathworld.wolfram.com/Collinear.html
    # if val = 0 then they are collinear
    val = a.x*(b.y-c.y) + b.x*(c.y-a.y)+c.x*(a.y-b.y)
    return abs(val) < tol