import math
from src.Vec import VecE2, VecSE2


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


def are_collinear(a: VecSE2.VecSE2, b: VecE2.VecE2, c: VecE2.VecE2, tol: float=1e-8):
    # http://mathworld.wolfram.com/Collinear.html
    # if val = 0 then they are collinear
    val = a.x*(b.y-c.y) + b.x*(c.y-a.y)+c.x*(a.y-b.y)
    return abs(val) < tol


def sign(a):
    if a == 0:
        return 0
    elif a > 0:
        return 1
    else:
        return -1

