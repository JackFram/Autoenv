# Zhihao Zhang
# VecE2 class in python

import math
from Vec.VecE2 import VecE2
from Vec.geom.geom import lerp_angle, clamp


class VecSE2:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = theta

    def show(self):
        print("VecSE2({:.3f}, {:.3f}, theta: {:.3f})".format(self.x, self.y, self.theta))

    def atan(self):
        return math.atan(self.y/self.x)

    def rot180(self):
        return VecSE2(self.x, self.y, self.theta + math.pi)

    def rotl90(self):
        return VecSE2(self.x, self.y, self.theta + 0.5*math.pi)

    def rotr90(self):
        return VecSE2(self.x, self.y, self.theta - 0.5*math.pi)

    def convert(self, t: type = VecE2):
        if t == VecE2:
            return VecE2(self.x, self.y)

    def mod2pi(self):
        return VecSE2(self.x, self.y, self.theta % (2*math.pi))

    def __add__(self, other):
        return VecSE2(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other):
        return VecSE2(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __radd__(self, other: VecE2):
        return VecSE2(self.x+other.x, self.y+other.y, self.theta)

    def __rsub__(self, other: VecE2):
        return VecSE2(self.x - other.x, self.y - other.y, self.theta)  # we have some problems here
    # Base.:-(a::VecSE2, b::VecE2) = VecSE2(a.x-b.x, a.y-b.y, a.θ)
    # Base.:-(a::VecE2,  b::VecSE2) = VecSE2(a.x-b.x, a.y-b.y, -b.θ)


def scale_euclidean(a: VecSE2, b):
    assert isinstance(b, int) or isinstance(b, float)
    return VecSE2(b*a.x, b*a.y, a.theta)


def clamp_euclidean(a: VecSE2, lo, hi):
    assert isinstance(lo, int) or isinstance(lo, float)
    assert isinstance(hi, int) or isinstance(hi, float)
    return VecSE2(clamp(a.x, lo, hi), clamp(a.y, lo, hi), a.theta)


def normaliza_euclidean(a: VecSE2, p=2):
    assert isinstance(p, int) or isinstance(p, float)
    n = math.sqrt(a.x*a.x + a.y*a.y)
    return VecSE2(a.x/n, a.y/n, a.theta)


def lerp(a: VecSE2, b: VecSE2, t):
    assert isinstance(t, int) or isinstance(t, float)
    x = a.x + (b.x-a.x)*t
    y = a.y + (b.y-a.y)*t
    theta = lerp_angle(a.theta, b.theta, t)
    return VecSE2(x, y, theta)


def rot(a: VecSE2, theta: float):
    '''
    rotate counter-clockwise about the origin
    :param a: VecSE2
    :param theta: rotate angle
    :return: VecSE2
    '''

    return VecSE2(a.x, a.y, a.theta + theta)







