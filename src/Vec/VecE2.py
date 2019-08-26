# Zhihao Zhang
# VecE2 class in python

import math

'''
VecE2 is a class to deal with 2 dimension vector
'''


class VecE2:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def show(self):
        print("VecE2({:.3f}, {:.3f})".format(self.x, self.y))

    def atan(self):
        if self.x == 0:
            if self.y > 0:
                return math.pi/2
            elif self.y < 0:
                return -math.pi/2
            else:
                return 0
        return math.atan(self.y/self.x)

    def hypot(self):
        return math.hypot(self.x, self.y)

    def rot180(self):
        return VecE2(-self.x, -self.y)

    def rotl90(self):
        return VecE2(-self.y, self.x)

    def rotr90(self):
        return VecE2(self.y, -self.x)

    def __sub__(self, other):
        return VecE2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return VecE2(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return VecE2(-self.x, -self.y)

    def __truediv__(self, scaler: int):
        return VecE2(self.x/scaler, self.y/scaler)


def polar(r, theta):
    return VecE2(r*math.cos(theta), r*math.sin(theta))


def dist(a: VecE2, b: VecE2):
    return math.hypot(a.x-b.x, a.y-b.y)


def dist2(a: VecE2, b: VecE2):
    return (a.x-b.x)**2 + (a.y-b.y)**2


def proj(a: VecE2, b: VecE2, t: type):
    if t == int:
        return (a.x*b.x + a.y*b.y) / math.hypot(b.x, b.y)  # dot(a,b)/|b|
    elif t == VecE2:
        # dot(a,b) / dot(b,b) ⋅ b
        s = (a.x*b.x + a.y*b.y) / (b.x*b.x + b.y*b.y)
        return VecE2(s*b.x, s*b.y)
    else:
        raise TypeError("Wrong type!")


def proj_(a: VecE2, b: VecE2):
    return (a.x * b.x + a.y * b.y) / math.hypot(b.x, b.y)  # dot(a,b)/|b|


def lerp(a: VecE2, b: VecE2, t):
    assert isinstance(t, int) or isinstance(t, float)
    return VecE2(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t)


def rot(a: VecE2, theta: float):
    '''
    rotate counter-clockwise about the origin
    :param a: VecE2
    :param theta: rotate angle
    :return: VecE2
    '''
    c = math.cos(theta)
    s = math.sin(theta)

    return VecE2(a.x*c - a.y*s, a.x*s+a.y*c)


def norm(v: VecE2):
    return math.sqrt(normsquared(v))


def normsquared(v: VecE2):
    return v.x*v.x + v.y*v.y







