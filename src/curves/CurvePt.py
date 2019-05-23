from Vec import VecSE2
from Vec import VecE2
from Vec.geom import geom


class CurvePt:
    def __init__(self, pos: VecSE2.VecSE2, s: float, k=None, kd=None):
        self.pos = pos
        self.s = s
        self.k = k
        self.kd = kd

    def show(self):
        print("CurvePt({{:.3f}, {:.3f}, {:.3f}}, {:.3f}, {:.3f}, {:.3f})".format(self.pos.x, self.pos.y, self.pos.theta,
                                                                                 self.s, self.k, self.kd))


def lerp(a: CurvePt, b: CurvePt, t: float):
    return CurvePt(VecSE2.lerp(a.pos, b.pos, t), a.s + (b.s - a.s)*t, a.k + (b.k - a.k)*t, a.kd + (b.kd - a.kd)*t)


class CurveIndex:
    def __init__(self, i: int, t: float):
        self.i = i
        self.t = t

    def __eq__(self, other):
        return self.t == other.t and self.i == other.i

    def __ne__(self, other):
        return self.t != other.t or self.i != other.i


def curveindex_end(curve: list):
    return CurveIndex(len(curve)-2, 1.0)


def get_curve_list_by_index(curve: list, ind: CurveIndex):
    return lerp(curve[ind.i], curve[ind.i+1], ind.t)


CURVEINDEX_START = CurveIndex(0, 0.0)


class CurveProjection:
    def __init__(self, ind: CurveIndex, t: float, phi: float):
        self.ind = ind
        self.t = t
        self.phi = phi


def div(a, b):
    return int(a/b)


def index_closest_to_point(curve: list, target: VecSE2.VecSE2):  # curve: list(CurvePt)

    a = 1
    b = len(curve)
    c = div(a+b, 2)

    assert(len(curve) >= b)

    sqdist_a = curve[a - 1].pos - target
    sqdist_b = curve[b - 1].pos - target
    sqdist_c = curve[c - 1].pos - target

    sqdist_a = VecE2.normsquared(VecE2.VecE2(sqdist_a.x, sqdist_a.y))
    sqdist_b = VecE2.normsquared(VecE2.VecE2(sqdist_b.x, sqdist_b.y))
    sqdist_c = VecE2.normsquared(VecE2.VecE2(sqdist_c.x, sqdist_c.y))

    while True:
        if b == a:
            return a - 1
        elif b == a + 1:
            return (b - 1) if sqdist_b < sqdist_a else (a - 1)
        elif c == a + 1 and c == b - 1:
            if sqdist_a < sqdist_b and sqdist_a < sqdist_c:
                return a - 1
            elif sqdist_b < sqdist_a and sqdist_b < sqdist_c:
                return b - 1
            else:
                return c - 1

        left = div(a+c, 2)
        sqdist_l = curve[left - 1].pos - target
        sqdist_l = VecE2.normsquared(VecE2.VecE2(sqdist_l.x, sqdist_l.y))

        right = div(c+b, 2)
        sqdist_r = curve[right - 1].pos - target
        sqdist_r = VecE2.normsquared(VecE2.VecE2(sqdist_r.x, sqdist_r.y))

        if sqdist_l < sqdist_r:
            b = c
            sqdist_b = sqdist_c
            c = left
            sqdist_c = sqdist_l
        else:
            a = c
            sqdist_a = sqdist_c
            c = right
            sqdist_c = sqdist_r

    raise OverflowError("index_closest_to_point reached unreachable statement")


"""
    get_lerp_time_unclamped(A::VecE2, B::VecE2, Q::VecE2)
Get the interpolation scalar t for the point on the line AB closest to Q
This point is P = A + (B-A)*t
"""


def get_lerp_time_unclamped_1(A: VecE2.VecE2, B: VecE2.VecE2, Q: VecE2.VecE2):

    a = Q - A
    # A.show()
    # B.show()
    b = B - A
    c = VecE2.proj(a, b, VecE2.VecE2)

    if b.x != 0.0:
        t = c.x / b.x
    elif b.y != 0.0:
        t = c.y / b.y
    else:
        t = 0.0 # no lerping to be done

    return t


def get_lerp_time_unclamped_2(A: CurvePt, B: CurvePt, Q: VecSE2.VecSE2):
    return get_lerp_time_unclamped_1(A.pos.convert(), B.pos.convert(), Q.convert())


def get_lerp_time_unclamped_3(A: VecSE2.VecSE2, B: VecSE2.VecSE2, Q: VecSE2.VecSE2):
    return get_lerp_time_unclamped_1(A.convert(), B.convert(), Q.convert())


def clamp(a, low, high):
    return min(high, max(low, a))


def get_lerp_time_1(A: VecE2.VecE2, B: VecE2.VecE2, Q: VecE2.VecE2):
    return clamp(get_lerp_time_unclamped_1(A, B, Q), 0.0, 1.0)


def get_lerp_time_2(A: CurvePt, B: CurvePt, Q: VecSE2.VecSE2):
    return get_lerp_time_1(A.pos.convert(), B.pos.convert(), Q.convert())


def get_curve_projection(posG: VecSE2.VecSE2, footpoint: VecSE2.VecSE2, ind: CurveIndex):
    F = geom.inertial2body(posG, footpoint)
    return CurveProjection(ind, F.y, F.theta)


def proj(posG: VecSE2.VecSE2, curve: list):  # TODO: adjust list index
    ind = index_closest_to_point(curve, posG)
    curveind = CurveIndex(-1, 0)
    footpoint = VecSE2.VecSE2(0, 0, 0)
    if 0 < ind < len(curve) - 1:
        t_lo = get_lerp_time_2(curve[ind - 1], curve[ind], posG)
        t_hi = get_lerp_time_2(curve[ind], curve[ind + 1], posG)

        p_lo = VecSE2.lerp(curve[ind - 1].pos, curve[ind].pos, t_lo)
        p_hi = VecSE2.lerp(curve[ind].pos, curve[ind + 1].pos, t_hi)

        vec_lo = p_lo - posG
        vec_hi = p_hi - posG

        d_lo = VecE2.norm(VecE2.VecE2(vec_lo.x, vec_lo.y))
        d_hi = VecE2.norm(VecE2.VecE2(vec_hi.x, vec_hi.y))
        if d_lo < d_hi:
            footpoint = p_lo
            curveind = CurveIndex(ind - 1, t_lo)
        else:
            footpoint = p_hi
            curveind = CurveIndex(ind, t_hi)
    elif ind == 0:
        t = get_lerp_time_2(curve[0], curve[1], posG)
        footpoint = VecSE2.lerp(curve[0].pos, curve[1].pos, t)
        curveind = CurveIndex(ind, t)
    else:  # ind == length(curve)
        t = get_lerp_time_2(curve[-2], curve[-1], posG)
        footpoint = VecSE2.lerp(curve[-2].pos, curve[-1].pos, t)
        curveind = CurveIndex(ind - 1, t)

    return get_curve_projection(posG, footpoint, curveind)


